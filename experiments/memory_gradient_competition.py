"""Correct/error gradient contributions to the existing equal-load objective."""
import argparse
import csv
import importlib.util
import json
import unittest

import torch
from torch.nn import functional as F
import target_only_history as clean

training=clean.training
ROOT=clean.ROOT
OUT=ROOT/'results/memory_gradient_competition'
NAMES=('key.weight','value.weight','write_logit')
BLOCKS={'keys':('key.weight',),'values':('value.weight',),'alpha':('write_logit',),'memory':NAMES}
SOURCES=('experiments/memory_gradient_competition.py','experiments/audit_memory_gradient_competition.py',
         'tests/test_memory_gradient_competition.py')
PROTOCOL='docs/protocols/memory-gradient-competition.md'


def partitions(logits,labels):
    assert logits.shape[:-1]==labels.shape and logits.shape[-1]==6 and torch.isfinite(logits).all()
    assert ((labels>=0)&(labels<6)).all()
    correct=logits.argmax(-1)==labels
    losses=F.cross_entropy(logits.reshape(-1,6),labels.flatten(),reduction='none').reshape(labels.shape)
    result={}
    for name,mask in (('correct',correct),('error',~correct)):
        n=int(mask.sum());result[name]=dict(count=n,ce=float(losses[mask].mean()) if n else None,
            weighted_loss=float(losses[mask].sum())/(4*labels.numel()))
    return result


def cell(model,rows,source=None,batch=64):
    state={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    for name,p in model.named_parameters():p.requires_grad_(name in NAMES)
    params=tuple(dict(model.named_parameters())[n] for n in NAMES)
    events,q,y,repeat=training.replay.pressure.prepare(rows,'all_queries')
    grads={part:{n:torch.zeros_like(p) for n,p in zip(NAMES,params)} for part in ('correct','error','total')};zs=[]
    for start in range(0,len(rows),batch):
        z=training.forward_queries(model,events[start:start+batch],q[start:start+batch]);labels=y[start:start+batch]
        losses=F.cross_entropy(z.reshape(-1,6),labels.flatten(),reduction='none')
        correct=(z.detach().argmax(-1)==labels).flatten()
        objectives=(losses[correct].sum(),losses[~correct].sum(),losses.sum())
        for j,(part,loss) in enumerate(zip(grads,objectives)):
            gs=torch.autograd.grad(loss/(4*y.numel()),params,retain_graph=j<2)
            for n,g in zip(NAMES,gs):grads[part][n]+=g.detach()
        zs.append(z.detach())
    logits=torch.cat(zs)
    for n in NAMES:torch.testing.assert_close(grads['correct'][n]+grads['error'][n],grads['total'][n],atol=1e-12,rtol=1e-9)
    if source is not None:
        assert torch.equal(y,source['labels']) and torch.equal(repeat,source['repeat']), 'Saved labels disagree'
        assert float((logits-source['logits']).abs().max())<=1e-9 and torch.equal(logits.argmax(-1),source['logits'].argmax(-1))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()) and torch.equal(rng,torch.get_rng_state())
    return dict(logits=logits,labels=y,repeat=repeat,metrics=training.reduce_scores(logits,y,repeat),
                partitions=partitions(logits,y),gradients=grads)


def pooled(cells):
    assert len(cells)==4 and {c['load'] for c in cells}=={2,8,16,32}, 'Missing load'
    chosen=next(c for c in cells if c['load']==32);counts=chosen['partitions']['error']['count']
    answer_count=chosen['labels'].numel()
    gradients={part:{} for part in ('correct','error32','other_errors','total','conditional_error32')}
    for n in NAMES:
        gradients['correct'][n]=sum(c['gradients']['correct'][n] for c in cells)
        gradients['error32'][n]=chosen['gradients']['error'][n]
        gradients['other_errors'][n]=sum(c['gradients']['error'][n] for c in cells if c['load']!=32)
        gradients['total'][n]=sum(c['gradients']['total'][n] for c in cells)
        gradients['conditional_error32'][n]=chosen['gradients']['error'][n]*(4*answer_count/counts) if counts else torch.zeros_like(chosen['gradients']['error'][n])
        torch.testing.assert_close(sum(gradients[p][n] for p in ('correct','error32','other_errors')),gradients['total'][n],atol=1e-12,rtol=1e-9)
    return gradients


def flatten(gradient,names):return torch.cat([gradient[n].reshape(-1) for n in names])


def geometry(gradients):
    records={}
    for block,names in BLOCKS.items():
        c,e,o,t,f=(flatten(gradients[p],names) for p in ('correct','error32','other_errors','total','conditional_error32'))
        nc,ne,nt=float(c.norm()),float(e.norm()),float(t.norm())
        def alignment(x,nx):
            cosine=float(x@e)/(nx*ne) if min(nx,ne)>1e-10 else None
            sign='unresolved' if cosine is None or abs(cosine)<=1e-6 else ('opposed' if cosine<0 else 'aligned')
            return cosine,sign
        cos,sign=alignment(c,nc);total_cos,total_sign=alignment(t,nt)
        records[block]=dict(dimensions=c.numel(),correct_norm=nc,error32_norm=ne,other_error_norm=float(o.norm()),total_norm=nt,
            correct_error_cosine=cos,correct_alignment=sign,total_error_cosine=total_cos,total_alignment=total_sign,
            error_ce_direction_correct=float(f@(-c)),error_ce_direction_total=float(f@(-t)),
            error_ce_direction_own=float(f@(-e)),error_ce_direction_other=float(f@(-o)))
    return records


def error_loss(model,events,q,y,mask,batch=64):
    assert int(mask.sum())>0;total=0.
    with torch.no_grad():
        for start in range(0,len(events),batch):
            z=training.forward_queries(model,events[start:start+batch],q[start:start+batch])
            losses=F.cross_entropy(z.reshape(-1,6),y[start:start+batch].flatten(),reduction='none').reshape(y[start:start+batch].shape)
            total+=float(losses[mask[start:start+batch]].sum())
    return total/int(mask.sum())


def probes(model,rows,source,gradients):
    events,q,y,_=training.replay.pressure.prepare(rows,'all_queries');mask=source['logits'].argmax(-1)!=y
    if not bool(mask.any()):return [],0
    params=dict(model.named_parameters());base={n:p.detach().clone() for n,p in params.items()};records=[];answers=0
    try:
        for block,names in BLOCKS.items():
            for part in ('correct','total'):
                g=flatten(gradients[part],names);norm=float(g.norm())
                if norm<=1e-10:
                    records.append(dict(block=block,part=part,skipped=True,norm=norm));continue
                directions={n:-gradients[part][n]/norm for n in names}
                analytic=sum(float((gradients['conditional_error32'][n]*directions[n]).sum()) for n in names)
                checks=[]
                for h in (1e-5,5e-6):
                    values=[]
                    for sign in (1,-1):
                        with torch.no_grad():
                            for n in names:params[n].copy_(base[n]+sign*h*directions[n])
                        values.append(error_loss(model,events,q,y,mask));answers+=y.numel()
                    derivative=(values[0]-values[1])/(2*h)
                    assert abs(derivative-analytic)<=1e-7+1e-4*abs(analytic), 'Directional finite difference disagrees'
                    checks.append(dict(h=h,plus_loss=values[0],minus_loss=values[1],derivative=derivative))
                records.append(dict(block=block,part=part,skipped=False,norm=norm,analytic=analytic,checks=checks))
                with torch.no_grad():
                    for n in names:params[n].copy_(base[n])
    finally:
        with torch.no_grad():
            for n in params:params[n].copy_(base[n])
    assert all(torch.equal(base[n],p) for n,p in params.items()), 'Probe restoration failed'
    return records,answers


def inputs():return [(source,rows) for source,rows in clean.inputs() if source['pool'].startswith('train_')]


def preflight():
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('test_memory_gradient_competition',ROOT/SOURCES[-1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    record=dict(passed=result.wasSuccessful(),methods=result.testsRun,sources={n:training.replay.digest(ROOT/n) for n in SOURCES})
    path=ROOT/'results'/('memory_gradient_competition_preflight'+('' if record['passed'] else '_failed')+'.json')
    assert not path.exists();training.replay.write(path,record);assert record['passed']


def run():
    torch.set_num_threads(1);assert not OUT.exists(), 'Preserve evidence'
    ready_path=ROOT/'results/memory_gradient_competition_preflight.json';ready=json.loads(ready_path.read_text())
    assert ready['passed'] and ready['sources']=={n:training.replay.digest(ROOT/n) for n in SOURCES}
    files=dict(json.loads((clean.OUT/'lock.json').read_text())['files']);prior=json.loads((clean.OUT/'audit.json').read_text())
    assert prior['passed'] and prior['report_sha256']==training.replay.digest(clean.OUT/'report.json')
    for name in ('report.json','audit.json','lock.json','metrics.csv'):
        path=clean.OUT/name;files[path.relative_to(ROOT).as_posix()]=training.replay.digest(path)
    for name in (*SOURCES,PROTOCOL,ready_path.relative_to(ROOT).as_posix()):files[name]=training.replay.digest(ROOT/name)
    training.replay.check_files(files);OUT.mkdir()
    training.replay.write(OUT/'lock.json',dict(files=files,seeds=[3461,3463],steps=[3000,12000],loads=[2,8,16,32],
        split='train',cells=16,primary_answers=178176,blocks=BLOCKS,load_weight=.25,mask_policy='fixed saved argmax',
        cosine_resolution=1e-6,norm_resolution=1e-10,probe_spacings=[1e-5,5e-6]))
    pairs=inputs();saved=[];records=[];objectives=[];probe_answers=0
    for seed in (3461,3463):
        for step in (3000,12000):
            cells=[];model=None;last_rows=None;last_source=None
            for source,rows in pairs:
                if (source['seed'],source['step'])!=(seed,step):continue
                model=training.previous.load_model(ROOT/source['checkpoint']);data=cell(model,rows,source)
                metadata={k:source[k] for k in ('seed','step','pool','checkpoint')};metadata['load']=int(source['pool'].split('_')[1])
                data.update(metadata);cells.append(data);saved.append(data)
                records.append(dict(**metadata,metrics=data['metrics'],partitions=data['partitions']))
                if data['load']==32:last_rows=rows;last_source=source
            gradients=pooled(cells);probe,targets=probes(model,last_rows,last_source,gradients);probe_answers+=targets
            objectives.append(dict(seed=seed,step=step,blocks=geometry(gradients),probes=probe,
                error32_count=next(c for c in cells if c['load']==32)['partitions']['error']['count'],
                other_error_count=sum(c['partitions']['error']['count'] for c in cells if c['load']!=32)))
            print(f'Completed seed {seed} step {step}',flush=True)
    assert len(saved)==16 and sum(c['labels'].numel() for c in saved)==2*2*768*sum((2,8,16,32))==178176
    assert probe_answers<=3145728
    torch.save(saved,OUT/'gradients.pt')
    report=dict(status='complete',training_updates=0,optimizer_calls=0,split='train',device='cpu',dtype='float64',
        cells=records,objectives=objectives,cells_count=16,primary_answers=178176,probe_answers=probe_answers,
        gradients_sha256=training.replay.digest(OUT/'gradients.pt'),lock_sha256=training.replay.digest(OUT/'lock.json'))
    training.replay.check_files(files);training.replay.write(OUT/'report.json',report)
    fields=('seed','step','block',*next(iter(objectives[0]['blocks'].values())).keys())
    with (OUT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for obj in objectives:
            for block,values in obj['blocks'].items():w.writerow(dict(seed=obj['seed'],step=obj['step'],block=block,**values))
    print(json.dumps({k:v for k,v in report.items() if k not in ('cells','objectives')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true')
    if p.parse_args().preflight:preflight()
    else:run()
