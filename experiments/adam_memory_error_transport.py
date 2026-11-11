"""Project saved original Adam memory movements onto full-pool error gradients."""
import argparse
import csv
import importlib.util
import json
import unittest

import torch
from torch.nn import functional as F
import memory_gradient_competition as competition
import readout_adam_transport as replay_source

training=competition.training
ROOT=competition.ROOT
OUT=ROOT/'results/adam_memory_error_transport'
MEMORY=competition.NAMES
BLOCKS={**competition.BLOCKS,'head':('output.weight','output.bias'),
        'full':(*MEMORY,'output.weight','output.bias')}
SOURCES=('experiments/adam_memory_error_transport.py','experiments/audit_adam_memory_error_transport.py',
         'tests/test_adam_memory_error_transport.py')
PROTOCOL='docs/protocols/adam-memory-error-transport.md'


def model_from(state):
    with torch.random.fork_rng():model=training.model_source.LoadReader('delta');model.load_state_dict(state)
    return model.eval()


def fixed_metrics(z,y,mask):
    assert z.shape[:-1]==y.shape==mask.shape and torch.isfinite(z).all()
    assert ((y>=0)&(y<6)).all()
    losses=F.cross_entropy(z.reshape(-1,6),y.flatten(),reduction='none').reshape(y.shape)
    n=int(mask.sum());correct=z.argmax(-1)==y
    return dict(error_count=n,error_ce=float(losses[mask].mean()) if n else None,
        originally_wrong_now_correct=int((mask&correct).sum()),originally_correct_now_wrong=int((~mask&~correct).sum()))


def before_gradient(model,rows,batch=64):
    state={n:v.clone() for n,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    e,q,y,repeat=training.replay.pressure.prepare(rows,'all_queries')
    zs=[]
    with torch.no_grad():
        for i in range(0,len(rows),batch):zs.append(training.forward_queries(model,e[i:i+batch],q[i:i+batch]))
    z=torch.cat(zs);mask=z.argmax(-1)!=y;count=int(mask.sum())
    params=dict(model.named_parameters());gradients={n:torch.zeros_like(p) for n,p in params.items()}
    if count:
        for i in range(0,len(rows),batch):
            logits=training.forward_queries(model,e[i:i+batch],q[i:i+batch])
            losses=F.cross_entropy(logits.reshape(-1,6),y[i:i+batch].flatten(),reduction='none').reshape(y[i:i+batch].shape)
            gs=torch.autograd.grad(losses[mask[i:i+batch]].sum()/count,tuple(params.values()))
            for n,g in zip(params,gs):gradients[n]+=g.detach()
    assert all(torch.equal(v,model.state_dict()[n]) for n,v in state.items()) and torch.equal(rng,torch.get_rng_state())
    return dict(logits=z,labels=y,repeat=repeat,error_mask=mask,gradients=gradients,
                metrics=training.reduce_scores(z,y,repeat),fixed=fixed_metrics(z,y,mask))


def adam_parts(trace):
    group=trace['before_optimizer']['param_groups'][0]
    assert group==trace['after_optimizer']['param_groups'][0] and group['weight_decay']==0 and not group['amsgrad'] and not group['maximize']
    lr=group['lr'];b1,b2=group['betas'];eps=group['eps'];parts={p:{} for p in ('actual','stored','current','sgd')}
    for n,pid in zip(trace['names'],group['params']):
        g=trace['gradients'][n];old=trace['before_optimizer']['state'].get(pid,{})
        new=trace['after_optimizer']['state'][pid];t=int(new['step'])
        old_m=old.get('exp_avg',torch.zeros_like(g));den=(new['exp_avg_sq']/(1-b2**t)).sqrt()+eps
        parts['actual'][n]=trace['after_model'][n]-trace['before_model'][n]
        parts['stored'][n]=-lr*b1*old_m/((1-b1**t)*den)
        parts['current'][n]=-lr*(1-b1)*g/((1-b1**t)*den)
        parts['sgd'][n]=-lr*g
        assert float((parts['actual'][n]-parts['stored'][n]-parts['current'][n]).abs().max())<=1e-11,'Adam terms disagree'
    return parts


def projections(gradient,parts):
    def sign(v):return 'unresolved' if abs(v)<=1e-10 else ('higher_error_ce' if v>0 else 'lower_error_ce')
    records={}
    for block,names in BLOCKS.items():
        values={p:sum(float((gradient[n]*d[n]).sum()) for n in names) for p,d in parts.items()}
        assert abs(values['actual']-values['stored']-values['current'])<=1e-11,'Projection terms disagree'
        records[block]=dict(**values,actual_direction=sign(values['actual']),sgd_direction=sign(values['sgd']))
    return records


def probes(model,rows,before,parts):
    params=dict(model.named_parameters());base={n:p.detach().clone() for n,p in params.items()}
    norm=float(competition.flatten(parts['actual'],MEMORY).norm())
    if norm<=1e-10 or not bool(before['error_mask'].any()):return dict(skipped=True,norm=norm),0
    e,q,y,_=training.replay.pressure.prepare(rows,'all_queries');checks=[]
    analytic=sum(float((before['gradients'][n]*parts['actual'][n]).sum()) for n in MEMORY)/norm
    try:
        for h in (1e-5,5e-6):
            values=[]
            for sign in (1,-1):
                with torch.no_grad():
                    for n in MEMORY:params[n].copy_(base[n]+sign*h*parts['actual'][n]/norm)
                values.append(competition.error_loss(model,e,q,y,before['error_mask']))
            slope=(values[0]-values[1])/(2*h)
            assert abs(slope-analytic)<=1e-7+1e-4*abs(analytic),'Finite difference disagrees'
            checks.append(dict(h=h,plus_loss=values[0],minus_loss=values[1],derivative=slope))
    finally:
        with torch.no_grad():
            for n,p in params.items():p.copy_(base[n])
    assert all(torch.equal(base[n],p) for n,p in params.items())
    return dict(skipped=False,norm=norm,analytic=analytic,checks=checks),4*y.numel()


def evaluate_trace(trace,rows):
    model=model_from(trace['before_model']);before=before_gradient(model,rows);conditions={'before':before};parts=adam_parts(trace)
    mixed={n:trace['after_model'][n] if n in MEMORY else v for n,v in trace['before_model'].items()}
    for name,state in (('memory_after',mixed),('full_after',trace['after_model'])):
        cell=training.evaluate(model_from(state),rows)
        assert torch.equal(cell['labels'],before['labels']) and torch.equal(cell['repeat'],before['repeat'])
        cell['fixed']=fixed_metrics(cell['logits'],cell['labels'],before['error_mask']);conditions[name]=cell
    probe,answers=probes(model,rows,before,parts)
    record={k:trace[k] for k in ('seed','start','step','load')}
    record.update(blocks=projections(before['gradients'],parts),probe=probe,
        conditions={k:{field:v[field] for field in ('metrics','fixed')} for k,v in conditions.items()})
    return dict(record=record,conditions=conditions,parts=parts),answers


def inputs():
    traces=torch.load(replay_source.OUT/'traces.pt',weights_only=True)
    payload=json.loads((training.parent.OUT/'development_data.json').read_text())
    return traces,payload['pools']['train_32']


def preflight():
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('test_adam_memory_error_transport',ROOT/SOURCES[-1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    record=dict(passed=result.wasSuccessful(),methods=result.testsRun,sources={n:training.replay.digest(ROOT/n) for n in SOURCES})
    path=ROOT/'results'/('adam_memory_error_transport_preflight'+('' if record['passed'] else '_failed')+'.json')
    assert not path.exists();training.replay.write(path,record);assert record['passed']


def run():
    torch.set_num_threads(1);assert not OUT.exists(),'Preserve evidence'
    ready_path=ROOT/'results/adam_memory_error_transport_preflight.json';ready=json.loads(ready_path.read_text())
    assert ready['passed'] and ready['sources']=={n:training.replay.digest(ROOT/n) for n in SOURCES}
    files=dict(json.loads((competition.OUT/'lock.json').read_text())['files'])
    for folder in (competition.OUT,replay_source.OUT):
        report=json.loads((folder/'report.json').read_text());audit=json.loads((folder/'audit.json').read_text())
        assert audit['passed'] and audit['report_sha256']==training.replay.digest(folder/'report.json')
        for name in ('report.json','audit.json','lock.json','metrics.csv'):
            p=folder/name;files[p.relative_to(ROOT).as_posix()]=training.replay.digest(p)
    archived=json.loads((replay_source.OUT/'report.json').read_text())
    assert archived['traces_sha256']==training.replay.digest(replay_source.OUT/'traces.pt')
    path=replay_source.OUT/'traces.pt';files[path.relative_to(ROOT).as_posix()]=training.replay.digest(path)
    for name in (*SOURCES,PROTOCOL,ready_path.relative_to(ROOT).as_posix()):files[name]=training.replay.digest(ROOT/name)
    training.replay.check_files(files);OUT.mkdir()
    training.replay.write(OUT/'lock.json',dict(files=files,seeds=[3461,3463],starts=[3000,11000],cycle_steps=4,
        pool='train_32',conditions=['before','memory_after','full_after'],cells=16,baseline_answers=393216,
        condition_answers=1179648,probe_spacings=[1e-5,5e-6],projection_resolution=1e-10))
    traces,rows=inputs();saved=[];probe_answers=0
    assert [(t['seed'],t['start'],t['step']) for t in traces]==[(s,a,t) for s in (3461,3463) for a in (3000,11000) for t in range(a+1,a+5)]
    for trace in traces:
        cell,answers=evaluate_trace(trace,rows);saved.append(cell);probe_answers+=answers
        print(f"Evaluated seed {trace['seed']} step {trace['step']}",flush=True)
    assert len(rows)==768 and all(len(r['events'])==32 and r['load']==32 for r in rows)
    assert sum(c['conditions']['before']['labels'].numel() for c in saved)==393216 and probe_answers<=1572864
    torch.save(saved,OUT/'scores.pt')
    report=dict(status='complete',training_updates=0,optimizer_calls=0,reused_replay_steps=16,pool='train_32',device='cpu',dtype='float64',
        cells=[c['record'] for c in saved],baseline_answers=393216,condition_answers=1179648,probe_answers=probe_answers,
        scores_sha256=training.replay.digest(OUT/'scores.pt'),lock_sha256=training.replay.digest(OUT/'lock.json'))
    training.replay.check_files(files);training.replay.write(OUT/'report.json',report)
    fields=('seed','start','step','load','block','actual','stored','current','sgd','actual_direction','sgd_direction')
    with (OUT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for c in report['cells']:
            for b,v in c['blocks'].items():w.writerow(dict(**{k:c[k] for k in ('seed','start','step','load')},block=b,**v))
    print(json.dumps({k:v for k,v in report.items() if k!='cells'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true')
    if p.parse_args().preflight:preflight()
    else:run()
