"""Training-objective pressure on the classifier's radial direction."""
import argparse
import csv
import importlib.util
import json
import unittest

import torch
from torch.nn import functional as F
import readout_scale_control as scale

training=scale.training
ROOT=scale.ROOT
OUT=ROOT/'results/readout_scale_pressure'
STEPS=tuple(range(3000,12001,1000))
LOADS=(2,8,16,32)
SEEDS=(3461,3463)
SOURCES=('experiments/readout_scale_pressure.py','experiments/audit_readout_scale_pressure.py',
         'tests/test_readout_scale_pressure.py')
PROTOCOL='docs/protocols/readout-scale-pressure.md'


def derivative(logits,labels):
    z=logits.reshape(-1,6);y=labels.flatten()
    assert logits.shape[:-1]==labels.shape and torch.isfinite(z).all()
    assert ((y>=0)&(y<6)).all()
    gaps=z-z.gather(-1,y[:,None])
    return (z.softmax(-1)*gaps).sum(-1)


def summarize(cell):
    z=cell['logits'];y=cell['labels'];repeat=cell['repeat'];assert y.shape==repeat.shape
    g=derivative(z,y);correct=z.argmax(-1).flatten()==y.flatten()
    ce=F.cross_entropy(z.reshape(-1,6),y.flatten(),reduction='none');groups={}
    for name in ('all','unique','repeat'):
        scope=torch.ones_like(correct) if name=='all' else repeat.flatten()==(name=='repeat')
        n=int(scope.sum());parts={}
        for part,is_correct in (('correct',True),('error',False)):
            chosen=scope&(correct==is_correct);count=int(chosen.sum())
            parts[part]=dict(count=count,ce=float(ce[chosen].mean()) if count else None,
                mean_g=float(g[chosen].mean()) if count else None,
                contribution=float(g[chosen].sum())/n if n else 0.)
        groups[name]=dict(count=n,accuracy=float(correct[scope].double().mean()) if n else None,
            ce=float(ce[scope].mean()) if n else None,mean_g=float(g[scope].mean()) if n else None,partitions=parts)
        assert sum(p['count'] for p in parts.values())==n
        if n:assert abs(sum(p['contribution'] for p in parts.values())-groups[name]['mean_g'])<=1e-9
    return groups


def objective(cells):
    assert len(cells)==4 and {c['load'] for c in cells}==set(LOADS),'Missing objective load'
    return dict(ce=sum(c['groups']['all']['ce'] for c in cells)/4,
        mean_g=sum(c['groups']['all']['mean_g'] for c in cells)/4,
        correct_contribution=sum(c['groups']['all']['partitions']['correct']['contribution'] for c in cells)/4,
        error_contribution=sum(c['groups']['all']['partitions']['error']['contribution'] for c in cells)/4)


def score_checks(cell):
    z=cell['logits'].detach();centered=z-z.mean(-1,keepdim=True);y=cell['labels'].flatten()
    t=torch.zeros((),dtype=z.dtype,requires_grad=True)
    loss=F.cross_entropy((z.mean(-1,keepdim=True)+t.exp()*centered).reshape(-1,6),y)
    grad=float(torch.autograd.grad(loss,t)[0]);reference=float(derivative(z,cell['labels']).mean())
    assert abs(grad-reference)<=1e-9+1e-7*abs(reference),'Scalar autograd disagreement'
    finite={}
    for h in (1e-4,5e-5):
        a=F.cross_entropy((z.mean(-1,keepdim=True)+torch.tensor(h,dtype=z.dtype).exp()*centered).reshape(-1,6),y)
        b=F.cross_entropy((z.mean(-1,keepdim=True)+torch.tensor(-h,dtype=z.dtype).exp()*centered).reshape(-1,6),y)
        finite[str(h)]=float((a-b)/(2*h))
        assert abs(finite[str(h)]-reference)<=1e-8+1e-5*abs(reference),'Finite difference disagreement'
    return dict(autograd_g=grad,finite_differences=finite)


def model_gradient(checkpoint,rows,cell,batch=64):
    model=training.previous.load_model(checkpoint)
    before=training.parent.study.parent.state_hash(model.state_dict());rng=torch.get_rng_state().clone()
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('output.'))
    events,q,y,_=training.replay.pressure.prepare(rows,'all_queries');parameters=(model.output.weight,model.output.bias)
    grads=[torch.zeros_like(p) for p in parameters];logits=[]
    for i in range(0,len(rows),batch):
        z=training.forward_queries(model,events[i:i+batch],q[i:i+batch])
        loss=F.cross_entropy(z.reshape(-1,6),y[i:i+batch].flatten(),reduction='sum')/y.numel()
        gs=torch.autograd.grad(loss,parameters)
        for j,g in enumerate(gs):grads[j]+=g.detach()
        logits.append(z.detach())
    actual=torch.cat(logits);error=float((actual-cell['logits']).abs().max())
    assert error<=1e-9 and torch.equal(actual.argmax(-1),cell['logits'].argmax(-1)),'Saved logits disagree'
    radial=sum(float((g*(p.detach()-p.detach().mean(0,keepdim=True))).sum()) for g,p in zip(grads,parameters))
    reference=float(derivative(cell['logits'],cell['labels']).mean())
    assert abs(radial-reference)<=1e-9+1e-7*abs(reference),'Head gradient disagreement'
    assert before==training.parent.study.parent.state_hash(model.state_dict()) and torch.equal(rng,torch.get_rng_state())
    return dict(head_gradient_g=radial,max_logit_error=error,head_norm=scale.head_norm(model.state_dict()))


def inputs():
    folder=scale.previous.parent.OUT
    report=json.loads((folder/'report.json').read_text());scores=torch.load(folder/'scores.pt',weights_only=True)
    payload=json.loads((training.parent.OUT/'development_data.json').read_text())
    chosen=[]
    for seed in SEEDS:
        for step in STEPS:
            for load in LOADS:
                pool=f'train_{load}'
                cell=next(c for c in scores if (c['seed'],c['step'],c['pool'])==(seed,step,pool))
                entry=next(c for c in report['cells'] if (c['seed'],c['step'],c['pool'])==(seed,step,pool))
                assert cell['arm']=='exhaustive' and entry=={k:cell[k] for k in ('seed','arm','step','pool','checkpoint','metrics')}
                chosen.append((cell,payload['pools'][pool]))
    return chosen


def preflight():
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('test_readout_scale_pressure',ROOT/SOURCES[-1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    record=dict(passed=result.wasSuccessful(),methods=result.testsRun,sources={n:training.replay.digest(ROOT/n) for n in SOURCES})
    p=ROOT/'results'/('readout_scale_pressure_preflight.json' if record['passed'] else 'readout_scale_pressure_preflight_failed.json')
    assert not p.exists();training.replay.write(p,record);assert record['passed']


def run():
    torch.set_num_threads(1);assert not OUT.exists(),'Preserve evidence'
    ready_path=ROOT/'results/readout_scale_pressure_preflight.json';ready=json.loads(ready_path.read_text())
    assert ready['passed'] and ready['sources']=={n:training.replay.digest(ROOT/n) for n in SOURCES}
    files=dict(json.loads((scale.OUT/'lock.json').read_text())['files'])
    audit=json.loads((scale.OUT/'audit.json').read_text())
    assert audit['passed'] and audit['report_sha256']==training.replay.digest(scale.OUT/'report.json')
    assert audit['corrections_sha256']==training.replay.digest(scale.OUT/'threshold-corrections.json')
    for n in ('report.json','audit.json','threshold-corrections.json','scales.json','lock.json'):
        p=scale.OUT/n;files[p.relative_to(ROOT).as_posix()]=training.replay.digest(p)
    for n in (*SOURCES,PROTOCOL,ready_path.relative_to(ROOT).as_posix()):files[n]=training.replay.digest(ROOT/n)
    training.replay.check_files(files);OUT.mkdir()
    training.replay.write(OUT/'lock.json',dict(files=files,seeds=list(SEEDS),steps=list(STEPS),loads=list(LOADS),split='train',sign_resolution=1e-10))
    cells=[]
    for cell,rows in inputs():
        groups=summarize(cell);checks=score_checks(cell);native=model_gradient(ROOT/cell['checkpoint'],rows,cell)
        for name,g in groups.items():
            m=cell['metrics']['groups'][name];assert g['count']==m['count'] and g['accuracy']==m['accuracy']
            if m['ce'] is not None:assert abs(g['ce']-m['ce'])<=1e-9
        cells.append(dict(seed=cell['seed'],step=cell['step'],load=int(cell['pool'].split('_')[1]),
            checkpoint=cell['checkpoint'],groups=groups,**checks,**native))
    objectives=[]
    for seed in SEEDS:
        for step in STEPS:
            selected=[c for c in cells if (c['seed'],c['step'])==(seed,step)]
            assert len({c['head_norm'] for c in selected})==1
            objectives.append(dict(seed=seed,step=step,head_norm=selected[0]['head_norm'],**objective(selected)))
    answers=sum(c['groups']['all']['count'] for c in cells)
    assert len(cells)==80 and answers==len(SEEDS)*len(STEPS)*768*sum(LOADS)==890880
    report=dict(status='complete',training_updates=0,split='train',device='cpu',dtype='float64',cells_count=80,
        evaluated_answers=answers,cells=cells,objectives=objectives,lock_sha256=training.replay.digest(OUT/'lock.json'))
    training.replay.check_files(files);training.replay.write(OUT/'report.json',report)
    with (OUT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=('seed','step','load','group','partition','count','ce','mean_g','contribution'));writer.writeheader()
        for c in cells:
            for group,g in c['groups'].items():
                for name,p in g['partitions'].items():writer.writerow(dict(seed=c['seed'],step=c['step'],load=c['load'],group=group,partition=name,**p))
    print(json.dumps(dict(cells=80,evaluated_answers=answers,training_updates=0,objectives=objectives)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    if parser.parse_args().preflight:preflight()
    else:run()
