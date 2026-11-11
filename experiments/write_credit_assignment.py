"""Fixed-weight, zero-start temporal gradients; no optimizer or oracle state."""
import argparse
import csv
import importlib.util
import json
import time
import unittest

import torch
from torch.nn import functional as F
import shared_state_feasibility as prior

ROOT=prior.ROOT
OUT=ROOT/'results/write_credit_assignment'
PROTOCOL='docs/protocols/write-credit-assignment.md'
SOURCES=('experiments/write_credit_assignment.py','experiments/audit_write_credit_assignment.py',
         'tests/test_write_credit_assignment.py')
GROUPS=('correct','error','total')


def ratio(a,b):return a/b if b>0 else None


def records(bundle):
    rows=[];summary=[]
    for group in GROUPS:
        g=bundle['gradients'][group];sg=g['states'];vg=g['events'];pooled=g['positions']
        last=float(sg[-1].norm());count=int(bundle['masks'][group].sum())
        position_norms=pooled.flatten(1).norm(dim=1)
        for t in range(len(sg)):
            norm=float(sg[t].norm())
            row=dict(seed=bundle['seed'],group=group,step=t,queries=count,state_norm=norm,
                     state_fraction_of_final=ratio(norm,last),event_norm=None,position_norm=None,history_sum_ratio=None)
            if t:
                event=float(vg[:,t-1].norm());position=float(position_norms[t-1])
                # Triangle bound uses sum of per-history event norms, not Frobenius norm.
                denominator=float(vg[:,t-1].norm(dim=-1).sum())
                row.update(event_norm=event,position_norm=position,history_sum_ratio=ratio(position,denominator))
            rows.append(row)
        denominator=float(position_norms.sum());joint=float(g['embedding'].norm())
        summary.append(dict(seed=bundle['seed'],group=group,queries=count,
            state_initial_fraction=ratio(float(sg[0].norm()),last),state_first_fraction=ratio(float(sg[1].norm()),last),
            final_state_norm=last,embedding_norm=joint,position_norm_sum=denominator,
            temporal_sum_ratio=ratio(joint,denominator),
            first_event_norm=float(vg[:,0].norm()),last_event_norm=float(vg[:,-1].norm())))
    return rows,summary


def measure(model,rows,seed):
    before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    q,y,repeat=prior.labels(rows);e=torch.tensor([r['events'] for r in rows])
    k=model.keys(e[...,0]);v=model.value(e[...,1]);beta=model.write_logit.sigmoid()
    s=torch.zeros(len(rows),16,8,dtype=torch.float64,requires_grad=True);states=[s]
    for t in range(e.shape[1]):
        key=k[:,t];s=s+beta*key[:,:,None]*(v[:,t]-(key[:,:,None]*s).sum(1))[:,None,:];states.append(s)
    z=model.output(torch.einsum('hmd,hdv->hmv',model.keys(q),s))
    with torch.no_grad():
        native=prior.historical.parent.forward_queries(model,e,q)
        assert torch.allclose(native,z,atol=1e-9,rtol=1e-9)
        assert torch.allclose(model.prefill(e)[0],s,atol=1e-9,rtol=1e-9)
    correct=z.detach().argmax(-1)==y
    masks=dict(correct=correct,error=~correct,total=torch.ones_like(correct))
    losses=F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='none').reshape_as(y)
    gradients={}
    for group,mask in masks.items():
        loss=(losses*mask).sum()/(4*y.numel())
        values=torch.autograd.grad(loss,(*states,v,model.value.weight),retain_graph=True)
        sg=torch.stack(values[:-2]).detach();vg=values[-2].detach();embedding=values[-1].detach()
        positions=torch.zeros(e.shape[1],6,8,dtype=torch.float64)
        for t in range(e.shape[1]):positions[t].index_add_(0,e[:,t,1],vg[:,t])
        assert torch.allclose(positions.sum(0),embedding,atol=1e-9,rtol=1e-9)
        gradients[group]=dict(states=sg,events=vg,positions=positions,embedding=embedding)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert torch.equal(rng,torch.get_rng_state()) and all(p.grad is None for p in model.parameters())
    return dict(seed=seed,queries=q,labels=y,repeat=repeat,logits=z.detach(),
                metrics=prior.metrics(z.detach(),y,repeat),masks=masks,gradients=gradients)


def execute(out,cases,fixture=False):
    assert not out.exists(),'Preserve evidence'
    files={p:prior.digest(ROOT/p) for p in (*SOURCES,PROTOCOL,'experiments/shared_state_feasibility.py')}
    for case in cases:
        for key in ('checkpoint','data'):files[case[key]]=prior.digest(ROOT/case[key])
    if not fixture:
        pre=json.loads((ROOT/'results/write_credit_assignment_preflight_v2.json').read_text())
        assert pre['passed'] and pre['sources']=={p:prior.digest(ROOT/p) for p in SOURCES}
        for name in ('shared_state_preservation','shared_state_feasibility'):
            source=ROOT/'results'/name;report=json.loads((source/'report.json').read_text())
            audit=json.loads((source/'audit.json').read_text());assert audit['passed'] and audit['report_sha256']==prior.digest(source/'report.json')
            for file in ('lock.json','report.json','audit.json'):
                p=source/file;files[p.relative_to(ROOT).as_posix()]=prior.digest(p)
            files.update(json.loads((source/'lock.json').read_text())['files'])
    prior.check_files(files);out.mkdir(parents=True)
    prior.write(out/'lock.json',dict(cases=cases,fixture=fixture,files=files,versions=prior.versions(),
        weighting='masked sum CE / (4 * histories * queries)',threads=torch.get_num_threads()))
    start=time.perf_counter();bundles=[];all_rows=[];summary=[]
    for case in cases:
        payload=json.loads((ROOT/case['data']).read_text(encoding='utf-8-sig'));rows=payload['pools'][case['pool']]
        model=prior.historical.parent.previous.load_model(ROOT/case['checkpoint']).eval()
        result=measure(model,rows,case['seed']);bundles.append(result)
        position_rows,stats=records(result);all_rows.extend(position_rows);summary.extend(stats)
        print(json.dumps(dict(seed=case['seed'],metrics=result['metrics'],summary=stats)),flush=True)
    torch.save(bundles,out/'gradients.pt')
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    answers=sum(b['labels'].numel() for b in bundles)
    report=dict(status='complete',fixture=fixture,training_updates=0,optimizer_calls=0,device='cpu',dtype='float64',
        unique_answers=answers,diagnostic_writes=sum(len(b['labels'])*(len(b['gradients']['total']['states'])-1) for b in bundles),
        native_fidelity_writes=2*sum(len(b['labels'])*(len(b['gradients']['total']['states'])-1) for b in bundles),
        state_rows=len(all_rows),event_positions=sum(len(b['gradients']['total']['states'])-1 for b in bundles)*len(GROUPS),
        metrics=[dict(seed=b['seed'],metrics=b['metrics']) for b in bundles],summary=summary,
        seconds=time.perf_counter()-start,lock_sha256=prior.digest(out/'lock.json'),
        gradients_sha256=prior.digest(out/'gradients.pt'),metrics_sha256=prior.digest(out/'metrics.csv'))
    prior.check_files(files);prior.write(out/'report.json',report)


def preflight():
    spec=importlib.util.spec_from_file_location('test_write_credit_assignment',ROOT/SOURCES[-1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if result.wasSuccessful():
        p=ROOT/'results/write_credit_assignment_preflight_v2.json';assert not p.exists()
        prior.write(p,dict(passed=True,methods=result.testsRun,sources={p:prior.digest(ROOT/p) for p in SOURCES}))
    assert result.wasSuccessful()


if __name__=='__main__':
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    if parser.parse_args().preflight:preflight()
    else:execute(OUT,prior.inputs())
