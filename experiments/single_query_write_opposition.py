"""Resolve query mixing versus signed single-query write sensitivities."""
import argparse
import csv
import importlib.util
import json
import time
import unittest
import torch
import write_history_opposition as history

primary=history.primary
ROOT=primary.ROOT
OUT=ROOT/'results/single_query_write_opposition'
PROTOCOL='docs/protocols/single-query-write-opposition.md'
SOURCES=('experiments/single_query_write_opposition.py','experiments/audit_single_query_write_opposition.py',
         'tests/test_single_query_write_opposition.py')


def classify(x,guard):return 'negative' if x < -guard else 'positive' if x > guard else 'unresolved'


def query_statistics(g,coeff,ids,masks,queries,labels,seed):
    h,m,t,v=g.shape;combined=torch.zeros(h,m,6,v,dtype=torch.float64)
    combined.scatter_add_(2,ids[:,None,:,None].expand(-1,m,-1,v),g)
    energy=g.square().sum((2,3));joint=combined.square().sum((2,3));within=joint-energy
    summed=torch.zeros(h,m,6,dtype=torch.float64)
    summed.scatter_add_(2,ids[:,None].expand(-1,m,-1),coeff)
    geometry=summed.square().sum(-1)-coeff.square().sum(-1)
    opposite=torch.zeros(h,m,dtype=torch.bool)
    for value in range(6):
        positions=ids[:,None]==value
        opposite|=((coeff>1e-12)&positions).any(-1)&((coeff < -1e-12)&positions).any(-1)
    records=[]
    for i in range(h):
        for q in range(m):
            records.append(dict(seed=seed,history=i,query=int(queries[i,q]),slot=q,target=int(labels[i,q]),
                correct=bool(masks['correct'][i,q]),event_energy=float(energy[i,q]),joint_energy=float(joint[i,q]),
                within=float(within[i,q]),sign=classify(float(within[i,q]),1e-14),geometry=float(geometry[i,q]),
                geometry_sign=classify(float(geometry[i,q]),1e-12),opposite_same_value=bool(opposite[i,q])))
    summaries=[]
    for group,mask in masks.items():
        mixed=(g*mask[:,:,None,None]).sum(1);map_sum=torch.zeros(h,6,v,dtype=torch.float64)
        map_sum.scatter_add_(1,ids[:,:,None].expand(-1,-1,v),mixed)
        w_mixed=float(map_sum.square().sum()-mixed.square().sum());w_single=float(within[mask].sum())
        selected=[r for r in records if bool(mask[r['history'],r['slot']])]
        s=dict(seed=seed,group=group,queries=len(selected),individual_event_energy=float(energy[mask].sum()),
            single_query_within=w_single,mixed_query_within=w_mixed,cross_query_within=w_mixed-w_single,
            opposite_same_value=sum(r['opposite_same_value'] for r in selected))
        for label in ('negative','positive','unresolved'):
            s['weighted_'+label]=sum(r['sign']==label for r in selected)
            s['geometry_'+label]=sum(r['geometry_sign']==label for r in selected)
        summaries.append(s)
    return records,summaries


@torch.no_grad()
def analyze(model,rows,bundle):
    before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    q,y,_=primary.prior.labels(rows);assert torch.equal(q,bundle['queries']) and torch.equal(y,bundle['labels'])
    e=torch.tensor([r['events'] for r in rows]);k=model.keys(e[...,0]);value=model.value(e[...,1]);ids=e[...,1]
    beta=model.write_logit.sigmoid();back=model.keys(q);reverse=[]
    for step in reversed(range(e.shape[1])):
        dot=(back*k[:,step,None]).sum(-1);reverse.append(beta*dot)
        back=back-beta*dot[...,None]*k[:,step,None]
    coeff=torch.stack(list(reversed(reverse)),dim=-1)
    z=model.output(coeff@value);torch.testing.assert_close(z,bundle['logits'],atol=1e-9,rtol=1e-9)
    assert torch.equal(z.argmax(-1)==y,bundle['masks']['correct'])
    dz=z.softmax(-1);dz.scatter_add_(-1,y[...,None],-torch.ones_like(y[...,None],dtype=torch.float64));dz/=4*y.numel()
    d=dz@model.output.weight;g=coeff[...,None]*d[:,:,None]
    for group,mask in bundle['masks'].items():
        torch.testing.assert_close((g*mask[:,:,None,None]).sum(1),bundle['gradients'][group]['events'],atol=1e-12,rtol=1e-12)
    records,summary=query_statistics(g,coeff,ids,bundle['masks'],q,y,bundle['seed'])
    examples=[]
    for row in records:
        if row['correct'] or row['sign']!='negative':continue
        h,j=row['history'],row['slot'];pair=None
        for a in range(e.shape[1]):
            for b in range(a+1,e.shape[1]):
                if ids[h,a]==ids[h,b] and float(coeff[h,j,a]*coeff[h,j,b]) < -1e-12:
                    pair=(a,b);break
            if pair:break
        if pair:
            a,b=pair;examples=[dict(seed=bundle['seed'],history=h,query=row['query'],target=row['target'],winner=int(z[h,j].argmax()),
                within=row['within'],geometry=row['geometry'],first_step=a+1,second_step=b+1,value_id=int(ids[h,a]),
                first_coefficient=float(coeff[h,j,a]),second_coefficient=float(coeff[h,j,b]),
                gradient_dot=float(g[h,j,a]@g[h,j,b]))];break
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()) and torch.equal(rng,torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    return dict(seed=bundle['seed'],queries=q,labels=y,masks=bundle['masks'],coefficients=coeff,read_gradient=d),records,summary,examples


def execute(source,out,fixture=False):
    assert not out.exists(),'Preserve evidence'
    lock,old,bundles=history.load(source);assert fixture==lock['fixture'];files=dict(lock['files'])
    for p in (*SOURCES,PROTOCOL):files[p]=primary.prior.digest(ROOT/p)
    for name in ('lock.json','report.json','audit.json','gradients.pt'):
        p=source/name;files[p.relative_to(ROOT).as_posix()]=primary.prior.digest(p)
    if not fixture:
        pre=ROOT/'results/single_query_write_opposition_preflight.json';data=json.loads(pre.read_text())
        assert data['passed'] and data['sources']=={p:primary.prior.digest(ROOT/p) for p in SOURCES}
        files[pre.relative_to(ROOT).as_posix()]=primary.prior.digest(pre)
        for p in (history.OUT/'lock.json',history.OUT/'report.json',history.OUT/'audit.json'):
            files[p.relative_to(ROOT).as_posix()]=primary.prior.digest(p)
        audited=json.loads((history.OUT/'audit.json').read_text());assert audited['passed'] and audited['report_sha256']==primary.prior.digest(history.OUT/'report.json')
    primary.prior.check_files(files);out.mkdir(parents=True)
    primary.prior.write(out/'lock.json',dict(source=source.relative_to(ROOT).as_posix(),cases=lock['cases'],fixture=fixture,files=files,
        threads=torch.get_num_threads(),dtype='float64',weighted_guard=1e-14,geometry_guard=1e-12))
    start=time.perf_counter();factors=[];records=[];summary=[];examples=[];sensitivities=0
    for case,bundle in zip(lock['cases'],bundles):
        assert case['seed']==bundle['seed'];rows=json.loads((ROOT/case['data']).read_text(encoding='utf-8-sig'))['pools'][case['pool']]
        model=primary.prior.historical.parent.previous.load_model(ROOT/case['checkpoint']).eval()
        factor,r,s,example=analyze(model,rows,bundle);factors.append(factor);records.extend(r);summary.extend(s);examples.extend(example)
        sensitivities+=factor['coefficients'].numel();print(json.dumps(s),flush=True)
    torch.save(factors,out/'factors.pt')
    with (out/'queries.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    report=dict(status='complete',fixture=fixture,summary=summary,examples=examples,query_rows=len(records),
        query_event_sensitivities=sensitivities,cohorts=len(summary),coefficient_logit_reconstructions=len(factors),
        recurrent_prefills=0,autograd_calls=0,optimizer_calls=0,training_updates=0,seconds=time.perf_counter()-start,
        lock_sha256=primary.prior.digest(out/'lock.json'),factors_sha256=primary.prior.digest(out/'factors.pt'),
        queries_sha256=primary.prior.digest(out/'queries.csv'))
    primary.prior.check_files(files);primary.prior.write(out/'report.json',report)


def preflight():
    spec=importlib.util.spec_from_file_location('test_single_query_write_opposition',ROOT/SOURCES[-1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if result.wasSuccessful():
        p=ROOT/'results/single_query_write_opposition_preflight.json';assert not p.exists()
        primary.prior.write(p,dict(passed=True,methods=result.testsRun,sources={p:primary.prior.digest(ROOT/p) for p in SOURCES},optimizer_calls=0))
    assert result.wasSuccessful()


if __name__=='__main__':
    torch.set_num_threads(1);parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    if parser.parse_args().preflight:preflight()
    else:execute(primary.OUT,OUT)
