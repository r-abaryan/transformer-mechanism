"""Fixed-history cold writing across three saved learning stages."""
import argparse
import importlib.util
import json
import time
import unittest
import torch
import readout_conflict_stages as stages
import shared_state_preservation as replay

ROOT=stages.ROOT
OUT=ROOT/'results/write_interference_stages'
PROTOCOL='docs/protocols/write-interference-training-stages.md'
SOURCES=('experiments/write_interference_stages.py','experiments/audit_write_interference_stages.py','tests/test_write_interference_stages.py')
prior=stages.original.prior
GROUPS=('all','endpoint_wrong','correct_in_incomplete','correct_in_complete')
GUARD=1e-9


def cohorts(endpoint):
    complete=endpoint.all(-1)[:,None]
    return dict(all=torch.ones_like(endpoint),endpoint_wrong=~endpoint,
                correct_in_incomplete=endpoint&~complete,correct_in_complete=endpoint&complete)


def summaries(fields,masks,seed,step):
    result=[]
    for group in GROUPS:
        for position in [None,*range(1,int(fields['position'].max())+1)]:
            mask=masks[group] if position is None else masks[group]&(fields['position']==position)
            n=int(mask.sum());r=dict(seed=seed,step=step,group=group,position=position,count=n)
            for name,value in fields.items():
                if name in ('position','competitor'):continue
                if value.dtype==torch.bool:r[name+'_count']=int(value[mask].sum())
                else:
                    r[name+'_mean']=float(value[mask].mean()) if n else None
                    if name in ('margin_change','target_change','later_contribution'):
                        r[name+'_negative']=int((value[mask]<-GUARD).sum())
                        r[name+'_positive']=int((value[mask]>GUARD).sum())
                        r[name+'_unresolved']=int((value[mask].abs()<=GUARD).sum())
            result.append(r)
    return result


@torch.no_grad()
def analyze(model,rows,endpoint_correct,seed,step):
    before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    q,y,repeat=prior.labels(rows);assert not repeat.any()
    e=torch.tensor([r['events'] for r in rows]);match=q[:,:,None]==e[:,None,:,0]
    assert (match.sum(-1)==1).all() and e.shape[1]==q.shape[1]
    position=match.long().argmax(-1)+1
    trace=replay.replay(model,rows,q,y,repeat,torch.zeros(len(rows),16,8,dtype=torch.float64),seed,'native')
    z=trace['logits'];h,m=y.shape;i=torch.arange(h)[:,None];j=torch.arange(m)[None,:]
    good=z.argmax(-1)==y;post=z[position,i,j]
    other=z[-1].clone();other.scatter_(-1,y[...,None],-torch.inf);c=other.argmax(-1)
    key=model.keys(e[...,0]);value=model.value(e[...,1]);beta=model.write_logit.sigmoid();back=model.keys(q);reverse=[]
    for t in reversed(range(e.shape[1])):
        dot=(back*key[:,t,None]).sum(-1);reverse.append(beta*dot)
        back-=beta*dot[...,None]*key[:,t,None]
    coeff=torch.stack(reverse[::-1],-1)
    torch.testing.assert_close(model.output(coeff@value),z[-1],atol=GUARD,rtol=GUARD)
    w=model.output.weight[y]-model.output.weight[c]
    contribution=coeff*torch.einsum('hmv,htv->hmt',w,value)
    times=torch.arange(e.shape[1])[None,None,:];slot=position-1
    target=contribution.gather(-1,slot[...,None]).squeeze(-1)
    earlier=(contribution*(times<slot[...,None])).sum(-1);later=(contribution*(times>slot[...,None])).sum(-1)
    bias=model.output.bias[y]-model.output.bias[c]
    margin=z[-1].gather(-1,y[...,None]).squeeze(-1)-z[-1].gather(-1,c[...,None]).squeeze(-1)
    torch.testing.assert_close(target+earlier+later+bias,margin,atol=GUARD,rtol=GUARD)
    immediate=beta*model.keys(q).square().sum(-1)
    target_now=immediate*(w*value[i,slot]).sum(-1)
    def best_margin(logits):
        others=logits.clone();others.scatter_(-1,y[...,None],-torch.inf)
        return logits.gather(-1,y[...,None]).squeeze(-1)-others.max(-1).values
    pm=best_margin(post);fm=best_margin(z[-1]);after=torch.arange(1,len(z))[:,None,None]>position
    fields=dict(position=position,competitor=c,post_correct=post.argmax(-1)==y,final_correct=good[-1],
        later_harm=((good[:-1]&~good[1:])&after).any(0),later_rescue=((~good[:-1]&good[1:])&after).any(0),
        post_margin=pm,final_margin=fm,margin_change=fm-pm,target_at_write=target_now,target_final=target,
        target_change=target-target_now,earlier_contribution=earlier,later_contribution=later,bias=bias,
        target_coefficient_ratio=coeff.gather(-1,slot[...,None]).squeeze(-1)/immediate)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()) and torch.equal(rng,torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    masks=cohorts(endpoint_correct)
    return dict(seed=seed,step=step,queries=q,labels=y,endpoint_correct=endpoint_correct,logits=z,
        coefficients=coeff,fields=fields,beta=float(beta),metrics=trace['records'][-1]['metrics'],
        summary=summaries(fields,masks,seed,step))


def execute(out,selected=None,fixture=False):
    assert not out.exists(),'Preserve executed evidence'
    selected=stages.cases() if selected is None else selected
    files={p:prior.digest(ROOT/p) for p in (*SOURCES,PROTOCOL,'experiments/readout_conflict_stages.py',
        'experiments/shared_state_preservation.py','experiments/association_load.py','experiments/query_gradient_variability.py')}
    for case in selected:
        for name in ('checkpoint','data'):files[case[name]]=prior.digest(ROOT/case[name])
    cached=None
    if not fixture:
        source=stages.OUT
        old=json.loads((source/'audit.json').read_text());assert old['passed'] and old['report_sha256']==prior.digest(source/'report.json')
        for name in ('report.json','audit.json','lock.json','stages.pt'):
            p=source/name;files[p.relative_to(ROOT).as_posix()]=prior.digest(p)
        cached=torch.load(source/'stages.pt',weights_only=True)
        p=ROOT/'results/write_interference_stages_preflight.json';pre=json.loads(p.read_text())
        assert pre['passed'] and pre['sources']=={s:prior.digest(ROOT/s) for s in SOURCES};files[p.relative_to(ROOT).as_posix()]=prior.digest(p)
    prior.check_files(files);out.mkdir(parents=True)
    prior.write(out/'lock.json',dict(files=files,cases=selected,fixture=fixture,guard=GUARD,groups=list(GROUPS),versions=prior.versions()))
    start=time.perf_counter();bundles=[]
    for case in selected:
        rows=json.loads((ROOT/case['data']).read_text(encoding='utf-8-sig'))['pools']['train_32']
        with torch.random.fork_rng(devices=[]):model=prior.historical.parent.previous.load_model(ROOT/case['checkpoint']).eval()
        assert torch.load(ROOT/case['checkpoint'],weights_only=True)['step']==case['step']
        if fixture:
            q,y,_=prior.labels(rows);endpoint=stages.original.features(model,rows,'train_32')['native_logits'];mask=endpoint.argmax(-1)==y
        else:
            endpoint=next(b for b in cached if b['case']['seed']==case['seed'] and b['case']['step']==12000)
            block=next(b for b in endpoint['source']['blocks'] if b['pool']=='train_32');mask=block['native_logits'].argmax(-1)==block['labels']
        b=analyze(model,rows,mask,case['seed'],case['step'])
        if not fixture:
            original=next(b for b in cached if b['case']['seed']==case['seed'] and b['case']['step']==case['step'])
            block=next(b for b in original['source']['blocks'] if b['pool']=='train_32')
            torch.testing.assert_close(b['logits'][-1],block['native_logits'],atol=GUARD,rtol=GUARD)
            assert torch.equal(b['logits'][-1].argmax(-1),block['native_logits'].argmax(-1))
        bundles.append(b);print(json.dumps(dict(seed=b['seed'],step=b['step'],beta=b['beta'],metrics=b['metrics'])),flush=True)
    torch.save(bundles,out/'traces.pt')
    report=dict(status='complete',summary=[r for b in bundles for r in b['summary']],
        stages=[{k:b[k] for k in ('seed','step','beta','metrics')} for b in bundles],
        query_rows=sum(b['labels'].numel() for b in bundles),answer_outputs=sum(b['logits'].shape[0]*b['labels'].numel() for b in bundles),
        replay_writes=sum((b['logits'].shape[0]-1)*len(b['labels']) for b in bundles),
        training_updates=0,optimizer_calls=0,autograd_calls=0,solver_calls=0,seconds=time.perf_counter()-start,
        lock_sha256=prior.digest(out/'lock.json'),traces_sha256=prior.digest(out/'traces.pt'))
    prior.check_files(files);prior.write(out/'report.json',report)


def preflight():
    spec=importlib.util.spec_from_file_location('checks',ROOT/SOURCES[-1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    p=ROOT/('results/write_interference_stages_preflight.json' if result.wasSuccessful() else f'results/write_interference_stages_preflight_failed_{time.time_ns()}.json')
    assert not p.exists();prior.write(p,dict(passed=result.wasSuccessful(),methods=result.testsRun,sources={s:prior.digest(ROOT/s) for s in SOURCES}))
    assert result.wasSuccessful()


if __name__=='__main__':
    torch.set_num_threads(1);parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    if parser.parse_args().preflight:preflight()
    else:execute(OUT)
