"""Observe fixed historical training without changing its updates or RNG streams."""
import csv
import hashlib
import json
import math
import time
import uuid

import torch
from torch.nn import functional as F
from copy_position_sites import (ROOT, make_model, forward, positions, sample,
    inputs, hash_state, digest, write, train_segment, evaluate_run)

OUT=ROOT/'results/copy_learning_trajectory'
PARENT=ROOT/'results/copy_position_sites'
MODELS=((3767,'cross_only'),(3779,'cross_only'),(3793,'cross_only'),(3767,'both'))
STEPS=tuple(range(0,6001,500))


def summarize(z,y,rows,prefix=False):
    assert len(z)==len(y)==len(rows)
    assert torch.isfinite(z).all()
    result={}
    for group in ('all','distinct','repeated'):
        ids=[i for i,r in enumerate(rows) if group=='all' or
             (len(set(r[:6] if prefix else r))==len(r[:6] if prefix else r))==(group=='distinct')]
        if not ids:result[group]=None;continue
        zz,yy=z[ids],y[ids];pred=zz.argmax(-1);correct=pred==yy
        loss=float(F.cross_entropy(zz.reshape(-1,16),yy.reshape(-1)))
        result[group]=dict(count=len(ids),token_loss=loss,token_accuracy=float(correct.double().mean()),
            exact_accuracy=float(correct.reshape(len(ids),-1).all(-1).double().mean()))
        if zz.ndim==2:
            result[group].update(correct=int(correct.sum()),eos=int((pred==2).sum()),
                margin=float((zz[:,2]-zz.gather(1,yy[:,None]).squeeze(1)).mean()))
    return result


def audit_logits(z,y,rows,stored,prefix=False):
    # Scalar log-sum-exp and decisions independently check vectorized metrics.
    assert set(stored)=={'all','distinct','repeated'}
    for group,m in stored.items():
        ids=[i for i,r in enumerate(rows) if group=='all' or
             (len(set(r[:6] if prefix else r))==len(r[:6] if prefix else r))==(group=='distinct')]
        if not ids:assert m is None;continue
        assert m is not None and m['count']==len(ids)
        zz=z[ids].reshape(len(ids),-1,16).tolist();yy=y[ids].reshape(len(ids),-1).tolist()
        losses=[];ok=[];eos=0;margins=[]
        for seq,labels in zip(zz,yy):
            flags=[]
            for v,t in zip(seq,labels):
                p=max(range(16),key=v.__getitem__);flags.append(p==t);eos+=p==2
                peak=max(v);losses.append(peak+math.log(sum(math.exp(a-peak) for a in v))-v[t]);margins.append(v[2]-v[t])
            ok.append(flags)
        assert abs(sum(losses)/len(losses)-m['token_loss'])<1e-9
        assert abs(sum(map(sum,ok))/len(losses)-m['token_accuracy'])<1e-12
        assert abs(sum(all(f) for f in ok)/len(ok)-m['exact_accuracy'])<1e-12
        if z.ndim==2:
            assert m['correct']==sum(map(sum,ok)) and m['eos']==eos
            assert abs(sum(margins)/len(margins)-m['margin'])<1e-9


def observe(model,data,destination):
    saved={};metrics={};error=0.
    with torch.random.fork_rng(),torch.no_grad():
        model.eval()
        for name,rows in data['pools'].items():
            if not name.startswith(('train_','validation_','test_')) or len(rows[0])>6:continue
            x=torch.tensor(rows);prefix,y=inputs(x)
            assert torch.equal(y[:,:-1],x) and bool((y[:,-1]==2).all())
            assert torch.equal(prefix[:,1:],x) and bool((prefix[:,0]==1).all())
            z=forward(model,x,prefix)[0];rz=forward(model,x,prefix,reference=True)[0]
            error=max(error,float((z-rz).abs().max()));assert torch.equal(z.argmax(-1),rz.argmax(-1))
            metrics[name]=summarize(z,y,rows);audit_logits(rz,y,rows,metrics[name]);saved[name]=dict(logits=z,labels=y)
        rows=data['pools']['test_10'];prefix=torch.tensor([[1]+r[:4] for r in rows]);y=torch.tensor([r[4] for r in rows])
        for length in (6,10):
            x=torch.tensor([r[:length] for r in rows]);z=forward(model,x,prefix)[0][:,-1]
            rz=forward(model,x,prefix,reference=True)[0][:,-1];error=max(error,float((z-rz).abs().max()))
            assert torch.equal(z.argmax(-1),rz.argmax(-1))
            name=f'paired_{length}';metrics[name]=summarize(z,y,rows,True);audit_logits(rz,y,rows,metrics[name],True)
            saved[name]=dict(logits=z,labels=y)
        assert error<=1e-9
        torch.save(saved,destination)
    model.train()
    return metrics,error


def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def replay(folder,seed,arm,data,stop=6000,snapshots=STEPS):
    torch.manual_seed(seed);model=make_model('rope');initial=hash_state(model.state_dict())
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    gen=torch.Generator().manual_seed(seed+1000);pg=torch.Generator().manual_seed(seed+2000)
    pools={p:torch.tensor(r) for p,r in data['pools'].items() if p.startswith('train_')}
    stream=hashlib.sha256();pstream=hashlib.sha256();records=[];examples=0;tick=time.perf_counter()
    for step in range(stop+1):
        if step:
            n,idx,_=sample(pools,gen,'mixed',step);pos=positions(n,pg)
            stream.update(json.dumps([n,idx.tolist()]).encode());pstream.update(json.dumps(pos.tolist()).encode())
            x=pools[f'train_{n}'][idx];prefix,y=inputs(x);opt.zero_grad(set_to_none=True)
            z=forward(model,x,prefix,pos if arm=='both' else None,pos)[0]
            loss=F.cross_entropy(z.flatten(0,1),y.flatten());assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step();examples+=len(x)
        if step in snapshots:
            name=f'{seed}_{arm}_{step}';cp=folder/f'{name}.pt';assert not cp.exists()
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,generator=gen.get_state(),position_generator=pg.get_state()),cp)
            # Exercise saved-checkpoint reload while preserving the training RNG.
            with torch.random.fork_rng():
                probe=make_model('rope');probe.load_state_dict(torch.load(cp,weights_only=True)['model'])
                metrics,error=observe(probe,data,folder/f'{name}_logits.pt')
            records.append(dict(step=step,metrics=metrics,reference_max_error=error,checkpoint=str(cp.relative_to(ROOT))))
            print(json.dumps(dict(seed=seed,arm=arm,step=step,short=metrics['paired_6']['all']['correct'],long_eos=metrics['paired_10']['all']['eos'])),flush=True)
    final=folder/f'{seed}_{arm}_{stop}.pt'
    return dict(seed=seed,arm=arm,initial_sha256=initial,stream_sha256=stream.hexdigest(),position_stream_sha256=pstream.hexdigest(),
        checkpoint=str(final.relative_to(ROOT)),examples=examples,targets=stop*840,updates=stop,records=records,seconds=time.perf_counter()-tick)


def preflight(data):
    folder=ROOT/'results/test_runs'/('copy_trajectory_'+uuid.uuid4().hex);folder.mkdir(parents=True)
    torch.manual_seed(31);m=make_model('rope');p=folder/'initial.pt';torch.save(m.state_dict(),p);m.load_state_dict(torch.load(p,weights_only=True))
    observe(m,data,folder/'initialized_logits.pt')
    z=torch.zeros(2,16,dtype=torch.float64);z[0,2]=3;z[1,3]=3;y=torch.tensor([3,3]);rows=[[3]*10,list(range(3,9))+[3]*4]
    met=summarize(z,y,rows,True);audit_logits(z,y,rows,met,True)
    failures=0
    for mode in ('label','logit','missing'):
        try:
            changed=z.clone();labels=y.clone();mm=dict(met)
            if mode=='label':labels[0]=2
            elif mode=='logit':changed[1,2]=4
            else:mm.pop('distinct')
            audit_logits(changed,labels,rows,mm,True)
        except AssertionError:failures+=1
    assert failures==3
    absent=summarize(z[:1],y[:1],rows[:1],True);audit_logits(z[:1],y[:1],rows[:1],absent,True);assert absent['distinct'] is None
    z[1,2]=3;ties=summarize(z,y,rows,True);audit_logits(z,y,rows,ties,True);assert ties['all']['eos']==2
    old=train_segment(folder,31,'cross_only',data,2)
    other=folder/'observed';other.mkdir();new=replay(other,31,'cross_only',data,2,(0,1,2))
    assert equal(torch.load(ROOT/old['checkpoint'],weights_only=True),torch.load(ROOT/new['checkpoint'],weights_only=True))
    for k in ('initial_sha256','stream_sha256','position_stream_sha256'):assert old[k]==new[k]
    return dict(passed=True,initialized_saved_evaluator_and_scalar_audit=True,three_error_controls_rejected=True,
        tie_and_absent_group=True,original_two_update_equivalence=True,fixture_training_updates=4)


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists()
    data=json.loads((PARENT/'data.json').read_text());parent=json.loads((PARENT/'report.json').read_text())
    pre=preflight(data);write(OUT/'preflight.json',pre)
    protected=json.loads((ROOT/'results/copy_added_read_mask/plan.json').read_text())['protected']
    paths=['experiments/copy_learning_trajectory.py','docs/protocols/copy-learning-trajectory.md','results/copy_learning_trajectory/preflight.json','results/copy_position_sites/profile.json']
    for p in paths:protected[p]=digest(ROOT/p)
    assert all(digest(ROOT/p)==v for p,v in protected.items())
    write(OUT/'plan.json',dict(models=MODELS,steps=STEPS,updates=24000,targets=20160000,protected=protected))
    report=dict(status='running',runs=[],plan_sha256=digest(OUT/'plan.json'));write(OUT/'report.json',report)
    for seed,arm in MODELS:
        run=replay(OUT,seed,arm,data);old=next(c for c in parent['confirmation'] if c['seed']==seed and c['arm']==arm)
        original=torch.load(ROOT/old['checkpoint'],weights_only=True);new=torch.load(ROOT/run['checkpoint'],weights_only=True)
        run['endpoint_exact']=equal(original,new)
        run['streams_exact']=all(old[k]==run[k] for k in ('initial_sha256','stream_sha256','position_stream_sha256'))
        assert run['endpoint_exact'] and run['streams_exact']
        model=make_model('rope');model.load_state_dict(new['model']);final,_=evaluate_run(model.eval(),data,arm,True)
        assert final==old['metrics'];run['original_final_metrics']=final
        fitsteps=[];signature=[]
        for rec in run['records']:
            mm=rec['metrics'];fit=all(mm[f'train_{n}'][g]['exact_accuracy']>=.99 for n in range(3,7) for g in ('distinct','repeated'))
            if fit:fitsteps.append(rec['step'])
            if mm['paired_6']['all']['correct']/128>=.99 and mm['paired_10']['all']['eos']/128>=.9:signature.append(rec['step'])
        run['short_fit_steps']=fitsteps;run['failure_signature_steps']=signature
        report['runs'].append(run);write(OUT/'report.json',report)
    assert all(digest(ROOT/p)==v for p,v in protected.items())
    report.update(status='complete',audit=dict(passed=True,endpoints_and_optimizer_and_generators_exact=True,streams_exact=True,
        original_final_metrics_exact=True,snapshots=52,protected_files=len(protected),max_logit_error=max(rec['reference_max_error'] for run in report['runs'] for rec in run['records'])))
    write(OUT/'report.json',report)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()
