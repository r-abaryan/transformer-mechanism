"""Frozen global-relay comparison; derived from the preserved local study."""
import copy
import hashlib
import json
import random
import sys
import time

import torch
from torch.nn import functional as F
from mqa_training import model_new, hash_state, forward as original_forward
from positional_comparison import rotate
from position_distractors import digest
from recurrent_retrieval import ROOT, write

SEEDS=(2521,2531,2539)
DISTANCES=(1,4,5,12,16)
ARMS=('full','local','global')
PREFIX='global_retrieval'


def forward(model,ids,arm):
    n=ids.shape[1];x=model.embedding(ids);positions=torch.arange(n)+4
    i=torch.arange(n)[:,None];j=torch.arange(n)[None,:]
    visible=(j<=i)
    if arm!='full':
        local=(i-j<=2)
        if arm=='global':local=local | (i==n//2) | (j==n//2)
        visible=visible & local
    mask=~visible
    for block in model.blocks:
        z=block.norm1(x);m=block.attn
        q,k,v=[F.linear(z,getattr(m,t+'_weight'),getattr(m,t+'_bias')) for t in ('q','k','v')]
        q=q.reshape(len(ids),n,2,8).transpose(1,2)
        k,v=[t.reshape(len(ids),n,1,8).transpose(1,2) for t in (k,v)]
        q,k=rotate(q,positions),rotate(k,positions)
        k,v=[t.repeat_interleave(2,dim=1) for t in (k,v)]
        a=(q@k.transpose(-1,-2)/(8**.5)).masked_fill(mask,-torch.inf).softmax(-1)
        h=(a@v).transpose(1,2).reshape(len(ids),n,16)
        x=x+m.out_proj(h)
        x=x+block.linear2(F.relu(block.linear1(block.norm2(x))))
    return model.output(model.final_norm(x))


def make_data():
    rng=random.Random(2026100205);seen=set();pools={}
    for name,n,count in (('train_17',17,128),('validation_17',17,32),('test_17',17,32),('test_21',21,32)):
        bases=[];rows=[]
        for _ in range(count):
            for attempt in range(100000):
                base=list(rng.randrange(11) for _ in range(n-1))
                base[n//2]=11
                base=tuple(base)
                signature=(n,base)
                if signature not in seen:break
            else:raise RuntimeError('Proposal budget exhausted')
            seen.add(signature);bases.append(list(base))
        for d in DISTANCES:
            for family,base in enumerate(bases):
                for y in range(12,18):
                    ids=list(base)+[18];ids[n-1-d]=y
                    rows.append(dict(ids=ids,label=y,distance=d,family=family))
        pools[name]=dict(bases=bases,rows=rows)
    return dict(generator_seed=2026100205,pools=pools)


def tensors(payload):
    return {name:(torch.tensor([r['ids'] for r in pool['rows']]),torch.tensor([r['label'] for r in pool['rows']]),
                  torch.tensor([r['distance'] for r in pool['rows']])) for name,pool in payload['pools'].items()}


def measure(scores,y,d):
    losses=F.cross_entropy(scores,y,reduction='none');correct=scores.argmax(-1)==y
    masks={'all':torch.ones_like(d,dtype=torch.bool),'near':d<=4,'after_global':d==5,'before_global':d>=12}
    masks.update({str(v):d==v for v in DISTANCES})
    return {k:dict(count=int(m.sum()),correct=int(correct[m].sum()),accuracy=correct[m].double().mean().item(),loss=losses[m].mean().item()) for k,m in masks.items()}


def evaluate(model,data,arm,fn=forward):
    x,y,d=data
    with torch.no_grad():scores=torch.cat([fn(model,c,arm)[:,-1] for c in x.split(128)])
    return measure(scores,y,d),scores


def batch_indices(generator):
    # Two independently selected families per distance, each with all six labels.
    selected=torch.randint(128,(5,2),generator=generator)
    return (torch.arange(5)[:,None,None]*768+selected[:,:,None]*6+torch.arange(6)[None,None,:]).flatten()


def decision(runs):
    out={}
    for arm in ARMS:
        rows=[r for r in runs if r['arm']==arm]
        groups={'local':('1','4'),'global':('1','4','12','16'),'full':tuple(map(str,DISTANCES))}[arm]
        ready=len(rows)==3 and all(r['status']=='trained' for r in rows)
        fit=ready and all(r['metrics']['train_17'][g]['accuracy']>=.99 for r in rows for g in groups)
        validation=ready and all(r['metrics']['validation_17'][g]['accuracy']>=.95 for r in rows for g in groups)
        out[arm]=dict(training_fit=fit,validation_reliability=validation,
            short_reliability=fit and validation and all(r['metrics']['test_17'][g]['accuracy']>=.95 for r in rows for g in groups),
            longer_transfer=fit and validation and all(r['metrics']['test_21'][g]['accuracy']>=.95 for r in rows for g in groups))
    return out


def profile():
    p=ROOT/f'results/{PREFIX}_profile.json';assert not p.exists()
    cells=[]
    for arm in ARMS:
        for n in (17,21):
            torch.manual_seed(2507);m=model_new('mqa');o=torch.optim.Adam(m.parameters(),lr=.001)
            x=torch.randint(12,(60,n));x[:,-1]=18;y=torch.arange(60)%6+12;x[:,-2]=y
            tick=time.perf_counter();o.zero_grad(set_to_none=True)
            loss=F.cross_entropy(forward(m,x,arm)[:,-1],y);loss.backward();o.step()
            assert torch.isfinite(loss) and all(torch.isfinite(v).all() for v in m.parameters())
            assert len(o.state)==len(list(m.parameters()))
            cells.append(dict(arm=arm,length=n,batch=60,parameters=sum(p.numel() for p in m.parameters()),
                complete_step_seconds=time.perf_counter()-tick,optimizer_state_tensors=sum(len(v) for v in o.state.values()),
                optimizer_tensor_bytes=sum(t.numel()*t.element_size() for v in o.state.values() for t in v.values() if isinstance(t,torch.Tensor))))
    write(p,dict(passed=True,device='cpu',dtype='float64',threads=1,gpu_peak_allocated_bytes=None,gpu_peak_reserved_bytes=None,
        reason='Tiny existing double-precision model; CPU chosen for arithmetic comparability, not throughput. GPU unused.',cells=cells,source_sha256=digest(ROOT/'experiments/global_retrieval.py')))
    print(json.dumps(cells))


def main():
    rp=ROOT/f'results/{PREFIX}.json';ip=ROOT/f'results/{PREFIX}_data.json'
    assert not rp.exists(),'Preserve attempts; no restart.'
    from audit_global_retrieval import audit_data, preflight
    payload=make_data()
    assert not ip.exists()
    write(ip,payload)
    data=tensors(payload)
    files=[*sorted((ROOT/'experiments').glob('*.py')),ROOT/'docs/protocols/global-retrieval.md']
    report=dict(phase='preflight',device='cpu',dtype='float64',threads=1,torch=torch.__version__,updates=6000,batch=60,
        effective_batch=60,examples_per_run=360000,supervised_targets_per_run=360000,input_tokens_per_run=6120000,
        gpu_peak_allocated_bytes=None,gpu_peak_reserved_bytes=None,profile_sha256=digest(ROOT/f'results/{PREFIX}_profile.json'),
        input_sha256=digest(ip),provenance={str(p.relative_to(ROOT)):digest(p) for p in files},
        historical={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'results').glob('*.json') if not p.name.startswith(PREFIX)},
        input_checks=audit_data(payload),preflight=preflight(data),runs=[])
    write(rp,report)
    directory=ROOT/f'results/{PREFIX}_checkpoints';directory.mkdir(exist_ok=True)
    x,y,_=data['train_17']
    for index,seed in enumerate(SEEDS):
        torch.manual_seed(seed);initial=model_new('mqa')
        for arm in ARMS[index:]+ARMS[:index]:
            model=copy.deepcopy(initial);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            run=dict(seed=seed,arm=arm,status='running',parameters=sum(p.numel() for p in model.parameters()),
                initial_sha256=hash_state(model.state_dict()),curve=[],metrics={})
            report['runs'].append(run);write(rp,report)
            gen=torch.Generator().manual_seed(seed+1000);stream=hashlib.sha256();tick=time.perf_counter()
            for step in range(1,6001):
                idx=batch_indices(gen);stream.update(json.dumps(idx.tolist()).encode())
                optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(forward(model,x[idx],arm)[:,-1],y[idx])
                finite=bool(torch.isfinite(loss))
                if finite:
                    loss.backward();finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                if not finite:
                    run.update(status='failed',failure_step=step);break
                optimizer.step()
                if step%1000==0:
                    row=dict(step=step,minibatch_loss=loss.item())
                    if step in (2000,4000,6000):row['pools']={k:evaluate(model,v,arm)[0] for k,v in data.items() if not k.startswith('test')}
                    run['curve'].append(row)
                    print(json.dumps(dict(seed=seed,arm=arm,step=step,loss=loss.item())),flush=True)
            if run['status']!='failed':run.update(status='trained',metrics=run['curve'][-1]['pools'].copy())
            cp=directory/f'{seed}_{arm}.pt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step),cp)
            run.update(step=step,elapsed_seconds=time.perf_counter()-tick,checkpoint=str(cp.relative_to(ROOT)),
                checkpoint_sha256=digest(cp),stream_sha256=stream.hexdigest())
            report['phase']='training';write(rp,report)
    report['phase']='all_attempts_finished_before_test';write(ROOT/f'results/{PREFIX}_pretest.json',report)
    for run in report['runs']:
        if run['status']!='trained':continue
        model=model_new('mqa');model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model']);scores={}
        for k,v in data.items():
            metric,scores[k]=evaluate(model,v,run['arm'])
            if k in run['metrics']:assert metric==run['metrics'][k]
            run['metrics'][k]=metric
        if run['arm']=='global':
            run['ablated']={}
            for k,v in data.items():
                run['ablated'][k],scores['ablated/'+k]=evaluate(model,v,'local')
        sp=directory/f"{run['seed']}_{run['arm']}_scores.pt";torch.save(scores,sp)
        run.update(scores=str(sp.relative_to(ROOT)),scores_sha256=digest(sp));write(rp,report)
    report.update(phase='complete',decision=decision(report['runs']));write(rp,report)
    print(json.dumps(report['decision']),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    profile() if '--profile' in sys.argv else main()
