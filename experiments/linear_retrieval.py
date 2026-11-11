"""Matched softmax/positive-kernel retrieval comparison; no speed claim."""
import copy
import hashlib
import json
import time
import torch
from torch.nn import functional as F
from mqa_training import model_new, hash_state, forward as softmax_forward
from positional_comparison import rotate
from linear_attention import pairwise
from recurrent_retrieval import ROOT, datasets, write
from query_exposure import measure
from position_distractors import digest

SEEDS=(2591,2609,2621)
ARMS=('softmax','linear')
UPDATES=6000


def models(seed):
    torch.manual_seed(seed)
    model=model_new('mqa')
    return dict(softmax=model,linear=copy.deepcopy(model))


def forward(model,ids,arm):
    if arm=='softmax':return softmax_forward(model,ids)
    x=model.embedding(ids);n=ids.shape[1];pos=torch.arange(n)+4
    valid=torch.ones(len(ids),1,n,dtype=torch.bool)
    for block in model.blocks:
        z=block.norm1(x);a=block.attn
        q,k,v=[F.linear(z,getattr(a,t+'_weight'),getattr(a,t+'_bias')) for t in ('q','k','v')]
        q=q.reshape(len(ids),n,2,8).transpose(1,2)
        k,v=[t.reshape(len(ids),n,1,8).transpose(1,2).repeat_interleave(2,dim=1) for t in (k,v)]
        q,k=rotate(q,pos),rotate(k,pos)
        y=pairwise(q,k,v,valid).transpose(1,2).reshape(len(ids),n,16)
        x=x+a.out_proj(y)
        x=x+block.linear2(F.relu(block.linear1(block.norm2(x))))
    return model.output(model.final_norm(x))


def evaluate(model,data,arm,fn=forward):
    x,y=data
    with torch.no_grad():scores=torch.cat([fn(model,c,arm)[:,-1] for c in x.split(256)])
    return measure(scores,y,x),scores


def decision(runs):
    fit={a:all(r['status']=='trained' and all(r['metrics'][f'{s}_{n}'][f'{k}_accuracy']>=.99 for s in ('train','validation') for n in (3,4) for k in ('repeated','unique')) for r in runs if r['arm']==a) for a in ARMS}
    reliable={a:fit[a] and all(r['metrics'][f'test_{n}'][f'{k}_accuracy']>=(.9 if n==6 else .99) for r in runs if r['arm']==a for n in (3,4,6) for k in ('repeated','unique')) for a in ARMS}
    gains=[]
    for seed in SEEDS:
        pair={r['arm']:r for r in runs if r['seed']==seed}
        if any(r['status']!='trained' for r in pair.values()):continue
        scores={a:sum(pair[a]['metrics']['test_6'][f'{k}_accuracy'] for k in ('repeated','unique'))/2 for a in ARMS}
        gains.append(dict(seed=seed,**scores,linear_minus_softmax=scores['linear']-scores['softmax']))
    winner=None
    if len(gains)==3:
        if all(g['linear_minus_softmax']>=.10 for g in gains):winner='linear'
        if all(g['linear_minus_softmax']<=-.10 for g in gains):winner='softmax'
    return dict(fit=fit,reliability=reliable,transfer_specific_eligible=all(fit.values()),consistent_advantage=winner,gains=gains)


def profile():
    torch.set_num_threads(1)
    _,data=datasets();rows=[]
    for arm,m in models(2587).items():
        optimizer=torch.optim.Adam(m.parameters(),lr=.001)
        x,y=data['train_4'];tick=time.perf_counter()
        optimizer.zero_grad();loss=F.cross_entropy(forward(m,x[:64],arm)[:,-1],y[:64]);loss.backward();optimizer.step()
        rows.append(dict(arm=arm,parameters=sum(p.numel() for p in m.parameters()),step_seconds=time.perf_counter()-tick,optimizer_state_tensors=sum(len(s) for s in optimizer.state.values()),loss=float(loss.detach()),device='cpu',dtype='float64',batch=64,sequence_length=10,peak_gpu_allocated_bytes=0,peak_gpu_reserved_bytes=0))
    p=ROOT/'results/linear_retrieval_profile.json';assert not p.exists()
    write(p,dict(passed=True,rows=rows,source_sha256=digest(ROOT/'experiments/linear_retrieval.py'),note='Complete CPU Adam step; GPU unused, CPU peak memory not measured. Profile weights discarded.'))
    print(rows,flush=True)


def main():
    from audit_linear_retrieval import preflight,baselines
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rp=ROOT/'results/linear_retrieval.json';assert not rp.exists()
    payload,data=datasets()
    history={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'results').glob('*.json')}
    files=list((ROOT/'experiments').glob('*.py'))+[ROOT/'docs/protocols/linear-retrieval.md',ROOT/'results/mqa_training_data.json']
    report=dict(phase='preflight',torch=torch.__version__,device='cpu',dtype='float64',provenance={str(p.relative_to(ROOT)):digest(p) for p in files},historical=history,runs=[])
    write(rp,report)
    report.update(preflight=preflight(payload,data),baselines=baselines(data));write(rp,report)
    directory=ROOT/'results/linear_retrieval_checkpoints';directory.mkdir(exist_ok=False)
    for i,seed in enumerate(SEEDS):
        pair=models(seed)
        for arm in ARMS[::1 if i%2==0 else -1]:
            model=pair[arm];optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            run=dict(seed=seed,arm=arm,parameters=sum(p.numel() for p in model.parameters()),initial_sha256=hash_state(model.state_dict()),status='running',curve=[],metrics={})
            report['runs'].append(run);write(rp,report)
            g=torch.Generator().manual_seed(seed+1000);stream=hashlib.sha256();tick=time.perf_counter()
            try:
                for step in range(1,UPDATES+1):
                    x,y=data[f'train_{3 if step%2 else 4}'];idx=torch.randint(len(x),(64,),generator=g)
                    stream.update(json.dumps(idx.tolist()).encode());optimizer.zero_grad()
                    loss=F.cross_entropy(forward(model,x[idx],arm)[:,-1],y[idx])
                    assert bool(torch.isfinite(loss));loss.backward()
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                    optimizer.step()
                    if step%1000==0:
                        run['curve'].append(dict(step=step,minibatch_loss=loss.item()))
                        print(dict(seed=seed,arm=arm,step=step,loss=loss.item()),flush=True);write(rp,report)
                run['status']='trained'
            except Exception as exc:
                run.update(status='failed',failure=repr(exc))
            cp=directory/f'{seed}_{arm}.pt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step),cp)
            run.update(step=step,elapsed_seconds=time.perf_counter()-tick,stream_sha256=stream.hexdigest(),checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=digest(cp),examples=step*64,supervised_targets=step*64,input_tokens=sum(64*(8 if s%2 else 10) for s in range(1,step+1)),peak_gpu_allocated_bytes=0,peak_gpu_reserved_bytes=0)
            if run['status']=='trained':
                for key in data:
                    if not key.startswith('test_'):run['metrics'][key],_=evaluate(model,data[key],arm)
            report['phase']='training';write(rp,report)
    report['phase']='all_attempts_finished_before_test';write(ROOT/'results/linear_retrieval_pretest.json',report)
    for run in report['runs']:
        if run['status']!='trained':continue
        model=models(run['seed'])[run['arm']];model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model']);scores={}
        for key in data:
            metric,scores[key]=evaluate(model,data[key],run['arm'])
            if key in run['metrics']:assert run['metrics'][key]==metric
            run['metrics'][key]=metric
        sp=directory/f"{run['seed']}_{run['arm']}_scores.pt";torch.save(scores,sp)
        run.update(scores=str(sp.relative_to(ROOT)),scores_sha256=digest(sp));write(rp,report)
    assert all(digest(ROOT/p)==h for p,h in history.items())
    report.update(phase='complete',decision=decision(report['runs']));write(rp,report)
    print(report['decision'],flush=True)


if __name__=='__main__':
    import sys
    profile() if '--profile' in sys.argv else main()
