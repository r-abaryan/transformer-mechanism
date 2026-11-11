"""Fresh bounded delta/gated-delta memory-control comparison."""
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from recurrent_retrieval import write
from position_distractors import digest
from mqa_training import hash_state

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(2711,2719,2729)
ARMS=('delta','gated')
GROUPS=('retain','overwrite','reset_present','reset_missing')
STEPS=3000


class Memory(nn.Module):
    def __init__(self,gated=False):
        super().__init__()
        self.key=nn.Linear(8,8);self.value=nn.Linear(8,8);self.beta=nn.Linear(8,1)
        self.query=nn.Embedding(3,8);self.output=nn.Linear(8,5)
        self.alpha=nn.Linear(8,1) if gated else None
        if gated:
            nn.init.zeros_(self.alpha.weight);nn.init.constant_(self.alpha.bias,2.)
        self.double()

    def forward(self,x,q):
        k=self.key(x);k=k/k.norm(dim=-1,keepdim=True)
        v=self.value(x);b=self.beta(x).sigmoid()
        a=self.alpha(x).sigmoid() if self.alpha is not None else torch.ones_like(b)
        s=x.new_zeros(len(x),8,8)
        for i in range(x.shape[1]):
            p=a[:,i,:,None]*s
            old=(k[:,i,:,None]*p).sum(1)
            s=p+b[:,i,:,None]*k[:,i,:,None]*(v[:,i]-old)[:,None,:]
        read=(self.query(q)[:,:,None]*s).sum(1)
        return self.output(read)


def pair(seed):
    torch.manual_seed(seed);base=Memory(False)
    other=Memory(True)
    other.load_state_dict(base.state_dict(),strict=False)
    return dict(delta=base,gated=other)


def reference(model,x,q,wrong=False):
    def linear(x,m):return x@m.weight.T+m.bias
    raw=linear(x,model.key);k=raw/raw.square().sum(-1,keepdim=True).sqrt()
    v=linear(x,model.value);b=linear(x,model.beta).sigmoid()
    a=linear(x,model.alpha).sigmoid() if model.alpha is not None else torch.ones_like(b)
    s=x.new_zeros(len(x),8,8);eye=torch.eye(8,dtype=x.dtype)
    for i in range(x.shape[1]):
        kk=k[:,i];bb=b[:,i,:,None];aa=a[:,i,:,None]
        transition=aa*(eye-bb*kk[:,:,None]*kk[:,None,:])
        if wrong:transition=aa*eye
        s=transition@s+bb*kk[:,:,None]*v[:,i,None,:]
    read=(model.query.weight[q][:,None,:]@s).squeeze(1)
    return linear(read,model.output)


def oracle(events,q,ignore_reset=False):
    memory={}
    for key,value in events:
        if key==3:
            if not ignore_reset:memory={}
        else:memory[key]=value
    return memory.get(q,4)


def encode(rows):
    events=torch.tensor([r['events'] for r in rows]);x=torch.zeros(len(rows),events.shape[1],8,dtype=torch.float64)
    for i,row in enumerate(rows):
        for j,(key,value) in enumerate(row['events']):
            if key==3:x[i,j,7]=1
            else:x[i,j,key]=1;x[i,j,3+value]=1
    return x,torch.tensor([r['query'] for r in rows]),torch.tensor([r['label'] for r in rows])


def make_data():
    target=ROOT/'results/learned_memory_control_data.json';assert not target.exists()
    rng=random.Random(2026100301);seen=set();pools={}
    for split,n,count in [('train',6,96),('train',8,96),('validation',6,24),('validation',8,24),('test',6,24),('test',8,24),('test',12,24)]:
        rows=[]
        for group in GROUPS:
            accepted=0
            for attempt in range(100000):
                if accepted==count:break
                query=rng.randrange(3);others=[k for k in range(3) if k!=query]
                events=[[rng.choice(others),rng.randrange(4)] for _ in range(n)]
                if group.startswith('reset'):
                    reset=rng.randrange(2,n-1);old=rng.randrange(reset)
                    events[reset]=[3,0];events[old]=[query,rng.randrange(4)]
                    if group=='reset_present':
                        new=rng.randrange(reset+1,n);events[new]=[query,(events[old][1]+rng.randrange(1,4))%4]
                else:
                    old=rng.randrange(n-2);events[old]=[query,rng.randrange(4)]
                    if group=='overwrite':
                        new=rng.randrange(old+1,n);events[new]=[query,(events[old][1]+rng.randrange(1,4))%4]
                family=min(tuple((k,(v+s)%4 if k!=3 else 0) for k,v in events) for s in range(4))
                if family in seen:continue
                seen.add(family);accepted+=1
                for shift in range(4):
                    ev=[[k,(v+shift)%4 if k!=3 else 0] for k,v in events]
                    rows.append(dict(events=ev,query=query,label=oracle(ev,query),group=group,family=family))
            assert accepted==count
        pools[f'{split}_{n}']=rows
    result=dict(generator_seed=2026100301,pools=pools)
    write(target,result);return result


def audit_data(payload):
    ownership={};count=0;baselines={}
    for split,rows in payload['pools'].items():
        base={g:{b:0 for b in ('unknown','constant0','first_match','last_value','ignore_reset','oracle')} for g in GROUPS}
        sizes={g:0 for g in GROUPS}
        for r in rows:
            events=r['events'];q=r['query'];g=r['group'];label=r['label']
            # Independent label: search backward, stopping at the most recent reset.
            expected=4
            for key,value in reversed(events):
                if key==3:break
                if key==q:expected=value;break
            assert expected==label
            f=min(tuple((k,(v+s)%4 if k!=3 else 0) for k,v in events) for s in range(4))
            assert f==tuple(tuple(p) for p in r['family'])
            assert ownership.setdefault(f,split)==split
            hits=[i for i,p in enumerate(events) if p[0]==q];resets=[i for i,p in enumerate(events) if p[0]==3]
            assert len(hits)==(2 if g in ('overwrite','reset_present') else 1)
            assert bool(resets)==g.startswith('reset')
            if resets:assert (hits[-1]>resets[-1])==(g=='reset_present')
            if len(hits)==2:assert events[hits[0]][1]!=events[hits[-1]][1]
            first=events[hits[0]][1];last=next(v for k,v in reversed(events) if k!=3)
            guesses=dict(unknown=4,constant0=0,first_match=first,last_value=last,ignore_reset=oracle(events,q,True),oracle=expected)
            for b,pred in guesses.items():base[g][b]+=pred==label
            sizes[g]+=1;count+=1
        baselines[split]={g:{b:v/sizes[g] for b,v in x.items()} for g,x in base.items()}
    assert all(b['oracle']==1 for cell in baselines.values() for b in cell.values())
    return dict(rows=count,families=len(ownership),baselines=baselines,passed=True)


def metrics(scores,rows):
    labels=torch.tensor([r['label'] for r in rows]);correct=scores.argmax(-1)==labels
    loss=scores.logsumexp(-1)-scores.gather(1,labels[:,None]).squeeze(1)
    return {g:dict(accuracy=float(correct[[r['group']==g for r in rows]].double().mean()),loss=float(loss[[r['group']==g for r in rows]].mean())) for g in GROUPS}


def evaluate(model,rows,fn=None):
    x,q,y=encode(rows)
    with torch.no_grad():
        scores=torch.cat([(model(x[i:i+256],q[i:i+256]) if fn is None else fn(model,x[i:i+256],q[i:i+256])) for i in range(0,len(rows),256)])
    return metrics(scores,rows),scores


def preflight(payload):
    data=audit_data(payload);cases=[]
    x,q,y=encode(payload['pools']['train_6'][:12])
    assert bool(((y+1)%5!=y).all())
    for seed in SEEDS:
        models=pair(seed)
        assert all(torch.equal(v,models['gated'].state_dict()[k]) for k,v in models['delta'].state_dict().items())
        for arm,m in models.items():
            a=m(x,q);b=reference(m,x,q)
            ga=torch.autograd.grad(F.cross_entropy(a,y),tuple(m.parameters()));gb=torch.autograd.grad(F.cross_entropy(b,y),tuple(m.parameters()))
            oe=float((a-b).abs().max().detach());ge=max(float((u-v).abs().max()) for u,v in zip(ga,gb))
            with torch.no_grad():bad=float((reference(m,x,q,True)-a).abs().max())
            assert max(oe,ge)<1e-10 and bad>1e-6
            cases.append(dict(seed=seed,arm=arm,output_error=oe,gradient_error=ge,omit_subtraction_effect=bad))
    return dict(data=data,cases=cases,shifted_label_control=True)


def profile():
    torch.set_num_threads(1);rows=[]
    for arm,m in pair(2709).items():
        x=torch.zeros(64,8,8,dtype=torch.float64);x[:,:,0]=1;x[:,:,3]=1;q=torch.zeros(64,dtype=torch.long);y=q.clone()
        opt=torch.optim.Adam(m.parameters(),lr=.003);start=time.perf_counter()
        loss=F.cross_entropy(m(x,q),y);loss.backward();opt.step()
        rows.append(dict(arm=arm,parameters=sum(p.numel() for p in m.parameters()),optimizer_entries=len(opt.state),seconds=time.perf_counter()-start,loss=float(loss.detach())))
    p=ROOT/'results/learned_memory_control_profile.json';assert not p.exists()
    write(p,dict(device='cpu',dtype='float64',batch=64,length=8,peak_gpu_allocated_bytes=0,peak_gpu_reserved_bytes=0,cpu_peak_memory_measured=False,source_sha256=digest(Path(__file__)),rows=rows));print(rows)


def decision(runs):
    fit={a:all(v['status']=='trained' and all(v['metrics'][f'{s}_{n}'][g]['accuracy']>=.99 for s in ('train','validation') for n in (6,8) for g in GROUPS) for v in runs if v['arm']==a) for a in ARMS}
    reliability={a:fit[a] and all(v['metrics'][f'test_{n}'][g]['accuracy']>=(.9 if n==12 else .99) for v in runs if v['arm']==a for n in (6,8,12) for g in GROUPS) for a in ARMS}
    gains=[]
    for seed in SEEDS:
        p={v['arm']:v for v in runs if v['seed']==seed}
        if any(v['status']!='trained' for v in p.values()):continue
        gains.append(dict(seed=seed,**{g:p['gated']['metrics']['test_12'][g]['accuracy']-p['delta']['metrics']['test_12'][g]['accuracy'] for g in GROUPS}))
    gain_rule=len(gains)==3 and all(min(v[g] for g in ('reset_present','reset_missing'))>=.10 and min(v[g] for g in ('retain','overwrite'))>=-.02 for v in gains)
    return dict(fit=fit,reliability=reliability,paired_gains=gains,reset_gain_without_retention_cost=gain_rule,adopt_gated=all(fit.values()) and reliability['gated'] and gain_rule)


def train():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rp=ROOT/'results/learned_memory_control.json';assert not rp.exists()
    history={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'results').glob('*.json')}
    payload=make_data();pools=payload['pools'];encoded={k:encode(v) for k,v in pools.items() if not k.startswith('test')}
    paths=['experiments/learned_memory_control.py','docs/protocols/learned-memory-control.md','results/learned_memory_control_data.json']
    report=dict(phase='preflight',provenance={p:digest(ROOT/p) for p in paths},historical=history,torch=torch.__version__,runs=[]);write(rp,report)
    report['preflight']=preflight(payload);write(rp,report)
    directory=ROOT/'results/learned_memory_control_checkpoints';directory.mkdir(exist_ok=False)
    for index,seed in enumerate(SEEDS):
        models=pair(seed)
        for arm in ARMS[::1 if index%2==0 else -1]:
            model=models[arm];opt=torch.optim.Adam(model.parameters(),lr=.003);gen=torch.Generator().manual_seed(seed+1000);stream=hashlib.sha256();tick=time.perf_counter()
            run=dict(seed=seed,arm=arm,status='running',parameters=sum(p.numel() for p in model.parameters()),initial_sha256=hash_state(model.state_dict()),shared_sha256=hash_state({k:v for k,v in model.state_dict().items() if not k.startswith('alpha.')}),curve=[],metrics={});report['runs'].append(run);write(rp,report)
            try:
                for step in range(1,STEPS+1):
                    n=6 if step%2 else 8;x,q,y=encoded[f'train_{n}'];block=len(y)//4
                    ids=torch.cat([torch.randint(block,(16,),generator=gen)+j*block for j in range(4)])
                    stream.update(json.dumps(ids.tolist()).encode());opt.zero_grad();loss=F.cross_entropy(model(x[ids],q[ids]),y[ids]);assert bool(torch.isfinite(loss));loss.backward()
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters());opt.step()
                    if step%500==0:run['curve'].append(dict(step=step,minibatch_loss=loss.item()));print(dict(seed=seed,arm=arm,step=step,loss=loss.item()),flush=True);write(rp,report)
                run['status']='trained'
            except Exception as exc:run.update(status='failed',failure=repr(exc))
            cp=directory/f'{seed}_{arm}.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step),cp)
            run.update(step=step,seconds=time.perf_counter()-tick,stream_sha256=stream.hexdigest(),checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=digest(cp),examples=step*64,event_records=sum(64*(6 if s%2 else 8) for s in range(1,step+1)))
            if run['status']=='trained':
                for key in encoded:run['metrics'][key],_=evaluate(model,pools[key])
            write(rp,report)
    report['phase']='all_attempts_finished_before_test';write(ROOT/'results/learned_memory_control_pretest.json',report)
    for run in report['runs']:
        if run['status']!='trained':continue
        model=pair(run['seed'])[run['arm']];model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model']);scores={}
        for key,rows in pools.items():
            metric,scores[key]=evaluate(model,rows)
            if key in run['metrics']:assert metric==run['metrics'][key]
            run['metrics'][key]=metric
        sp=directory/f"{run['seed']}_{run['arm']}_scores.pt";torch.save(scores,sp);run.update(scores=str(sp.relative_to(ROOT)),scores_sha256=digest(sp));write(rp,report)
    assert all(digest(ROOT/p)==h for p,h in history.items())
    report.update(phase='complete',decision=decision(report['runs']));write(rp,report);print(report['decision'],flush=True)


def audit():
    torch.set_num_threads(1);r=json.loads((ROOT/'results/learned_memory_control.json').read_text());assert r['phase']=='complete'
    assert all(digest(ROOT/p)==h for p,h in r['provenance'].items());assert all(digest(ROOT/p)==h for p,h in r['historical'].items())
    data=json.loads((ROOT/'results/learned_memory_control_data.json').read_text());assert audit_data(data)==r['preflight']['data']
    pre=json.loads((ROOT/'results/learned_memory_control_pretest.json').read_text());assert pre['phase']=='all_attempts_finished_before_test'
    cells=[]
    for seed in SEEDS:
        paired=[v for v in r['runs'] if v['seed']==seed];assert len(paired)==2 and len({v['shared_sha256'] for v in paired})==1 and len({v['stream_sha256'] for v in paired})==1
        gen=torch.Generator().manual_seed(seed+1000);stream=hashlib.sha256()
        for step in range(1,STEPS+1):
            block=len(data['pools'][f'train_{6 if step%2 else 8}'])//4
            stream.update(json.dumps(torch.cat([torch.randint(block,(16,),generator=gen)+j*block for j in range(4)]).tolist()).encode())
        for run in paired:
            if run['status']!='trained':continue
            assert stream.hexdigest()==run['stream_sha256']
            m=pair(seed)[run['arm']];assert hash_state(m.state_dict())==run['initial_sha256']
            cp=ROOT/run['checkpoint'];assert digest(cp)==run['checkpoint_sha256'];state=torch.load(cp,weights_only=True)
            assert state['step']==STEPS and all(v['step'].item()==STEPS for v in state['optimizer']['state'].values())
            prior=next(v for v in pre['runs'] if v['seed']==seed and v['arm']==run['arm']);assert prior['checkpoint_sha256']==run['checkpoint_sha256'] and not any(k.startswith('test') for k in prior['metrics'])
            m.load_state_dict(state['model']);sp=ROOT/run['scores'];assert digest(sp)==run['scores_sha256'];saved=torch.load(sp,weights_only=True)
            for key,rows in data['pools'].items():
                _,scores=evaluate(m,rows,reference);err=float((scores-saved[key]).abs().max());changes=int((scores.argmax(-1)!=saved[key].argmax(-1)).sum())
                # Independent loss using library CE and group selection by indices.
                y=torch.tensor([v['label'] for v in rows]);loss=F.cross_entropy(scores,y,reduction='none');me=0.
                for group in GROUPS:
                    idx=torch.tensor([i for i,v in enumerate(rows) if v['group']==group]);acc=float((scores[idx].argmax(-1)==y[idx]).double().mean());le=float(loss[idx].mean())
                    me=max(me,abs(acc-run['metrics'][key][group]['accuracy']),abs(le-run['metrics'][key][group]['loss']))
                assert err<1e-10 and changes==0 and me<1e-10
                cells.append(dict(seed=seed,arm=run['arm'],split=key,predictions=len(rows),logit_error=err,metric_error=me,argmax_changes=changes))
    assert decision(r['runs'])==r['decision']
    p=ROOT/'results/learned_memory_control_audit.json';assert not p.exists()
    out=dict(passed=True,cells=cells,evaluations=len(cells),predictions=sum(v['predictions'] for v in cells),max_logit_error=max(v['logit_error'] for v in cells),max_metric_error=max(v['metric_error'] for v in cells),report_sha256=digest(ROOT/'results/learned_memory_control.json'));write(p,out);print({k:v for k,v in out.items() if k!='cells'})


if __name__=='__main__':
    {'profile':profile,'train':train,'audit':audit}[sys.argv[1]]()
