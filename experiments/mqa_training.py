"""Paired from-scratch MHA/MQA study; preserve earlier executed sources."""
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from reserved_cycle import model_new as template_model
from query_exposure import family, expand_rows, measure
from positional_comparison import rotate
from position_training import weight_hash
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(1411,1423,1451,1481,1499)
PRIORS=('repeated_key_data.json','gelu_comparison_data.json','ffn_width_data.json','query_exposure_data.json')


class Attention(nn.Module):
    def __init__(self, original, groups):
        super().__init__();self.groups=groups
        weight,bias=original.in_proj_weight,original.in_proj_bias
        for name,start,count in [('q',0,16),('k',16,groups*8),('v',32,groups*8)]:
            setattr(self,name+'_weight',nn.Parameter(weight[start:start+count].detach().clone()))
            setattr(self,name+'_bias',nn.Parameter(bias[start:start+count].detach().clone()))
        self.out_proj=copy.deepcopy(original.out_proj)


class Block(nn.Module):
    def __init__(self, original, groups):
        super().__init__()
        self.norm1=original.norm1;self.norm2=original.norm2
        self.linear1=original.linear1;self.linear2=original.linear2
        self.attn=Attention(original.self_attn,groups)


class Decoder(nn.Module):
    def __init__(self, original, groups):
        super().__init__();self.groups=groups
        self.embedding=original.embedding
        self.blocks=nn.ModuleList([Block(b,groups) for b in original.blocks])
        self.final_norm=original.final_norm;self.output=original.output

    def forward(self,ids):
        return forward(self,ids)


def model_new(group='mha'):
    assert group in ('mha','mqa')
    return Decoder(template_model(),2 if group=='mha' else 1)


def forward(model,ids,group=None,capture=False):
    n=ids.shape[1];x=model.embedding(ids);positions=torch.arange(n)+4
    mask=torch.ones(n,n,dtype=torch.bool).triu(1);cache=[]
    for block in model.blocks:
        z=block.norm1(x);m=block.attn
        q,k,v=[F.linear(z,getattr(m,t+'_weight'),getattr(m,t+'_bias')) for t in ('q','k','v')]
        q=q.reshape(len(ids),n,2,8).transpose(1,2)
        k,v=[t.reshape(len(ids),n,model.groups,8).transpose(1,2) for t in (k,v)]
        q,k=rotate(q,positions),rotate(k,positions);cache.append((k,v))
        k,v=[t.repeat_interleave(2//model.groups,dim=1) for t in (k,v)]
        a=(q@k.transpose(-1,-2)/(8**.5)).masked_fill(mask,-torch.inf).softmax(-1)
        h=(a@v).transpose(1,2).reshape(len(ids),n,16)
        x=x+m.out_proj(h)
        x=x+block.linear2(F.relu(block.linear1(block.norm2(x))))
    logits=model.output(model.final_norm(x))
    return (logits,cache) if capture else logits


def hash_state(state):
    return hashlib.sha256(json.dumps({k:v.tolist() for k,v in state.items()},sort_keys=True).encode()).hexdigest()


def shared_hash(model):
    return hash_state({k:v for k,v in model.state_dict().items() if not k.endswith(('.k_weight','.k_bias','.v_weight','.v_bias'))})


def retained_kv_hash(model):
    return hash_state({k:v[:8] for k,v in model.state_dict().items() if k.endswith(('.k_weight','.k_bias','.v_weight','.v_bias'))})


def make_data(path):
    rng=random.Random(20270108);seen=set();hashes={};datasets={};identities={}
    for name in PRIORS:
        p=ROOT/'results'/name;prior=json.loads(p.read_text());hashes[name]=digest(p)
        seen.update(tuple(tuple(pair) for pair in f) for fs in prior['families'].values() for f in fs)
    for split in ('train','validation','test'):
        key=f'{split}_3';datasets[key]=prior['datasets'][key];identities[key]=prior['families'][key]
    for n in (4,6):
        counts={'train':800,'validation':100,'test':200} if n==4 else {'test':200}
        capacity=19800 if n==4 else 237600
        assert sum(counts.values())<=capacity-sum(len(f)==n for f in seen)
        for split,count in counts.items():
            rows=[];families=[]
            for _ in range(count):
                for attempt in range(100000):
                    keys=rng.sample(range(12),n-1);q=keys[0]
                    pairs=list(zip(keys+[q],rng.sample(range(12,18),n)));signature=family(pairs)
                    if signature not in seen:break
                else:raise RuntimeError('Fresh-family support exhausted')
                seen.add(signature);families.append(signature);orders=[];used=set()
                while len(orders)<6:
                    order=tuple(rng.sample(pairs,n));reverse=order[::-1]
                    if order in used:continue
                    orders.extend((order,reverse));used.update((order,reverse))
                for shift in range(6):
                    for order in orders:
                        renamed=[(k,12+(v-12+shift)%6) for k,v in order]
                        label=next(v for k,v in reversed(renamed) if k==q)
                        rows.append(([t for pair in renamed for t in pair]+[18,q],label))
            key=f'{split}_{n}';datasets[key]=rows;identities[key]=families
    payload=dict(generator_seed=20270108,prior_hashes=hashes,datasets=datasets,families=identities)
    path.write_text(json.dumps(payload,separators=(',',':'))+'\n');return payload


def training_batch(x,y,group):
    return x,y


def evaluate(model,data,group):
    x,y=data
    with torch.no_grad():scores=torch.cat([forward(model,c,group)[:,-1] for c in x.split(256)])
    return measure(scores,y,x),scores


def decision(runs):
    if any(r['status']!='trained' for r in runs):return dict(keep=False,reason='training failure')
    pairs=[tuple(next(r for r in runs if r['seed']==seed and r['group']==g) for g in ('mha','mqa')) for seed in SEEDS]
    checks={'parameters_and_cache':all(b['parameters']==5088 and m['parameters']==4544 for b,m in pairs)};gains={}
    for n in (3,4,6):
        for kind in ('repeated','unique'):
            diff=[m['test'][str(n)][kind+'_accuracy']-b['test'][str(n)][kind+'_accuracy'] for b,m in pairs]
            gains[f'{n}_{kind}']=diff
            checks[f'{n}_{kind}_floor']=all(m['test'][str(n)][kind+'_accuracy']>=(.9 if n==6 else .99) for b,m in pairs)
            if n>3:checks[f'{n}_{kind}_mean_loss1']=sum(diff)/5>=-.01
            if n==6:checks[f'{n}_{kind}_seed_loss5']=min(diff)>=-.05
    for b,m in pairs:
        checks['parameters_and_cache'] &= all(b['cache_bytes'][str(n)]==2*m['cache_bytes'][str(n)] for n in (3,4,6))
    return dict(keep=all(checks.values()),checks=checks,paired_gains=gains)


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    ip=ROOT/'results/mqa_training_data.json'; rp=ROOT/'results/mqa_training.json'; directory=ROOT/'results/mqa_training_checkpoints'; directory.mkdir(exist_ok=True)
    provenance={n:digest(ROOT/'experiments'/n) for n in ('mqa_training.py','audit_mqa_training.py','verify_mqa_training.py','audit_kv_cache_equivalence.py','query_exposure.py','ffn_width.py','verify_ffn_width.py','audit_ffn_width.py','audit_gelu_comparison.py','repeated_key.py','audit_repeated_key.py','audit_positional_comparison.py','activation_paths.py','positional_comparison.py','reserved_cycle.py','compare_depth.py','position_training.py','position_distractors.py')}
    provenance['protocol']=digest(ROOT/'docs/mqa-training-protocol.md')
    if rp.exists():
        report=json.loads(rp.read_text()); assert report['provenance']==provenance and report['input_sha256']==digest(ip)
        if report['phase']=='complete': print('Already complete.'); return
        payload=json.loads(ip.read_text())
    else:
        assert not ip.exists(); payload=make_data(ip)
        report=dict(phase='training',provenance=provenance,input_sha256=digest(ip),torch=torch.__version__,runs=[])
    from audit_mqa_training import audit_inputs
    verification=json.loads((ROOT/'results/mqa_training_verification.json').read_text())
    assert verification['passed'] and verification['source_sha256']==digest(ROOT/'experiments/mqa_training.py')
    report['input_checks']=audit_inputs(payload)
    expanded={k:expand_rows(rows) for k,rows in payload['datasets'].items()}
    data={k:(torch.tensor([x for x,y in rows]),torch.tensor([y for x,y in rows])) for k,rows in expanded.items()}
    def save():
        tmp=rp.with_suffix('.tmp'); tmp.write_text(json.dumps(report,indent=2)+'\n'); tmp.replace(rp)
    save()
    for seed in SEEDS:
        for group in ('mha','mqa'):
            run=next((r for r in report['runs'] if r['seed']==seed and r['group']==group),None)
            path=directory/f'{group}_{seed}.pt'
            if run and run['status'] in ('trained','failed'):
                assert digest(path)==run['checkpoint_sha256']; continue
            torch.manual_seed(seed); model=model_new(group); assert sum(p.numel() for p in model.parameters())==(5088 if group=='mha' else 4544)
            initial=weight_hash(model); opt=torch.optim.Adam(model.parameters(),lr=.001)
            g3=torch.Generator().manual_seed(seed+1000); g4=torch.Generator().manual_seed(seed+2000); start=0
            if path.exists():
                state=torch.load(path,weights_only=True); assert state['provenance']==provenance
                model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); g3.set_state(state['g3']); g4.set_state(state['g4']); torch.set_rng_state(state['rng'])
                run=state['run']; start=run['step']; report['runs']=[r for r in report['runs'] if (r['seed'],r['group'])!=(seed,group)]; report['runs'].append(run)
            elif run is None:
                run=dict(seed=seed,group=group,initial_sha256=initial,shared_initial_sha256=shared_hash(model),retained_kv_initial_sha256=retained_kv_hash(model),parameters=sum(p.numel() for p in model.parameters()),status='running',step=0,curve=[],tokens=0,attention_pairs=0,elapsed_seconds=0.,query_counts={str(n):dict(repeated=0,unique=0) for n in (3,4)})
                report['runs'].append(run)
                run['curve'].append(dict(step=0,validation={str(n):evaluate(model,data[f'validation_{n}'],group)[0] for n in (3,4)}))
            if start==0 and not path.exists():
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),g3=g3.get_state(),g4=g4.get_state(),rng=torch.get_rng_state(),run=dict(run),provenance=provenance),path)
            h3=hashlib.sha256(); h4=hashlib.sha256()
            rg3=torch.Generator().manual_seed(seed+1000); rg4=torch.Generator().manual_seed(seed+2000)
            for _ in range(start):
                h3.update(json.dumps(torch.randint(len(data['train_3'][0]),(64,),generator=rg3).tolist()).encode())
                h4.update(json.dumps(torch.randint(len(data['train_4'][0]),(64,),generator=rg4).tolist()).encode())
            tick=time.perf_counter()
            for step in range(start+1,6001):
                i3=torch.randint(len(data['train_3'][0]),(64,),generator=g3); i4=torch.randint(len(data['train_4'][0]),(64,),generator=g4)
                h3.update(json.dumps(i3.tolist()).encode()); h4.update(json.dumps(i4.tolist()).encode())
                n=4 if step%2==0 else 3; idx=i4 if n==4 else i3; x,y=data[f'train_{n}']
                bx,by=training_batch(x[idx],y[idx],group)
                repeated_count=int(((bx[:,:-2:2]==bx[:,-1:]).sum(-1)==2).sum())
                run['query_counts'][str(n)]['repeated']+=repeated_count
                run['query_counts'][str(n)]['unique']+=64-repeated_count
                opt.zero_grad(); loss=F.cross_entropy(forward(model,bx,group)[:,-1],by); finite=bool(torch.isfinite(loss))
                if finite:
                    loss.backward(); finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                if finite: opt.step()
                else: run.update(status='failed',failure_step=step)
                run['tokens']+=64*(2*n+2); run['attention_pairs']+=64*(2*n+2)**2
                if step%1000==0 or not finite:
                    run['curve'].append(dict(step=step,validation={str(n):evaluate(model,data[f'validation_{n}'],group)[0] for n in (3,4)}))
                    run.update(step=step,stream3_sha256=h3.hexdigest(),stream4_sha256=h4.hexdigest())
                    if step==6000 and finite: run['status']='trained'
                    run['elapsed_seconds']+=time.perf_counter()-tick
                    state=dict(model=model.state_dict(),optimizer=opt.state_dict(),g3=g3.get_state(),g4=g4.get_state(),rng=torch.get_rng_state(),run=dict(run),provenance=provenance)
                    tmp=path.with_suffix('.tmp'); torch.save(state,tmp); tmp.replace(path)
                    run.update(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=digest(path)); save(); tick=time.perf_counter()
                    print(json.dumps(dict(seed=seed,group=group,step=step,validation=run['curve'][-1]['validation'])),flush=True)
                if not finite: break
    assert len(report['runs'])==10; report['phase']='all_attempts_finished_before_test'; save(); details=[]
    for run in report['runs']:
        if run['status']!='trained': continue
        cp=ROOT/run['checkpoint']; assert digest(cp)==run['checkpoint_sha256']
        model=model_new(run['group']); model.load_state_dict(torch.load(cp,weights_only=True)['model']); run['test']={}; scores={}
        run['cache_bytes']={}
        for n in (3,4,6):
            run['test'][str(n)],scores[str(n)]=evaluate(model,data[f'test_{n}'],run['group'])
            with torch.no_grad(): _,cache=forward(model,data[f'test_{n}'][0][:1],capture=True)
            size=sum(t.untyped_storage().nbytes() for pair in cache for t in pair)
            assert size==2*model.groups*(2*n+2)*16*8
            run['cache_bytes'][str(n)]=size
        details.append(dict(seed=run['seed'],group=run['group'],scores=scores)); save()
    dp=ROOT/'results/mqa_training_scores.pt'; torch.save(details,dp)
    report.update(phase='complete',scores_sha256=digest(dp),decision=decision(report['runs'])); save(); print(json.dumps(report['decision']))


if __name__=='__main__': main()
