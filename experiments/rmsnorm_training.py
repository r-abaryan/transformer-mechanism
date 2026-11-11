"""Matched centering comparison; reuse archived MQA forward and task metrics."""
import hashlib
import json
import random
import time

import torch
from torch import nn
from torch.nn import functional as F
from mqa_training import ROOT, model_new as mqa_model, forward, hash_state
from query_exposure import family, expand_rows, measure
from position_distractors import digest

SEEDS=(1511,1523,1543,1567,1597)
PRIORS=('repeated_key_data.json','gelu_comparison_data.json','ffn_width_data.json','query_exposure_data.json','mqa_training_data.json')


def model_new(group):
    assert group in ('ln','rms')
    model=mqa_model('mqa')
    def layer():
        return nn.LayerNorm(16,eps=1e-5,bias=False,dtype=torch.float64) if group=='ln' else nn.RMSNorm(16,eps=1e-5,dtype=torch.float64)
    for block in model.blocks:block.norm1=layer();block.norm2=layer()
    model.final_norm=layer()
    return model


def parameter_hash(model):return hash_state(model.state_dict())


def make_data(path):
    rng=random.Random(20270110);seen=set();hashes={};datasets={};identities={}
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
    payload=dict(generator_seed=20270110,prior_hashes=hashes,datasets=datasets,families=identities)
    path.write_text(json.dumps(payload,separators=(',',':'))+'\n');return payload


def evaluate(model,data):
    x,y=data
    with torch.inference_mode():scores=torch.cat([forward(model,c)[:,-1] for c in x.split(256)])
    return measure(scores,y,x),scores


def milestone(curve):
    return next((point['step'] for point in curve if all(point['validation'][str(n)][k+'_accuracy']>=.99 for n in (3,4) for k in ('repeated','unique'))),None)


def decision(runs):
    if len(runs)!=10 or any(r['status']!='trained' for r in runs):return dict(keep=False,reason='training failure or incomplete attempts')
    paired=[tuple(next(r for r in runs if r['seed']==seed and r['group']==g) for g in ('ln','rms')) for seed in SEEDS]
    checks=dict(equal_parameters=all(b['parameters']==m['parameters']==4464 for b,m in paired))
    gains={}
    for n in (3,4,6):
        for kind in ('repeated','unique'):
            diff=[m['test'][str(n)][kind+'_accuracy']-b['test'][str(n)][kind+'_accuracy'] for b,m in paired]
            gains[f'{n}_{kind}']=diff
            checks[f'{n}_{kind}_floor']=all(m['test'][str(n)][kind+'_accuracy']>=(.9 if n==6 else .99) for b,m in paired)
            if n==6:
                checks[f'6_{kind}_mean']=sum(diff)/5>=(.01 if kind=='unique' else -.005)
                checks[f'6_{kind}_seed_loss1']=min(diff)>=-.01
    return dict(keep=all(checks.values()),checks=checks,paired_gains=gains)


def provenance():
    prior_path=ROOT/'results/mqa_training.json';prior=json.loads(prior_path.read_text())
    for name,h in prior['provenance'].items():assert digest(ROOT/('docs/mqa-training-protocol.md' if name=='protocol' else 'experiments/'+name))==h
    paths=('experiments/rmsnorm_training.py','experiments/verify_rmsnorm_training.py','experiments/audit_rmsnorm_training.py','docs/rmsnorm-training-protocol.md')
    return dict(sources={p:digest(ROOT/p) for p in paths},baseline_sha256=digest(prior_path),baseline_source_hashes=prior['provenance'])


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    ip=ROOT/'results/rmsnorm_training_data.json';rp=ROOT/'results/rmsnorm_training.json'
    directory=ROOT/'results/rmsnorm_training_checkpoints';directory.mkdir(exist_ok=True);prov=provenance()
    vp=ROOT/'results/rmsnorm_training_verification.json';verification=json.loads(vp.read_text())
    assert verification['passed'] and verification['provenance']==prov
    if rp.exists():
        report=json.loads(rp.read_text());assert report['provenance']==prov and report['input_sha256']==digest(ip) and report['verification_sha256']==digest(vp)
        if report['phase']=='complete':print('Already complete.');return
        payload=json.loads(ip.read_text())
    else:
        assert not ip.exists();payload=make_data(ip)
        report=dict(phase='training',provenance=prov,verification_sha256=digest(vp),input_sha256=digest(ip),torch=torch.__version__,device='cpu',dtype='float64',threads=1,runs=[])
    from audit_rmsnorm_training import audit_inputs
    report['input_checks']=audit_inputs(payload)
    expanded={k:expand_rows(rows) for k,rows in payload['datasets'].items()}
    data={k:(torch.tensor([x for x,y in rows]),torch.tensor([y for x,y in rows])) for k,rows in expanded.items()}
    def save():
        temporary=rp.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(rp)
    save()
    for seed in SEEDS:
        for group in ('ln','rms'):
            run=next((r for r in report['runs'] if r['seed']==seed and r['group']==group),None)
            path=directory/f'{group}_{seed}.pt'
            if run and run['status'] in ('trained','failed'):
                assert digest(path)==run['checkpoint_sha256'];continue
            torch.manual_seed(seed);model=model_new(group);assert sum(p.numel() for p in model.parameters())==4464
            initial=parameter_hash(model);opt=torch.optim.Adam(model.parameters(),lr=.001)
            generators={n:torch.Generator().manual_seed(seed+offset) for n,offset in ((3,1000),(4,2000))};start=0
            if path.exists():
                state=torch.load(path,weights_only=True);assert state['provenance']==prov
                model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
                for n in (3,4):generators[n].set_state(state[f'g{n}'])
                run=state['run'];start=run['step'];run.update(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=digest(path))
                report['runs']=[r for r in report['runs'] if (r['seed'],r['group'])!=(seed,group)];report['runs'].append(run)
            elif run is None:
                run=dict(seed=seed,group=group,initial_sha256=initial,parameters=4464,status='running',step=0,curve=[],tokens=0,attention_pairs=0,elapsed_seconds=0.,query_counts={str(n):dict(repeated=0,unique=0) for n in (3,4)})
                run['curve'].append(dict(step=0,validation={str(n):evaluate(model,data[f'validation_{n}'])[0] for n in (3,4)}));report['runs'].append(run)
            hashes={n:hashlib.sha256() for n in (3,4)}
            replay={n:torch.Generator().manual_seed(seed+offset) for n,offset in ((3,1000),(4,2000))}
            for _ in range(start):
                for n in (3,4):hashes[n].update(json.dumps(torch.randint(len(data[f'train_{n}'][0]),(64,),generator=replay[n]).tolist()).encode())
            tick=time.perf_counter();window_loss=0.;window_batches=0
            for step in range(start+1,6001):
                draw={n:torch.randint(len(data[f'train_{n}'][0]),(64,),generator=generators[n]) for n in (3,4)}
                for n in (3,4):hashes[n].update(json.dumps(draw[n].tolist()).encode())
                n=3 if step%2 else 4;x,y=data[f'train_{n}'];bx,by=x[draw[n]],y[draw[n]]
                repeated=int(((bx[:,:-2:2]==bx[:,-1:]).sum(-1)==2).sum())
                run['query_counts'][str(n)]['repeated']+=repeated;run['query_counts'][str(n)]['unique']+=64-repeated
                opt.zero_grad();loss=F.cross_entropy(forward(model,bx)[:,-1],by);finite=bool(torch.isfinite(loss))
                if finite:
                    loss.backward();finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                if finite:opt.step();window_loss+=loss.item();window_batches+=1
                else:run.update(status='failed',failure_step=step)
                run['tokens']+=64*(2*n+2);run['attention_pairs']+=64*(2*n+2)**2
                if step%1000==0 or not finite:
                    point=dict(step=step,training_window_mean_loss=window_loss/max(1,window_batches),finite_training=finite,
                               validation={str(n):evaluate(model,data[f'validation_{n}'])[0] for n in (3,4)})
                    run['curve'].append(point);window_loss=0.;window_batches=0
                    run.update(step=step,stream3_sha256=hashes[3].hexdigest(),stream4_sha256=hashes[4].hexdigest())
                    if step==6000 and finite:run['status']='trained'
                    run['elapsed_seconds']+=time.perf_counter()-tick
                    state=dict(model=model.state_dict(),optimizer=opt.state_dict(),g3=generators[3].get_state(),g4=generators[4].get_state(),run=dict(run),provenance=prov)
                    temporary=path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(path)
                    run.update(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=digest(path));save();tick=time.perf_counter()
                    print(json.dumps(dict(seed=seed,group=group,step=step,validation_four=point['validation']['4']['accuracy'])),flush=True)
                if not finite:break
    assert len(report['runs'])==10;report['phase']='all_attempts_finished_before_test';save();details=[]
    for run in report['runs']:
        if run['status']!='trained':continue
        cp=ROOT/run['checkpoint'];assert digest(cp)==run['checkpoint_sha256']
        model=model_new(run['group']).eval();model.load_state_dict(torch.load(cp,weights_only=True)['model'])
        run['test']={};scores={}
        for n in (3,4,6):run['test'][str(n)],scores[str(n)]=evaluate(model,data[f'test_{n}'])
        run['sampled_validation99_step']=milestone(run['curve'])
        details.append(dict(seed=run['seed'],group=run['group'],scores=scores));save()
    dp=ROOT/'results/rmsnorm_training_scores.pt';assert not dp.exists();torch.save(details,dp)
    report.update(phase='complete',scores_sha256=digest(dp),decision=decision(report['runs']));save();print(json.dumps(report['decision']))


if __name__=='__main__':main()
