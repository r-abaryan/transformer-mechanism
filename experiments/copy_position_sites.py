"""Factorial learning intervention: positional exposure in self versus cross attention.

Frozen predecessor helpers own labels, sampling, metrics and model construction.
The forward is separate because the executed predecessor must remain immutable.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from copy_position_exposure import (ROOT, digest, write, positions, specs, measures,
                                    inputs, generate, sample, make_model, hash_state)
from copy_rope_model import attention, rotate, matrix_rotation
from audit_kv_cache_equivalence import norm

OUT = ROOT/'results/copy_position_sites'
ARMS = ('ordinary', 'self_only', 'cross_only', 'both')
SITES = {'ordinary': (False, False), 'self_only': (True, False),
         'cross_only': (False, True), 'both': (True, True)}
DEV = 3761
CONFIRM = (3767, 3779, 3793)
STAGES = (6000, 12000)


def forward(model, content, prefix, self_pos=None, cross_pos=None, reference=False):
    n = content.shape[1]
    def layout(pos):
        if pos is None:
            return torch.arange(n+1), torch.arange(prefix.shape[1])
        assert len(pos)==n+1 and bool((pos[1:]>pos[:-1]).all())
        tail=pos[-1]+torch.arange(1,max(0,prefix.shape[1]-len(pos))+1)
        return pos, torch.cat((pos,tail))[:prefix.shape[1]]
    ep,dp=layout(self_pos);ck,cq=layout(cross_pos)
    base=model.base;enc,dec=base.encoder,base.decoder
    source=torch.cat((content,torch.full((len(content),1),2,dtype=torch.long)),1)
    s,t=base.source(source)*math.sqrt(8),base.target(prefix)*math.sqrt(8)
    sp,tp=torch.zeros_like(source,dtype=torch.bool),torch.zeros_like(prefix,dtype=torch.bool)
    ln=norm if reference else lambda x,l:l(x)
    def ff(l,x):
        return (x@l.linear1.weight.T+l.linear1.bias).clamp_min(0)@l.linear2.weight.T+l.linear2.bias if reference else l.linear2(F.relu(l.linear1(x)))
    memory=ln(s+attention(enc.self_attn,s,s,s,sp,ep,ep,reference=reference),enc.norm1)
    memory=ln(memory+ff(enc,memory),enc.norm2)
    x=ln(t+attention(dec.self_attn,t,t,t,tp,dp,dp,True,reference),dec.norm1)
    layer=dec.multihead_attn;h=layer.num_heads;d=layer.embed_dim//h
    q,k,v=[(a@w.T+b if reference else F.linear(a,w,b)).reshape(len(content),a.shape[1],h,d).transpose(1,2)
           for a,w,b in zip((x,memory,memory),layer.in_proj_weight.chunk(3),layer.in_proj_bias.chunk(3))]
    rot=matrix_rotation if reference else rotate
    scores=rot(q,cq)@rot(k,ck).transpose(-1,-2)/math.sqrt(d)
    if reference:
        ex=(scores-scores.max(-1,keepdim=True).values).exp();weights=ex/ex.sum(-1,keepdim=True)
    else:weights=scores.softmax(-1)
    mixed=(weights@v).transpose(1,2).reshape_as(x)
    cross=mixed@layer.out_proj.weight.T+layer.out_proj.bias
    x=ln(x+cross,dec.norm2)
    x=ln(x+ff(dec,x),dec.norm3)
    return base.output(x),scores[:,:,0]


def evaluate(model, rows, arm, pos=None, reference=False):
    a,b=SITES[arm];sp=pos if a else None;cp=pos if b else None
    zs=[];ss=[];outputs=[]
    with torch.no_grad():
        for x in torch.tensor(rows).split(64):
            prefix,_=inputs(x)
            z,s=forward(model,x,prefix,sp,cp,reference);zs.append(z);ss.append(s)
            outputs.extend(generate(lambda xx,pp:forward(model,xx,pp,sp,cp,reference)[0][:,-1],x,x.shape[1]+3))
    z,s=torch.cat(zs),torch.cat(ss)
    return measures(rows,z,s,outputs),dict(logits=z,scores=s,generated=outputs)


def evaluate_run(model,data,arm,long=False):
    metrics={};saved={}
    for name,spec in specs(data,long).items():
        pos=torch.tensor(spec['positions']) if spec['positions'] is not None else None
        metrics[name],saved[name]=evaluate(model,spec['rows'],arm,pos)
    return metrics,saved


def fit(metrics,arm):
    names=[f'train_{n}' for n in range(3,7)]
    if arm!='ordinary':names += [f'layout{j}_{n}' for n in range(3,7) for j in range(4)]
    return all(metrics[p][g]['exact_accuracy']>=.99 for p in names for g in ('distinct','repeated'))


def decision(runs):
    result=[]
    for seed in sorted({r['seed'] for r in runs}):
        rr={r['arm']:r for r in runs if r['seed']==seed};assert set(rr)==set(ARMS)
        fit_ok=all(fit(r['metrics'],arm) for arm,r in rr.items())
        short={arm:min(rr[arm]['metrics'][f'test_{n}'][g]['exact_accuracy']-rr['ordinary']['metrics'][f'test_{n}'][g]['exact_accuracy']
                       for n in range(3,7) for g in ('distinct','repeated')) for arm in ARMS}
        cells=[]
        for n,g in ((8,'distinct'),(8,'repeated'),(10,'repeated')):
            ms={arm:r['metrics'][f'test_{n}'][g] for arm,r in rr.items()}
            acc={arm:m['exact_accuracy'] for arm,m in ms.items()}
            rank={arm:m['outranked_fraction'] for arm,m in ms.items()}
            gains={arm:dict(copy=acc[arm]-acc['ordinary'],ranking=rank['ordinary']-rank[arm]) for arm in ARMS}
            sufficient={arm:acc[arm]>=.99 and acc[arm]>=acc['both']-.05 and gains[arm]['copy']>=.1 and gains[arm]['ranking']>=.1
                        for arm in ('self_only','cross_only')}
            cells.append(dict(length=n,group=g,accuracy=acc,outranked=rank,gains=gains,sufficient=sufficient,
                copy_interaction=acc['both']-acc['self_only']-acc['cross_only']+acc['ordinary']))
        replicated=fit_ok and min(short.values())>=-.05 and all(c['accuracy']['both']>=.99 and c['gains']['both']['copy']>=.1 and c['gains']['both']['ranking']>=.1 for c in cells)
        result.append(dict(seed=seed,fit=fit_ok,short_worst_change=short,cells=cells,combined_replicated=replicated))
    valid=bool(result) and all(r['combined_replicated'] for r in result)
    return dict(seeds=result,combined_replicated=valid,
        self_only_sufficient=valid and all(c['sufficient']['self_only'] for r in result for c in r['cells']),
        cross_only_sufficient=valid and all(c['sufficient']['cross_only'] for r in result for c in r['cells']))


def train_segment(directory,seed,arm,data,stop,start=None):
    torch.manual_seed(seed);model=make_model('rope');initial=hash_state(model.state_dict())
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    gen=torch.Generator().manual_seed(seed+1000);pg=torch.Generator().manual_seed(seed+2000)
    begin=0;previous=''
    if start:
        cp=torch.load(start,weights_only=True);model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer'])
        gen.set_state(cp['generator']);pg.set_state(cp['position_generator']);begin=cp['step'];previous=digest(start)
    stream=hashlib.sha256();pstream=hashlib.sha256();curve=[];tick=time.perf_counter()
    pools={p:torch.tensor(rows) for p,rows in data['pools'].items() if p.startswith('train_')}
    a,b=SITES[arm]
    for step in range(begin+1,stop+1):
        n,idx,_=sample(pools,gen,'mixed',step);pos=positions(n,pg)
        stream.update(json.dumps([n,idx.tolist()]).encode());pstream.update(json.dumps(pos.tolist()).encode())
        x=pools[f'train_{n}'][idx];prefix,labels=inputs(x);opt.zero_grad(set_to_none=True)
        z,_=forward(model,x,prefix,pos if a else None,pos if b else None)
        loss=F.cross_entropy(z.flatten(0,1),labels.flatten());assert bool(torch.isfinite(loss));loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        opt.step()
        if step%1000==0 or step==stop:
            row=dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-tick);curve.append(row)
            print(json.dumps(dict(seed=seed,arm=arm,**row)),flush=True)
    cp=directory/f'{seed}_{arm}_{stop}.pt'
    assert not cp.exists()
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=stop,generator=gen.get_state(),position_generator=pg.get_state()),cp)
    metrics,saved=evaluate_run(model.eval(),data,arm)
    scores=directory/f'{seed}_{arm}_{stop}_development_scores.pt';torch.save(saved,scores)
    return dict(seed=seed,arm=arm,step=stop,begin=begin,initial_sha256=initial,previous_sha256=previous,
        checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=digest(cp),scores=str(scores.relative_to(ROOT)),scores_sha256=digest(scores),
        metrics=metrics,fit=fit(metrics,arm),curve=curve,seconds=time.perf_counter()-tick,
        stream_sha256=stream.hexdigest(),position_stream_sha256=pstream.hexdigest())


def experiment(directory=OUT):
    directory=Path(directory);assert not (directory/'report.json').exists()
    lock=json.loads((directory/'lock.json').read_text());data=json.loads((directory/'data.json').read_text())
    assert all(digest(ROOT/p)==sha for p,sha in lock['protected'].items())
    report=dict(status='running',development=[],confirmation=[],selected_steps=None,lock_sha256=digest(directory/'lock.json'),device='cpu',dtype='float64',threads=1)
    write(directory/'report.json',report)
    for stop in lock['stages']:
        quartet=[]
        for arm in ARMS:
            prior=next((r for r in reversed(report['development']) if r['arm']==arm),None)
            run=train_segment(directory,lock['dev'],arm,data,stop,ROOT/prior['checkpoint'] if prior else None)
            quartet.append(run);report['development'].append(run);write(directory/'report.json',report)
        for key in ('initial_sha256','stream_sha256','position_stream_sha256'):assert len({r[key] for r in quartet})==1
        if all(r['fit'] for r in quartet):report['selected_steps']=stop;break
    write(directory/'development-selection.json',dict(selected_steps=report['selected_steps'],runs=report['development']))
    if report['selected_steps'] is None:
        report['status']='development_fit_failed';write(directory/'report.json',report);return report
    for seed in lock['confirm']:
        for arm in ARMS:
            run=train_segment(directory,seed,arm,data,report['selected_steps']);report['confirmation'].append(run);write(directory/'report.json',report)
    write(directory/'pretest.json',report)
    for run in report['confirmation']:
        model=make_model('rope');model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        metrics,saved=evaluate_run(model.eval(),data,run['arm'],True)
        for name,value in run['metrics'].items():assert metrics[name]==value
        dest=directory/f"{run['seed']}_{run['arm']}_final_scores.pt";torch.save(saved,dest)
        run.update(metrics=metrics,scores=str(dest.relative_to(ROOT)),scores_sha256=digest(dest));write(directory/'report.json',report)
    report.update(status='complete',decision=decision(report['confirmation']))
    assert all(digest(ROOT/p)==sha for p,sha in lock['protected'].items())
    write(directory/'report.json',report);return report


def prepare_data():
    OUT.mkdir(exist_ok=True);assert not (OUT/'data.json').exists()
    old=json.loads((ROOT/'results/copy_marker_training/report.json').read_text())['evaluation_pools']
    previous=json.loads((ROOT/'results/copy_position_exposure/data.json').read_text())['pools']
    pools={p:rows for p,rows in old.items() if p not in ('test_8','test_10')}
    seen={tuple(r) for source in (old,previous) for rows in source.values() for r in rows}
    gen=random.Random(2026100721)
    for n in (8,10):
        rows=[]
        for distinct in ((True,False) if n==8 else (False,)):
            part=[];wanted=64 if n==8 else 128
            for _ in range(100000):
                row=tuple(gen.sample(range(3,11),n)) if distinct else tuple(gen.choices(range(3,11),k=n))
                if (len(set(row))==n)!=distinct or row in seen:continue
                seen.add(row);part.append(list(row))
                if len(part)==wanted:break
            assert len(part)==wanted;rows+=part
        pools[f'test_{n}']=rows
    pg=torch.Generator().manual_seed(2026100722)
    layouts={str(n):[positions(n,pg).tolist() for _ in range(4)] for n in range(3,7)}
    write(OUT/'data.json',dict(pools=pools,layouts=layouts,long_seed=2026100721,layout_seed=2026100722))


def profile():
    OUT.mkdir(exist_ok=True);assert not (OUT/'profile.json').exists();cases=[]
    for arm in ARMS:
        for n in range(3,7):
            torch.manual_seed(23);model=make_model('rope');opt=torch.optim.Adam(model.parameters(),lr=.001)
            x=torch.randint(3,11,(840//(n+1),n));prefix,y=inputs(x)
            pos=positions(n,torch.Generator().manual_seed(23));a,b=SITES[arm];times=[]
            for _ in range(3):
                tick=time.perf_counter();opt.zero_grad(set_to_none=True)
                z,_=forward(model,x,prefix,pos if a else None,pos if b else None)
                F.cross_entropy(z.flatten(0,1),y.flatten()).backward();opt.step();times.append(time.perf_counter()-tick)
            cases.append(dict(arm=arm,length=n,batch=len(x),step_seconds=times,optimizer_bytes=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))))
    write(OUT/'profile.json',dict(cases=cases,discarded_steps=48,device='cpu',dtype='float64',threads=1,
        peak_allocated_bytes=None,peak_reserved_bytes=None,cpu_peak_unmeasured=True))


def freeze():
    assert not (OUT/'lock.json').exists();assert json.loads((OUT/'preflight.json').read_text())['passed']
    oldlock=json.loads((ROOT/'results/copy_position_exposure/lock.json').read_text())
    paths=list(oldlock['protected'])+['experiments/copy_position_sites.py','experiments/audit_copy_position_sites.py',
        'tests/test_copy_position_sites.py','docs/protocols/copy-position-sites.md',
        'results/copy_position_sites/data.json','results/copy_position_sites/profile.json','results/copy_position_sites/preflight.json',
        'results/copy_position_exposure/report.json','results/copy_position_exposure/audit.json']
    for p,sha in oldlock['protected'].items():assert digest(ROOT/p)==sha
    write(OUT/'lock.json',dict(dev=DEV,confirm=list(CONFIRM),stages=list(STAGES),protected={p:digest(ROOT/p) for p in paths}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('profile','data','freeze','run'));args=parser.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    {'profile':profile,'data':prepare_data,'freeze':freeze,'run':experiment}[args.action]()
