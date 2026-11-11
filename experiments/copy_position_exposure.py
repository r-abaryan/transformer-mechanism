"""Matched positional-exposure learning, gated development then confirmation."""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from copy_marker_model import make_model
from copy_rope_model import attention, rotate, matrix_rotation
from encoder_decoder_copy import inputs, generate
from copy_length_exposure import sample, metrics as ordinary_metrics
from audit_kv_cache_equivalence import norm
from mqa_training import hash_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/copy_position_exposure'
ARMS = ('contiguous', 'spaced')
DEV = 3711
CONFIRM = (3719, 3727, 3733)
STAGES = (6000, 12000)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def positions(n, generator):
    return torch.randperm(11, generator=generator)[:n+1].sort().values


def forward(model, content, prefix, pos=None, reference=False):
    base = model.base
    n = content.shape[1]
    ep = torch.arange(n+1) if pos is None else pos
    assert len(ep) == n+1 and bool((ep[1:] > ep[:-1]).all())
    dp = torch.arange(prefix.shape[1]) if pos is None else torch.cat((pos, pos[-1]+torch.arange(1, max(0,prefix.shape[1]-len(pos))+1)))[:prefix.shape[1]]
    source = torch.cat((content, torch.full((len(content),1),2,dtype=torch.long)),1)
    s, t = base.source(source)*math.sqrt(8), base.target(prefix)*math.sqrt(8)
    enc, dec = base.encoder, base.decoder
    sp, tp = torch.zeros_like(source,dtype=torch.bool), torch.zeros_like(prefix,dtype=torch.bool)
    ln = norm if reference else lambda x,l:l(x)
    def ff(l,x):
        return (x@l.linear1.weight.T+l.linear1.bias).clamp_min(0)@l.linear2.weight.T+l.linear2.bias if reference else l.linear2(F.relu(l.linear1(x)))
    memory=ln(s+attention(enc.self_attn,s,s,s,sp,ep,ep,reference=reference),enc.norm1)
    memory=ln(memory+ff(enc,memory),enc.norm2)
    x=ln(t+attention(dec.self_attn,t,t,t,tp,dp,dp,True,reference),dec.norm1)
    layer=dec.multihead_attn;h=layer.num_heads;d=layer.embed_dim//h
    projections=[]
    for a,w,b in zip((x,memory,memory),layer.in_proj_weight.chunk(3),layer.in_proj_bias.chunk(3)):
        y=a@w.T+b if reference else F.linear(a,w,b)
        projections.append(y.reshape(len(content),a.shape[1],h,d).transpose(1,2))
    rotation=matrix_rotation if reference else rotate
    q,k,v=projections;q,k=rotation(q,dp),rotation(k,ep)
    scores=q@k.transpose(-1,-2)/math.sqrt(d)
    if reference:
        ex=(scores-scores.max(-1,keepdim=True).values).exp();weights=ex/ex.sum(-1,keepdim=True)
    else:weights=scores.softmax(-1)
    mixed=(weights@v).transpose(1,2).reshape_as(x)
    cross=mixed@layer.out_proj.weight.T+layer.out_proj.bias
    x=ln(x+cross,dec.norm2);x=ln(x+ff(dec,x),dec.norm3)
    return base.output(x),scores[:,:,0]


def measures(rows,z,scores,generated):
    met=ordinary_metrics(z,rows,generated)
    gaps=scores[:,:,0]-scores[:,:,1:].max(-1).values
    correct=z[:,0].argmax(-1)==torch.tensor([r[0] for r in rows])
    for group in ('all','distinct','repeated'):
        ids=[i for i,r in enumerate(rows) if group=='all' or (len(set(r))==len(r))==(group=='distinct')]
        if not ids:
            met[group]=None
            continue
        met[group].update(first_accuracy=float(correct[ids].double().mean()),
             outranked_fraction=float((gaps[ids]<-1e-10).any(-1).double().mean()),
             near_tie_fraction=float((gaps[ids].abs()<=1e-10).any(-1).double().mean()))
    return met


def evaluate(model,rows,pos=None,reference=False):
    zs=[];ss=[];outputs=[]
    with torch.no_grad():
        for x in torch.tensor(rows).split(64):
            prefix,_=inputs(x);z,s=forward(model,x,prefix,pos,reference);zs.append(z);ss.append(s)
            outputs.extend(generate(lambda a,b:forward(model,a,b,pos,reference)[0][:,-1],x,x.shape[1]+3))
    z,s=torch.cat(zs),torch.cat(ss)
    return measures(rows,z,s,outputs),dict(logits=z,scores=s,generated=outputs)


def specs(data,long=False):
    result={p:dict(rows=rows,positions=None) for p,rows in data['pools'].items() if long or p.startswith(('train_','validation_'))}
    for n in range(3,7):
        for j,pos in enumerate(data['layouts'][str(n)]):
            result[f'layout{j}_{n}']=dict(rows=data['pools'][f'train_{n}'],positions=pos)
    return result


def fit(metrics,arm):
    names=[f'train_{n}' for n in range(3,7)]
    if arm=='spaced':names += [f'layout{j}_{n}' for n in range(3,7) for j in range(4)]
    return all(metrics[p][g]['exact_accuracy']>=.99 for p in names for g in ('distinct','repeated'))


def evaluate_run(model,data,long=False):
    metrics={};saved={}
    for name,spec in specs(data,long).items():
        pos=torch.tensor(spec['positions']) if spec['positions'] is not None else None
        metrics[name],saved[name]=evaluate(model,spec['rows'],pos)
    return metrics,saved


def train_segment(directory,seed,arm,data,stop,start=None):
    torch.manual_seed(seed);model=make_model('rope');initial=hash_state(model.state_dict())
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    gen=torch.Generator().manual_seed(seed+1000);pg=torch.Generator().manual_seed(seed+2000)
    begin=0;stream=hashlib.sha256();posstream=hashlib.sha256()
    previous=''
    if start:
        cp=torch.load(start,weights_only=True);model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer'])
        gen.set_state(cp['generator']);pg.set_state(cp['position_generator']);begin=cp['step'];previous=digest(start)
    pools={p:torch.tensor(rows) for p,rows in data['pools'].items() if p.startswith('train_')}
    curve=[];tick=time.perf_counter()
    for step in range(begin+1,stop+1):
        n,idx,_=sample(pools,gen,'mixed',step)
        pos=positions(n,pg) if arm=='spaced' else None
        stream.update(json.dumps([n,idx.tolist()]).encode());posstream.update(json.dumps(None if pos is None else pos.tolist()).encode())
        x=pools[f'train_{n}'][idx];prefix,labels=inputs(x);opt.zero_grad(set_to_none=True)
        z,_=forward(model,x,prefix,pos);loss=F.cross_entropy(z.flatten(0,1),labels.flatten())
        assert bool(torch.isfinite(loss));loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        opt.step()
        if step%1000==0 or step==stop:
            row=dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-tick);curve.append(row)
            print(json.dumps(dict(seed=seed,arm=arm,**row)),flush=True)
    cp=directory/f'{seed}_{arm}_{stop}.pt'
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=stop,generator=gen.get_state(),position_generator=pg.get_state()),cp)
    metrics,saved=evaluate_run(model.eval(),data)
    scores=directory/f'{seed}_{arm}_{stop}_development_scores.pt';torch.save(saved,scores)
    return dict(seed=seed,arm=arm,step=stop,begin=begin,initial_sha256=initial,previous_sha256=previous,
      checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=digest(cp),scores=str(scores.relative_to(ROOT)),scores_sha256=digest(scores),
      metrics=metrics,fit=fit(metrics,arm),curve=curve,seconds=time.perf_counter()-tick,
      stream_sha256=stream.hexdigest(),position_stream_sha256=posstream.hexdigest())


def experiment(directory=OUT,dev=DEV,confirm=CONFIRM,stages=STAGES):
    directory=Path(directory);assert not (directory/'report.json').exists()
    lock=json.loads((directory/'lock.json').read_text());data=json.loads((directory/'data.json').read_text())
    assert all(digest(ROOT/p)==h for p,h in lock['protected'].items())
    report=dict(status='running',development=[],confirmation=[],selected_steps=None,device='cpu',dtype='float64',threads=1,
                lock_sha256=digest(directory/'lock.json'))
    write(directory/'report.json',report)
    for stop in stages:
        pair=[]
        for arm in ARMS:
            prior=next((r for r in reversed(report['development']) if r['arm']==arm),None)
            pair.append(train_segment(directory,dev,arm,data,stop,ROOT/prior['checkpoint'] if prior else None))
            report['development'].append(pair[-1]);write(directory/'report.json',report)
        assert pair[0]['initial_sha256']==pair[1]['initial_sha256'] and pair[0]['stream_sha256']==pair[1]['stream_sha256']
        if all(r['fit'] for r in pair):report['selected_steps']=stop;break
    write(directory/'development-selection.json',dict(selected_steps=report['selected_steps'],runs=report['development']))
    if report['selected_steps'] is None:
        report['status']='development_fit_failed';write(directory/'report.json',report);return report
    for seed in confirm:
        for arm in ARMS:
            run=train_segment(directory,seed,arm,data,report['selected_steps'])
            report['confirmation'].append(run);write(directory/'report.json',report)
    write(directory/'pretest.json',report)
    for run in report['confirmation']:
        model=make_model('rope');model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        metrics,saved=evaluate_run(model.eval(),data,long=True)
        for name,values in run['metrics'].items():assert metrics[name]==values
        dest=directory/f"{run['seed']}_{run['arm']}_final_scores.pt";torch.save(saved,dest)
        run.update(metrics=metrics,scores=str(dest.relative_to(ROOT)),scores_sha256=digest(dest))
        write(directory/'report.json',report)
    report['status']='complete'
    report['decision']=decision(report['confirmation'])
    assert all(digest(ROOT/p)==h for p,h in lock['protected'].items())
    write(directory/'report.json',report)
    return report


def decision(runs):
    pairs=[]
    for seed in sorted({r['seed'] for r in runs}):
        a,b=[next(r for r in runs if r['seed']==seed and r['arm']==arm) for arm in ARMS]
        groups=[]
        for n in (8,10):
            for g in ('distinct','repeated') if n==8 else ('repeated',):
                x,y=a['metrics'][f'test_{n}'][g],b['metrics'][f'test_{n}'][g]
                groups.append(dict(length=n,group=g,copy_gain=y['exact_accuracy']-x['exact_accuracy'],ranking_gain=x['outranked_fraction']-y['outranked_fraction']))
        short=min(b['metrics'][f'test_{n}'][g]['exact_accuracy']-a['metrics'][f'test_{n}'][g]['exact_accuracy'] for n in range(3,7) for g in ('distinct','repeated'))
        pairs.append(dict(seed=seed,fit=all(fit(r['metrics'],r['arm']) for r in (a,b)),short_worst_change=short,groups=groups))
    return dict(pairs=pairs,supported=bool(pairs) and all(p['fit'] and p['short_worst_change']>=-.05 and all(g['copy_gain']>=.1 and g['ranking_gain']>=.1 for g in p['groups']) for p in pairs))


def prepare_data(directory=OUT):
    directory.mkdir(exist_ok=True)
    old=json.loads((ROOT/'results/copy_marker_training/report.json').read_text())['evaluation_pools']
    pools={p:rows for p,rows in old.items() if not p.endswith(('_8','_10'))}
    seen={tuple(r) for rows in old.values() for r in rows}
    gen=random.Random(2026100711)
    for n in (8,10):
        rows=[]
        for distinct in ((True,False) if n==8 else (False,)):
            wanted=64 if n==8 else 128;part=[]
            for attempt in range(100000):
                row=tuple(gen.sample(range(3,11),n)) if distinct else tuple(gen.choices(range(3,11),k=n))
                if (len(set(row))==n)!=distinct or row in seen:continue
                seen.add(row);part.append(list(row))
                if len(part)==wanted:break
            assert len(part)==wanted;rows+=part
        pools[f'test_{n}']=rows;pools[f'uniform_{n}']=old[f'uniform_{n}']
    pg=torch.Generator().manual_seed(2026100712)
    layouts={str(n):[positions(n,pg).tolist() for _ in range(4)] for n in range(3,7)}
    assert not (directory/'data.json').exists()
    write(directory/'data.json',dict(pools=pools,layouts=layouts,long_seed=2026100711,layout_seed=2026100712,
          short_pools_reused=True,fresh_long_test_only=True))


def profile():
    OUT.mkdir(exist_ok=True);assert not (OUT/'profile.json').exists();cases=[]
    for arm in ARMS:
        for n in range(3,7):
            torch.manual_seed(19);model=make_model('rope');opt=torch.optim.Adam(model.parameters(),lr=.001)
            gen=torch.Generator().manual_seed(19);x=torch.randint(3,11,(840//(n+1),n));prefix,y=inputs(x)
            pos=positions(n,gen) if arm=='spaced' else None;times=[]
            for _ in range(3):
                tick=time.perf_counter();opt.zero_grad(set_to_none=True)
                z,_=forward(model,x,prefix,pos);F.cross_entropy(z.flatten(0,1),y.flatten()).backward();opt.step();times.append(time.perf_counter()-tick)
            cases.append(dict(arm=arm,length=n,batch=len(x),step_seconds=times,optimizer_bytes=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))))
    write(OUT/'profile.json',dict(cases=cases,device='cpu',dtype='float64',threads=1,discarded_steps=24,
          peak_allocated_bytes=None,peak_reserved_bytes=None,cpu_peak_unmeasured=True))


def freeze():
    assert not (OUT/'lock.json').exists()
    assert json.loads((OUT/'preflight.json').read_text())['passed']
    paths=[Path(__file__),ROOT/'experiments/audit_copy_position_exposure.py',ROOT/'tests/test_copy_position_exposure.py',
           ROOT/'docs/protocols/copy-position-exposure.md',OUT/'data.json',OUT/'profile.json',OUT/'preflight.json']
    paths += [ROOT/'experiments'/n for n in ('copy_marker_model.py','copy_rope_model.py','encoder_decoder_contract.py','encoder_decoder_copy.py','copy_length_exposure.py','audit_copy_length_exposure.py','audit_kv_cache_equivalence.py','mqa_training.py')]
    paths += [ROOT/'results/copy_marker_training'/n for n in ('report.json','audit.json')]
    write(OUT/'lock.json',dict(dev=DEV,confirm=list(CONFIRM),stages=list(STAGES),protected={str(p.relative_to(ROOT)):digest(p) for p in paths}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('profile','data','freeze','run'));args=parser.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    {'profile':profile,'data':prepare_data,'freeze':freeze,'run':experiment}[args.action]()
