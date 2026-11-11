"""Training-only optimization development; historical evaluation pools stay unused."""
import copy
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from state_tracking import ROOT, STRATA, model_new, forward, tensors, save_json, provenance
from audit_state_tracking import independent, oracle
from mqa_training import hash_state
from position_distractors import digest

DIRECTORY = ROOT/'results/state_recipe_development'
SEEDS = (2903, 2909)
STEPS = 6000
BATCH = 64
RECIPES = [('constant_001', .001, False), ('constant_003', .003, False),
           ('cosine_001', .001, True), ('cosine_003', .003, True)]
PROTOCOL = ROOT/'docs/protocols/state-recipe-development.md'


def rate(step, peak, scheduled):
    if not scheduled:
        return peak
    if step <= 200:
        return peak * step/200
    phase = (step-200)/(STEPS-200)
    return peak*(.1+.9*(1+math.cos(math.pi*phase))/2)


def metric(scores, labels):
    predicted=scores.argmax(-1)
    losses=scores.logsumexp(-1)-scores.gather(1,labels[:,None]).squeeze(1)
    return dict(count=len(labels),accuracy=float((predicted==labels).double().mean()),loss=float(losses.mean()))


def evaluate(model, pools, fn=forward):
    results={};scores={}
    with torch.no_grad():
        for key,rows in pools.items():
            x,y=tensors(rows)
            scores[key]=torch.cat([fn(model,b)[:,-1] for b in x.split(128)])
            results[key]=metric(scores[key],y)
    return results,scores


def training_pools():
    path=ROOT/'results/state_tracking_data.json'
    original=json.loads((ROOT/'results/state_tracking.json').read_text())
    assert digest(path)==original['input_sha256']
    full=json.loads(path.read_text())
    return {k:v for k,v in full.items() if k.startswith('train/')}


def preflight(pools):
    label_count=negative_count=0;records=[]
    for key,rows in pools.items():
        assert key.startswith('train/')
        for ids,y in rows:
            assert oracle(ids)==y
            # Wrong label control ignores every FLIP.
            wrong=oracle([ids[0]]+[0 if v==1 else v for v in ids[1:-1]]+[ids[-1]])
            negative_count+=wrong!=y;label_count+=1
        assert sum(y for _,y in rows)==len(rows)//2
    assert negative_count>0
    for seed in SEEDS:
        torch.manual_seed(seed);m=model_new()
        for n in [8,12]:
            rows=[row for g in STRATA for row in pools[f'train/{n}/{g}'][:4]]
            x,y=tensors(rows)
            a,b=forward(m,x)[:,-1],independent(m,x)[:,-1]
            ga=torch.autograd.grad(F.cross_entropy(a,y),list(m.parameters()),retain_graph=True)
            gb=torch.autograd.grad(F.cross_entropy(b,y),list(m.parameters()))
            bad=torch.autograd.grad(F.cross_entropy(a,1-y),list(m.parameters()))
            oe=float((a-b).abs().max().detach());ge=max(float((u-v).abs().max()) for u,v in zip(ga,gb))
            ne=max(float((u-v).abs().max()) for u,v in zip(ga,bad))
            assert max(oe,ge)<=1e-10 and ne>1e-6
            records.append(dict(seed=seed,length=n,output_error=oe,gradient_error=ge,shifted_label_gradient_effect=ne))
    # Independent boundaries plus an intentionally missing warmup.
    assert rate(1,.001,True)==.001/200 and rate(200,.001,True)==.001
    assert abs(rate(STEPS,.001,True)-.0001)<1e-15
    assert abs(rate(1,.001,False)-rate(1,.001,True))>1e-6
    return dict(labels=label_count,ignored_flip_errors=negative_count,cases=records,schedule_boundaries_pass=True)


def profile():
    torch.set_num_threads(1);torch.manual_seed(2897)
    assert not DIRECTORY.exists(),'Use a distinct explicit run directory'
    # Validate inputs first; a missing dependency must not leave a partial data artifact.
    prov=provenance();pools=training_pools()
    x,y=tensors(pools['train/12/reset_free'][:BATCH])
    m=model_new();opt=torch.optim.Adam(m.parameters(),lr=.001)
    start=time.perf_counter();loss=F.cross_entropy(forward(m,x)[:,-1],y)
    loss.backward();opt.step()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters())
    report=dict(python=platform.python_version(),torch=torch.__version__,device='cpu',dtype='float64',
                batch=BATCH,operation_length=12,sequence_tokens=14,parameters=sum(p.numel() for p in m.parameters()),
                optimizer_entries=len(opt.state),optimizer_tensor_bytes=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v)),
                full_step_seconds=time.perf_counter()-start,peak_gpu_allocated_bytes=0,peak_gpu_reserved_bytes=0,cpu_peak_memory_measured=False,
                source_sha256=digest(Path(__file__)),input_sha256=digest(ROOT/'results/state_tracking_data.json'),dependencies=prov)
    DIRECTORY.mkdir();save_json(DIRECTORY/'profile.json',report);print(json.dumps(report),flush=True)


def choose(runs):
    candidates=[]
    for name,_,_ in RECIPES:
        rs=[r for r in runs if r['recipe']==name]
        passed=len(rs)==2 and all(r['status']=='trained' and min(v['accuracy'] for v in r['metrics'].values())>=.99 for r in rs)
        candidates.append(dict(recipe=name,passes_fit=passed,
                               worst_accuracy=min((v['accuracy'] for r in rs for v in r['metrics'].values()),default=0),
                               worst_loss=max((v['loss'] for r in rs for v in r['metrics'].values()),default=None)))
    eligible=[r for r in candidates if r['passes_fit']]
    selected=min(eligible,key=lambda v:(v['worst_loss'],[x[0] for x in RECIPES].index(v['recipe'])))['recipe'] if eligible else None
    return dict(candidates=candidates,selected_recipe=selected,rule='final full-training >=99% every group/length/seed; lowest worst-cell CE; fixed order breaks ties')


def train():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert PROTOCOL.is_file() and (DIRECTORY/'profile.json').is_file()
    assert not (DIRECTORY/'report.json').exists()
    assert not (DIRECTORY/'training_data.json').exists() and not (DIRECTORY/'checkpoints').exists()
    prof=json.loads((DIRECTORY/'profile.json').read_text());assert digest(Path(__file__))==prof['source_sha256']
    pools=training_pools();checks=preflight(pools)
    sources={str(PROTOCOL.relative_to(ROOT)):digest(PROTOCOL),'experiments/state_recipe_development.py':digest(Path(__file__)),
             'results/state_tracking_data.json':digest(ROOT/'results/state_tracking_data.json'),**provenance()}
    old=json.loads((ROOT/'results/state_tracking.json').read_text())
    preserved={p:digest(ROOT/p) for p in ['results/state_tracking.json','results/state_tracking_fit.json','results/state_tracking_audit.json',
                                         'results/state_tracking_scores.pt','results/state_tracking_fit_scores.pt']+[v['checkpoint'] for v in old['runs']]}
    save_json(DIRECTORY/'training_data.json',pools)
    report=dict(phase='development',device='cpu',dtype='float64',torch=torch.__version__,python=platform.python_version(),
                protocol_sha256=digest(PROTOCOL),source_dependencies=sources,preserved=preserved,preflight=checks,
                training_snapshot_sha256=digest(DIRECTORY/'training_data.json'),runs=[],uses_validation_or_test_for_selection=False)
    path=DIRECTORY/'report.json';save_json(path,report);(DIRECTORY/'checkpoints').mkdir()
    encoded={n:tensors([row for g in STRATA for row in pools[f'train/{n}/{g}']]) for n in (8,12)}
    for seed_index,seed in enumerate(SEEDS):
        torch.manual_seed(seed);base=model_new();init=hash_state(base.state_dict())
        order=RECIPES[seed_index:]+RECIPES[:seed_index]
        for name,peak,schedule in order:
            m=copy.deepcopy(base);opt=torch.optim.Adam(m.parameters(),lr=peak)
            gen=torch.Generator().manual_seed(seed+1000);stream=hashlib.sha256();tick=time.perf_counter()
            run=dict(seed=seed,recipe=name,peak_lr=peak,scheduled=schedule,initial_sha256=init,status='running',step=0,curve=[],metrics={})
            report['runs'].append(run);save_json(path,report)
            try:
                for step in range(1,STEPS+1):
                    n=8 if step%2 else 12;x,y=encoded[n];ix=torch.randint(len(y),(BATCH,),generator=gen)
                    stream.update(json.dumps(ix.tolist()).encode());lr=rate(step,peak,schedule)
                    for group in opt.param_groups:group['lr']=lr
                    opt.zero_grad();loss=F.cross_entropy(forward(m,x[ix])[:,-1],y[ix])
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
                    loss.backward()
                    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters()):raise FloatingPointError('Nonfinite gradient')
                    opt.step();run['step']=step
                    if not all(bool(torch.isfinite(p).all()) for p in m.parameters()):raise FloatingPointError('Nonfinite parameter')
                    if step%1000==0:
                        metric,_=evaluate(m,pools);run['curve'].append(dict(step=step,lr=lr,minibatch_loss=float(loss.detach()),training=metric))
                        save_json(path,report)
                        print(json.dumps(dict(seed=seed,recipe=name,step=step,worst_accuracy=min(v['accuracy'] for v in metric.values()),worst_loss=max(v['loss'] for v in metric.values()))),flush=True)
                run['status']='trained'
            except Exception as exc:
                run.update(status='failed',failure=repr(exc))
            cp=DIRECTORY/'checkpoints'/f'{seed}_{name}.pt'
            torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),generator=gen.get_state(),steps=run['step']),cp)
            run.update(checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=digest(cp),stream_sha256=stream.hexdigest(),seconds=time.perf_counter()-tick,
                       examples=run['step']*BATCH,supervised_targets=run['step']*BATCH,
                       input_tokens=sum(BATCH*(10 if s%2 else 14) for s in range(1,run['step']+1)))
            if run['status']=='trained':
                run['metrics'],scores=evaluate(m,pools)
                sp=DIRECTORY/'checkpoints'/f'{seed}_{name}_scores.pt';torch.save(scores,sp)
                run.update(scores=str(sp.relative_to(ROOT)),scores_sha256=digest(sp))
            save_json(path,report)
    for seed in SEEDS:
        rs=[r for r in report['runs'] if r['seed']==seed]
        assert len(set(r['initial_sha256'] for r in rs))==1
        done=[r for r in rs if r['status']=='trained']
        if done:assert len(set(r['stream_sha256'] for r in done))==1
    report.update(phase='development_finished',decision=choose(report['runs']))
    assert all(digest(ROOT/p)==h for p,h in preserved.items());save_json(path,report)
    print(json.dumps(report['decision'],indent=2),flush=True)


def audit():
    torch.set_num_threads(1)
    path=DIRECTORY/'report.json';r=json.loads(path.read_text());pools=json.loads((DIRECTORY/'training_data.json').read_text())
    assert all(digest(ROOT/p)==h for p,h in r['source_dependencies'].items())
    assert all(digest(ROOT/p)==h for p,h in r['preserved'].items())
    assert digest(DIRECTORY/'training_data.json')==r['training_snapshot_sha256']
    cells=[];rows=[]
    for seed in SEEDS:
        gen=torch.Generator().manual_seed(seed+1000);stream=hashlib.sha256()
        for step in range(1,STEPS+1):
            n=8 if step%2 else 12;count=sum(len(pools[f'train/{n}/{g}']) for g in STRATA)
            ids=torch.randint(count,(BATCH,),generator=gen);stream.update(json.dumps(ids.tolist()).encode())
        for run in [v for v in r['runs'] if v['seed']==seed]:
            if run['status']!='trained':continue
            assert run['stream_sha256']==stream.hexdigest()
            cp=ROOT/run['checkpoint'];assert digest(cp)==run['checkpoint_sha256']
            saved=torch.load(cp,weights_only=True)
            assert saved['steps']==STEPS and torch.equal(saved['generator'],gen.get_state())
            assert all(int(v['step'])==STEPS for v in saved['optimizer']['state'].values())
            peak=run['peak_lr'];scheduled=run['scheduled']
            assert abs(saved['optimizer']['param_groups'][0]['lr']-rate(STEPS,peak,scheduled))<1e-15
            torch.manual_seed(seed);m=model_new();assert hash_state(m.state_dict())==run['initial_sha256']
            m.load_state_dict(saved['model']);metrics,scores=evaluate(m,pools,independent)
            assert digest(ROOT/run['scores'])==run['scores_sha256'];native=torch.load(ROOT/run['scores'],weights_only=True)
            for key,rows_pool in pools.items():
                y=[oracle(ids) for ids,_ in rows_pool]
                assert y==[v for _,v in rows_pool]
                z=scores[key];oe=float((z-native[key]).abs().max());changes=int((z.argmax(-1)!=native[key].argmax(-1)).sum())
                # Independent Python loss and accuracy reduction.
                correct=0;losses=[]
                for row,label in zip(z.tolist(),y):
                    correct+=max(range(2),key=lambda i:row[i])==label
                    mx=max(row);losses.append(mx+math.log(sum(math.exp(v-mx) for v in row))-row[label])
                me=max(abs(correct/len(y)-run['metrics'][key]['accuracy']),abs(sum(losses)/len(y)-run['metrics'][key]['loss']))
                assert max(oe,me)<=1e-10 and changes==0
                cells.append(dict(seed=seed,recipe=run['recipe'],pool=key,count=len(y),logit_error=oe,scalar_metric_error=me,argmax_changes=changes))
                rows.append(dict(seed=seed,recipe=run['recipe'],pool=key,**run['metrics'][key]))
    assert choose(r['runs'])==r['decision']
    out=dict(passed=True,evaluations=len(cells),predictions=sum(c['count'] for c in cells),max_logit_error=max(c['logit_error'] for c in cells),
             max_scalar_metric_error=max(c['scalar_metric_error'] for c in cells),cells=cells,report_sha256=digest(path),
             complete_attempts=sum(v['status']=='trained' for v in r['runs']),attempts=len(r['runs']))
    target=DIRECTORY/'audit.json';assert not target.exists();save_json(target,out)
    with (DIRECTORY/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in out.items() if k!='cells'}),flush=True)


if __name__=='__main__':
    {'profile':profile,'train':train,'audit':audit}[sys.argv[1]]()
