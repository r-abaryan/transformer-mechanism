"""Direction-only intervention in the installed, hash-bound L-BFGS step."""
import argparse
import copy
import hashlib
import inspect
import json
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.optim.lbfgs as engine
import full_pool_fitting as prior

ROOT=prior.ROOT
OUT=ROOT/'results/fitting_direction'
SOURCE=ROOT/'results/full_pool_fitting_confirmation/study'
SEEDS=(3473,3479,3481)
ARMS=('lbfgs','negative_gradient')
SOURCES=('experiments/fitting_direction.py','experiments/audit_fitting_direction.py','tests/test_fitting_direction.py')
PROTOCOL='docs/protocols/fitting-direction.md'
replay=prior.prior.training.replay


def adapted_step():
    """No public direction hook exists; inject one guarded statement only."""
    original=textwrap.dedent(inspect.getsource(engine.LBFGS.step))
    anchor='        gtd = flat_grad.dot(d)'
    assert original.count(anchor)==1 and torch.__version__.startswith('2.11.0')
    modified=original.replace(anchor,'        if self.direction_rule == "negative_gradient":\n            d = flat_grad.neg()\n'+anchor)
    scope=dict(vars(engine));exec(compile(modified,'<direction-only-lbfgs-step>','exec'),scope)
    return scope['step'],hashlib.sha256(original.encode()).hexdigest(),hashlib.sha256(modified.encode()).hexdigest()


STEP,NATIVE_HASH,ADAPTED_HASH=adapted_step()


class DirectionLBFGS(engine.LBFGS):
    step=STEP

    def __init__(self,parameters,rule):
        assert rule in ARMS
        super().__init__(parameters,lr=1.,max_iter=1,max_eval=26,history_size=20,
                         tolerance_grad=1e-10,tolerance_change=1e-12,line_search_fn='strong_wolfe')
        self.direction_rule=rule


def inputs():
    pools=torch.load(SOURCE/'pools.pt',weights_only=True)
    return [dict(seed=s,checkpoint=(SOURCE/f'warmup/{s}_adam_12000.pt').relative_to(ROOT).as_posix()) for s in SEEDS],pools


def train(case,pools,out,arm,budget,interval,scores):
    model,_,start=prior.load(ROOT/case['checkpoint']);opt=DirectionLBFGS(model.parameters(),arm)
    tensors=prior.tables(pools);calls=[];steps=[];snapshots=[];used=0;began=time.perf_counter()
    prior.snapshot(model,opt,out,case['seed'],arm,0,pools,scores,snapshots)
    while budget-used>=2:
        first=used;before=prior.model_copy(model)
        opt.param_groups[0]['max_eval']=min(26,budget-used-1)
        def closure():
            nonlocal used
            assert used<budget,'Closure budget exceeded'
            opt.zero_grad(set_to_none=True);loss=prior.objective(model,tensors);used+=1
            calls.append(dict(model=prior.model_copy(model),gradients=prior.gradients(model),loss=loss))
            return torch.tensor(loss,dtype=torch.float64)
        opt.step(closure)
        assert 1<=used-first<=opt.param_groups[0]['max_eval']+1
        steps.append(dict(first=first,last=used,before=before,after=prior.model_copy(model),optimizer=copy.deepcopy(opt.state_dict())))
        if used//interval>first//interval:
            item=prior.snapshot(model,opt,out,case['seed'],arm,used,pools,scores,snapshots)
            print(json.dumps(dict(seed=case['seed'],arm=arm,closures=used,train32=item['training']['32']['groups']['all'])),flush=True)
    if snapshots[-1]['step']!=used:prior.snapshot(model,opt,out,case['seed'],arm,used,pools,scores,snapshots)
    trace=out/f"{case['seed']}_{arm}_trace.pt";torch.save(dict(calls=calls,steps=steps),trace)
    h=sum(len(pools[f'train_{m}']) for m in prior.LOADS);n=sum(len(pools[f'train_{m}'])*m for m in prior.LOADS)
    return dict(seed=case['seed'],arm=arm,source_step=start,closure_evaluations=used,optimizer_calls=len(steps),
                histories=used*h,input_events=used*h*32,answer_targets=used*n,snapshots=snapshots,
                final_checkpoint=snapshots[-1]['checkpoint'],trace=trace.relative_to(ROOT).as_posix(),trace_sha256=replay.digest(trace),
                seconds=time.perf_counter()-began,stop_reason='closure_budget_remaining_below_two')


def paired_decision(runs):
    pairs=[]
    for seed in sorted({r['seed'] for r in runs}):
        a,b=[next(r for r in runs if (r['seed'],r['arm'])==(seed,arm)) for arm in ARMS]
        def stats(run):
            metrics=run['snapshots'][-1]['training']
            return dict(objective=sum(metrics[str(m)]['groups']['all']['ce'] for m in prior.LOADS)/4,
                        errors=sum(round((1-metrics[str(m)]['groups']['all']['accuracy'])*metrics[str(m)]['groups']['all']['count']) for m in prior.LOADS))
        x,y=stats(a),stats(b)
        pairs.append(dict(seed=seed,lbfgs=x,negative_gradient=y,
                          lower_objective=x['objective']<y['objective']-1e-8,no_more_errors=x['errors']<=y['errors']))
    return dict(pairs=pairs,all_seed_direction_benefit=all(p['lower_objective'] and p['no_more_errors'] for p in pairs))


def worker(args):
    case,pools,out,budget,interval=args
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    scores=[];runs=[train(case,pools,out,arm,budget,interval,scores) for arm in ARMS]
    return runs,scores


def execute(out,cases,pools,budget=128,interval=32,fixture=False):
    assert not out.exists(),'Preserve prior evidence'
    files={p:replay.digest(ROOT/p) for p in prior.dependencies()|set(SOURCES)}
    for c in cases:files[c['checkpoint']]=replay.digest(ROOT/c['checkpoint'])
    if not fixture:
        assert [c['seed'] for c in cases]==list(SEEDS) and (budget,interval)==(128,32)
        for p in (PROTOCOL,'results/fitting_direction_preflight.json','results/fitting_direction_profiles.json',
                  'results/full_pool_fitting_confirmation/study/report.json','results/full_pool_fitting_confirmation/study/audit.json',
                  'results/full_pool_fitting_confirmation/study/pools.pt'):
            files[p]=replay.digest(ROOT/p)
        pre=json.loads((ROOT/'results/fitting_direction_preflight.json').read_text());assert pre['passed']
        assert pre['sources']=={p:files[p] for p in SOURCES}
        profile=json.loads((ROOT/'results/fitting_direction_profiles.json').read_text())
        assert profile['sources']==pre['sources']
    out.mkdir(parents=True);torch.save(pools,out/'pools.pt')
    replay.write(out/'lock.json',dict(files=files,cases=cases,budget=budget,interval=interval,fixture=fixture,
                 pools_sha256=replay.digest(out/'pools.pt'),torch=torch.__version__,native_hash=NATIVE_HASH,adapted_hash=ADAPTED_HASH))
    started=time.perf_counter();runs=[];scores=[]
    jobs=[(c,pools,out,budget,interval) for c in cases]
    if fixture:results=map(worker,jobs)
    else:
        pool=ProcessPoolExecutor(max_workers=3);results=pool.map(worker,jobs)
    try:
        for r,s in results:runs.extend(r);scores.extend(s)
    finally:
        if not fixture:pool.shutdown()
    decision=paired_decision(runs);replay.write(out/'pretest_decision.json',decision)
    for run in runs:
        with torch.random.fork_rng(devices=[]):model=prior.prior.training.previous.load_model(ROOT/run['final_checkpoint'])
        for split in ('validation','test'):
            for m in prior.LOADS:
                cell=prior.prior.training.evaluate(model,pools[f'{split}_{m}'])
                cell.update(seed=run['seed'],arm=run['arm'],step=run['snapshots'][-1]['step'],pool=f'{split}_{m}',checkpoint=run['final_checkpoint']);scores.append(cell)
    torch.save(scores,out/'scores.pt');replay.check_files(files)
    report=dict(status='complete',runs=runs,decision=decision,cells=[{k:v for k,v in c.items() if k not in ('logits','labels','repeat')} for c in scores],
                evaluation_targets=sum(c['labels'].numel() for c in scores),gradient_evaluations=sum(r['closure_evaluations'] for r in runs),
                optimizer_calls=sum(r['optimizer_calls'] for r in runs),answer_targets=sum(r['answer_targets'] for r in runs),
                histories=sum(r['histories'] for r in runs),input_events=sum(r['input_events'] for r in runs),seconds=time.perf_counter()-started,
                device='cpu',dtype='float64',threads_per_worker=1,workers=1 if fixture else 3,test_is_exploratory=True,
                selection='none',scores_sha256=replay.digest(out/'scores.pt'),lock_sha256=replay.digest(out/'lock.json'),
                pretest_sha256=replay.digest(out/'pretest_decision.json'))
    replay.write(out/'report.json',report);print(json.dumps(dict(status='complete',decision=decision)),flush=True)


def profile():
    cases,pools=inputs();rows=[]
    for arm in ARMS:
        model,_,_=prior.load(ROOT/cases[0]['checkpoint']);opt=DirectionLBFGS(model.parameters(),arm);tables=prior.tables(pools)
        for index in range(2):
            count=0;began=time.perf_counter()
            def closure():
                nonlocal count
                opt.zero_grad(set_to_none=True);loss=prior.objective(model,tables);count+=1
                return torch.tensor(loss,dtype=torch.float64)
            opt.step(closure)
            rows.append(dict(arm=arm,step=index+1,seconds=time.perf_counter()-began,gradient_evaluations=count,
                             histories=count*3072,input_events=count*3072*32,answer_targets=count*44544,
                             optimizer_state_present=bool(opt.state),peak_allocated_bytes=None,peak_reserved_bytes=None))
    return dict(records=rows,device='cpu',dtype='float64',threads=1,sources={p:replay.digest(ROOT/p) for p in SOURCES})


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser();parser.add_argument('--profile',action='store_true');args=parser.parse_args()
    if args.profile:
        path=ROOT/'results/fitting_direction_profiles.json';assert not path.exists();replay.write(path,profile())
    else:execute(OUT,*inputs())
