"""Bounded full-objective L-BFGS versus exposure-matched original Adam."""
import argparse
import copy
import inspect
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
import torch.optim.lbfgs as lbfgs_source
import query_feedback_continuation as continuation
import readout_norm_cap_training as prior

ROOT=prior.ROOT
OUT=ROOT/'results/full_pool_fitting'
PROTOCOL='docs/protocols/full-pool-fitting.md'
SOURCES=('experiments/full_pool_fitting.py','experiments/audit_full_pool_fitting.py','tests/test_full_pool_fitting.py')
LOADS=(2,8,16,32)
SEEDS=(3461,3463)
CLOSURES=128
RATIO=48
INTERVAL=32


def load(path):
    with torch.random.fork_rng(devices=[]):model,opt,start=continuation.restore(path)
    model.source_step=start
    return model,opt,start


def tables(pools):
    return {m:prior.training.replay.pressure.prepare(pools[f'train_{m}'],'all_queries') for m in LOADS}


def objective(model,tensors):
    """Accumulate exact equal-load gradients; each load graph is released."""
    loss=0.
    for m in LOADS:
        e,q,y,_=tensors[m]
        part=F.cross_entropy(prior.training.forward_queries(model,e,q).flatten(0,1),y.flatten())/4
        assert torch.isfinite(part)
        part.backward();loss+=float(part.detach())
    return loss


def make_lbfgs(model):
    return torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=1,max_eval=26,history_size=20,
                            tolerance_grad=1e-10,tolerance_change=1e-12,line_search_fn='strong_wolfe')


def fit(metrics):return continuation.training_fit(metrics)


def model_copy(model):return copy.deepcopy(model.state_dict())


def gradients(model):return {n:p.grad.detach().clone() for n,p in model.named_parameters()}


def snapshot(model,opt,folder,seed,arm,step,pools,scores,records):
    path=folder/f'{seed}_{arm}_{step}.pt';assert not path.exists()
    counter=int(next(iter(opt.state.values()),{}).get('step',0)) if isinstance(opt,torch.optim.Adam) else next(iter(opt.state.values()),{}).get('n_iter',0)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=counter,procedure_step=step,source_step=model.source_step),path)
    training={}
    for m in LOADS:
        cell=prior.training.evaluate(model,pools[f'train_{m}'])
        cell.update(seed=seed,arm=arm,step=step,pool=f'train_{m}',checkpoint=path.relative_to(ROOT).as_posix())
        scores.append(cell);training[str(m)]=cell['metrics']
    item=dict(step=step,checkpoint=path.relative_to(ROOT).as_posix(),sha256=prior.training.replay.digest(path),
              training=training,fit=fit(training),perfect=all(v['groups']['all']['accuracy']==1. for v in training.values()),
              beta=float(model.write_logit.detach().sigmoid()))
    records.append(item)
    return item


def train_lbfgs(case,pools,folder,scores,budget,interval):
    model,_,start=load(ROOT/case['checkpoint']);opt=make_lbfgs(model);tensor=tables(pools)
    calls=[];steps=[];snapshots=[];used=0;began=time.perf_counter()
    snapshot(model,opt,folder,case['seed'],'lbfgs',0,pools,scores,snapshots)
    while budget-used>=2:
        first=len(calls);before=model_copy(model)
        # PyTorch 2.11 permits max_eval+1 calls at the line-search limit.
        opt.param_groups[0]['max_eval']=min(26,budget-used-1)
        def closure():
            nonlocal used
            assert used<budget,'Closure budget exceeded'
            opt.zero_grad(set_to_none=True);loss=objective(model,tensor);used+=1
            calls.append(dict(model=model_copy(model),gradients=gradients(model),loss=loss))
            return torch.tensor(loss,dtype=torch.float64)
        opt.step(closure)
        assert 1<=len(calls)-first<=opt.param_groups[0]['max_eval']+1
        steps.append(dict(first=first,last=len(calls),before=before,after=model_copy(model),optimizer=copy.deepcopy(opt.state_dict())))
        crossed=used//interval>first//interval
        if crossed:
            item=snapshot(model,opt,folder,case['seed'],'lbfgs',used,pools,scores,snapshots)
            print(json.dumps(dict(seed=case['seed'],arm='lbfgs',closures=used,fit=item['fit'],train32=item['training']['32']['groups']['all'])),flush=True)
    if snapshots[-1]['step']!=used:snapshot(model,opt,folder,case['seed'],'lbfgs',used,pools,scores,snapshots)
    path=folder/f"{case['seed']}_lbfgs_trace.pt";torch.save(dict(calls=calls,steps=steps),path)
    n=sum(t[2].numel() for t in tensor.values());h=sum(len(t[0]) for t in tensor.values())
    return dict(seed=case['seed'],arm='lbfgs',source_step=start,closure_evaluations=used,optimizer_calls=len(steps),
                histories=used*h,input_events=used*h*32,answer_targets=used*n,snapshots=snapshots,
                final_checkpoint=snapshots[-1]['checkpoint'],trace=path.relative_to(ROOT).as_posix(),trace_sha256=prior.training.replay.digest(path),
                seconds=time.perf_counter()-began,stop_reason='closure_budget_remaining_below_two')


def train_adam(case,pools,folder,scores,closures,ratio,interval):
    model,opt,start=load(ROOT/case['checkpoint']);tensor=tables(pools)
    hr,qr,_,tab,_=continuation.streams(case['seed'],dict(pools=pools),start)
    steps=[];snapshots=[];targets=0;began=time.perf_counter();updates=closures*ratio
    snapshot(model,opt,folder,case['seed'],'adam',0,pools,scores,snapshots)
    for index in range(updates):
        step=start+index+1;m=LOADS[(step-1)%4]
        ids=torch.randint(len(tensor[m][0]),(64,),generator=hr);draws=torch.randint(m,(64,m),generator=qr)
        q,y,_=prior.training.choose(*tab[m],ids,draws,'exhaustive');e=tensor[m][0][ids]
        opt.zero_grad(set_to_none=True);loss=F.cross_entropy(prior.training.forward_queries(model,e,q).flatten(0,1),y.flatten())
        assert torch.isfinite(loss);loss.backward();g=gradients(model);opt.step();targets+=y.numel()
        steps.append(dict(indices=ids,draws=draws,load=m,loss=float(loss.detach()),gradients=g,after=model_copy(model)))
        if (index+1)%(interval*ratio)==0:
            item=snapshot(model,opt,folder,case['seed'],'adam',index+1,pools,scores,snapshots)
            print(json.dumps(dict(seed=case['seed'],arm='adam',updates=index+1,fit=item['fit'],train32=item['training']['32']['groups']['all'])),flush=True)
    if snapshots[-1]['step']!=updates:snapshot(model,opt,folder,case['seed'],'adam',updates,pools,scores,snapshots)
    path=folder/f"{case['seed']}_adam_trace.pt";torch.save(steps,path)
    return dict(seed=case['seed'],arm='adam',source_step=start,closure_evaluations=0,optimizer_calls=updates,histories=updates*64,
                input_events=updates*64*32,answer_targets=targets,snapshots=snapshots,final_checkpoint=snapshots[-1]['checkpoint'],
                trace=path.relative_to(ROOT).as_posix(),trace_sha256=prior.training.replay.digest(path),seconds=time.perf_counter()-began,
                stop_reason='matched_lbfgs_gradient_target_exposure')


def decision(runs):
    return {arm:dict(all_seed_fit=all(r['snapshots'][-1]['fit'] for r in runs if r['arm']==arm),
                     all_seed_perfect=all(r['snapshots'][-1]['perfect'] for r in runs if r['arm']==arm)) for arm in ('lbfgs','adam')}


def inputs():
    _,pools=prior.inputs()
    cases=[dict(seed=s,checkpoint=f'results/query_feedback_continuation/{s}_exhaustive_12000.pt') for s in SEEDS]
    return cases,pools


def dependencies():
    paths=set(SOURCES)|{'experiments/audit_query_feedback_training.py','experiments/audit_objective_beta_pressure.py','experiments/audit_shared_state_feasibility.py'}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.parent==ROOT/'experiments' and p.suffix=='.py':paths.add(p.relative_to(ROOT).as_posix())
    return paths


def execute(out,cases,pools,budget=CLOSURES,ratio=RATIO,interval=INTERVAL,fixture=False):
    assert not out.exists(),'Preserve previous evidence'
    assert budget>=2 and ratio>0 and interval>0
    files={p:prior.training.replay.digest(ROOT/p) for p in dependencies()}
    for case in cases:files[case['checkpoint']]=prior.training.replay.digest(ROOT/case['checkpoint'])
    if not fixture:
        assert budget==CLOSURES and ratio==RATIO and interval==INTERVAL and [c['seed'] for c in cases]==list(SEEDS)
        for p in (PROTOCOL,'results/full_pool_fitting_preflight.json','results/full_pool_fitting_profiles.json',
                  'results/delta_query_coverage/development_data.json','results/query_feedback_training/test_data.json',
                  'results/query_feedback_continuation/report.json','results/query_feedback_continuation/audit.json'):
            files[p]=prior.training.replay.digest(ROOT/p)
        pre=json.loads((ROOT/'results/full_pool_fitting_preflight.json').read_text());assert pre['passed']
        assert pre['sources']=={p:files[p] for p in SOURCES}
    prior.training.replay.check_files(files);out.mkdir(parents=True)
    torch.save(pools,out/'pools.pt')
    prior.training.replay.write(out/'lock.json',dict(files=files,cases=cases,budget=budget,ratio=ratio,interval=interval,fixture=fixture,
        pools_sha256=prior.training.replay.digest(out/'pools.pt'),torch=torch.__version__,
        lbfgs_implementation_sha256=prior.training.replay.digest(Path(inspect.getsourcefile(lbfgs_source)))))
    runs=[];scores=[]
    for case in cases:
        a=train_lbfgs(case,pools,out,scores,budget,interval);runs.append(a)
        b=train_adam(case,pools,out,scores,a['closure_evaluations'],ratio,interval);runs.append(b)
        if not fixture:
            assert a['answer_targets']==b['answer_targets'] and a['histories']==b['histories']
    choice=decision(runs);prior.training.replay.write(out/'pretest_decision.json',choice)
    for run in runs:
        with torch.random.fork_rng(devices=[]):model=prior.training.previous.load_model(ROOT/run['final_checkpoint'])
        for split in ('validation','test'):
            for m in LOADS:
                cell=prior.training.evaluate(model,pools[f'{split}_{m}'])
                cell.update(seed=run['seed'],arm=run['arm'],step=run['snapshots'][-1]['step'],pool=f'{split}_{m}',checkpoint=run['final_checkpoint']);scores.append(cell)
    torch.save(scores,out/'scores.pt')
    report=dict(status='complete',runs=runs,decision=choice,cells=[{k:v for k,v in c.items() if k not in ('logits','labels','repeat')} for c in scores],
        evaluation_targets=sum(c['labels'].numel() for c in scores),optimizer_calls=sum(r['optimizer_calls'] for r in runs),
        gradient_evaluations=sum(r['optimizer_calls'] if r['arm']=='adam' else r['closure_evaluations'] for r in runs),
        answer_targets=sum(r['answer_targets'] for r in runs),histories=sum(r['histories'] for r in runs),input_events=sum(r['input_events'] for r in runs),
        device='cpu',dtype='float64',threads=1,test_is_exploratory=True,selection='final_training_only',
        scores_sha256=prior.training.replay.digest(out/'scores.pt'),lock_sha256=prior.training.replay.digest(out/'lock.json'),
        pretest_sha256=prior.training.replay.digest(out/'pretest_decision.json'))
    prior.training.replay.check_files(files);prior.training.replay.write(out/'report.json',report)
    print(json.dumps(dict(status='complete',decision=choice)),flush=True)


def profile():
    cases,pools=inputs();tensor=tables(pools);records=[]
    for arm in ('adam','lbfgs'):
        model,opt,start=load(ROOT/cases[0]['checkpoint']);calls=0
        if arm=='lbfgs':opt=make_lbfgs(model)
        started=time.perf_counter()
        if arm=='lbfgs':
            def closure():
                nonlocal calls
                opt.zero_grad(set_to_none=True);loss=objective(model,tensor);calls+=1
                return torch.tensor(loss,dtype=torch.float64)
            opt.step(closure)
        else:
            e,q,y,_=tensor[32];opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(prior.training.forward_queries(model,e[:64],q[:64]).flatten(0,1),y[:64].flatten());loss.backward();opt.step();calls=1
        records.append(dict(arm=arm,seconds=time.perf_counter()-started,gradient_evaluations=calls,device='cpu',dtype='float64',
            histories_per_gradient=3072 if arm=='lbfgs' else 64,answer_targets=calls*44544 if arm=='lbfgs' else 2048,
            optimizer_state_present=bool(opt.state),peak_allocated_bytes=None,peak_reserved_bytes=None))
    return records


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser();parser.add_argument('--profile',action='store_true');args=parser.parse_args()
    if args.profile:
        path=ROOT/'results/full_pool_fitting_profiles.json';assert not path.exists()
        prior.training.replay.write(path,dict(records=profile(),sources={p:prior.training.replay.digest(ROOT/p) for p in SOURCES}))
    else:execute(OUT,*inputs())
