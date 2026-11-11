"""One direction intervention; reuse the archived full-objective training path."""
import argparse
import hashlib
import inspect
import json
import textwrap
import time
from unittest.mock import patch

import torch
import torch.optim.lbfgs as engine
import fitting_direction as previous
import fitting_representation as representation
import review_fitting_transport as transport
import audit_full_pool_fitting as numerical

ROOT=previous.ROOT
OUT=ROOT/'results/fitting_transport_training'
ARMS=('lbfgs','block_diagonal')
SOURCES=('experiments/train_fitting_transport.py','experiments/audit_fitting_transport_training.py','tests/test_fitting_transport_training.py')
PROTOCOL='docs/protocols/fitting-transport-training.md'
replay=previous.replay
TWO_LOOP=numerical.two_loop


def adapted_step():
    original=textwrap.dedent(inspect.getsource(engine.LBFGS.step))
    assert hashlib.sha256(original.encode()).hexdigest()==previous.NATIVE_HASH
    anchor='        gtd = flat_grad.dot(d)'
    assert original.count(anchor)==1
    replacement='        if self.direction_rule == "block_diagonal":\n            d = self.block_direction(flat_grad, dict(old_dirs=old_dirs, old_stps=old_stps, ro=ro, H_diag=H_diag))\n'
    adapted=original.replace(anchor,replacement+anchor)
    scope=dict(vars(engine));exec(compile(adapted,'<block-diagonal-lbfgs-step>','exec'),scope)
    return scope['step'],hashlib.sha256(adapted.encode()).hexdigest()


STEP,ADAPTED_HASH=adapted_step()


class TransportLBFGS(previous.DirectionLBFGS):
    step=STEP

    def __init__(self,parameters,rule,names):
        assert rule in ARMS
        parameters=list(parameters)
        assert len(parameters)==len(names) and set(names)==representation.ROUTING|representation.DECODER
        super().__init__(parameters,'lbfgs');self.direction_rule=rule
        self.routing_mask=torch.cat([torch.full((p.numel(),),n in representation.ROUTING,dtype=torch.bool,device=p.device) for n,p in zip(names,parameters)])
        assert self.routing_mask.any() and (~self.routing_mask).any()

    def block_direction(self,g,state):
        mask=self.routing_mask
        return TWO_LOOP(g*mask,state)*mask+TWO_LOOP(g*~mask,state)*~mask


def make_optimizer(model,arm):
    return TransportLBFGS(model.parameters(),arm,list(dict(model.named_parameters())))


def decision(runs):
    pairs=[]
    for seed in sorted({r['seed'] for r in runs}):
        a,b=[next(r for r in runs if (r['seed'],r['arm'])==(seed,arm)) for arm in ARMS]
        def stats(run):
            m=run['snapshots'][-1]['training']
            return dict(errors=sum(round((1-m[str(load)]['groups']['all']['accuracy'])*m[str(load)]['groups']['all']['count']) for load in previous.prior.LOADS),
                        ce=sum(m[str(load)]['groups']['all']['ce'] for load in previous.prior.LOADS)/4,
                        maps32=m['32']['complete_maps']['correct'])
        x,y=stats(a),stats(b)
        pairs.append(dict(seed=seed,full=x,block_diagonal=y,removal_increases_errors=y['errors']>x['errors']))
    return dict(pairs=pairs,all_seed_transport_fitting_benefit=all(p['removal_increases_errors'] for p in pairs))


def train(case,pools,out,arm,budget,interval,scores):
    model,_,_=previous.prior.load(ROOT/case['checkpoint']);names=list(dict(model.named_parameters()))
    with patch.object(previous,'DirectionLBFGS',lambda parameters,rule:TransportLBFGS(parameters,rule,names)):
        return previous.train(case,pools,out,arm,budget,interval,scores)


def files_for(cases,fixture):
    files={n:replay.digest(ROOT/n) for n in previous.prior.dependencies()|set(previous.SOURCES)|set(SOURCES)|{'experiments/review_fitting_transport.py','experiments/fitting_representation.py'}}
    for case in cases:files[case['checkpoint']]=replay.digest(ROOT/case['checkpoint'])
    if not fixture:
        for name in [PROTOCOL,'results/fitting_transport_training_preflight.json','results/fitting_transport_training_profiles.json',
                     'results/fitting_direction/report.json','results/fitting_direction/audit.json','results/fitting_direction/pools.pt',
                     'results/fitting_transport_review/report.json','results/fitting_transport_review/audit.json']:
            files[name]=replay.digest(ROOT/name)
        old=json.loads((previous.OUT/'report.json').read_text())
        for run in old['runs']:
            if run['arm']=='lbfgs':
                for name in [run['trace'],run['final_checkpoint']]:files[name]=replay.digest(ROOT/name)
        pre=json.loads((ROOT/'results/fitting_transport_training_preflight.json').read_text())
        profile=json.loads((ROOT/'results/fitting_transport_training_profiles.json').read_text())
        assert pre['passed'] and pre['sources']=={n:files[n] for n in SOURCES}==profile['sources']
    return files


def execute(out,cases,pools,budget=128,interval=32,fixture=False):
    assert not out.exists(),'Preserve evidence'
    files=files_for(cases,fixture)
    if not fixture:
        assert [c['seed'] for c in cases]==list(previous.SEEDS) and (budget,interval)==(128,32)
        oldaudit=json.loads((previous.OUT/'audit.json').read_text())
        assert oldaudit['passed'] and oldaudit['report_sha256']==replay.digest(previous.OUT/'report.json')
    for rows in pools.values():numerical.reference.independent.check_rows(rows)
    out.mkdir(parents=True);torch.save(pools,out/'pools.pt')
    replay.write(out/'lock.json',dict(files=files,cases=cases,budget=budget,interval=interval,fixture=fixture,
        pools_sha256=replay.digest(out/'pools.pt'),torch=torch.__version__,native_hash=previous.NATIVE_HASH,adapted_hash=ADAPTED_HASH))
    started=time.perf_counter();runs=[];scores=[]
    for case in cases:
        for arm in ARMS:
            print(json.dumps(dict(start_seed=case['seed'],arm=arm)),flush=True)
            runs.append(train(case,pools,out,arm,budget,interval,scores))
    choice=decision(runs);replay.write(out/'pretest_decision.json',choice)
    for run in runs:
        with torch.random.fork_rng(devices=[]):model=previous.prior.prior.training.previous.load_model(ROOT/run['final_checkpoint'])
        for split in ('validation','test'):
            for load in previous.prior.LOADS:
                cell=previous.prior.prior.training.evaluate(model,pools[f'{split}_{load}'])
                cell.update(seed=run['seed'],arm=run['arm'],step=run['snapshots'][-1]['step'],pool=f'{split}_{load}',checkpoint=run['final_checkpoint']);scores.append(cell)
    torch.save(scores,out/'scores.pt');replay.check_files(files)
    report=dict(status='complete',runs=runs,decision=choice,cells=[{k:v for k,v in c.items() if k not in ('logits','labels','repeat')} for c in scores],
        evaluation_targets=sum(c['labels'].numel() for c in scores),gradient_evaluations=sum(r['closure_evaluations'] for r in runs),
        optimizer_calls=sum(r['optimizer_calls'] for r in runs),answer_targets=sum(r['answer_targets'] for r in runs),
        histories=sum(r['histories'] for r in runs),input_events=sum(r['input_events'] for r in runs),seconds=time.perf_counter()-started,
        device='cpu',dtype='float64',threads=1,workers=1,test_is_exploratory=True,selection='none',
        scores_sha256=replay.digest(out/'scores.pt'),lock_sha256=replay.digest(out/'lock.json'),pretest_sha256=replay.digest(out/'pretest_decision.json'))
    replay.write(out/'report.json',report);print(json.dumps(dict(status='complete',decision=choice)),flush=True)


def profile():
    cases,pools=previous.inputs();records=[]
    def storage_bytes(state):
        if isinstance(state,torch.Tensor):return state.numel()*state.element_size()
        if isinstance(state,dict):return sum(storage_bytes(x) for x in state.values())
        if isinstance(state,(list,tuple)):return sum(storage_bytes(x) for x in state)
        return 0
    for arm in ARMS:
        model,_,_=previous.prior.load(ROOT/cases[0]['checkpoint']);opt=make_optimizer(model,arm);tables=previous.prior.tables(pools)
        for i in range(2):
            calls=0;started=time.perf_counter()
            def closure():
                nonlocal calls
                opt.zero_grad(set_to_none=True);loss=previous.prior.objective(model,tables);calls+=1
                return torch.tensor(loss,dtype=torch.float64)
            opt.step(closure)
            records.append(dict(arm=arm,call=i+1,seconds=time.perf_counter()-started,gradient_evaluations=calls,
                histories_per_gradient=sum(len(pools[f'train_{m}']) for m in previous.prior.LOADS),
                answers_per_gradient=sum(len(pools[f'train_{m}'])*m for m in previous.prior.LOADS),
                optimizer_tensor_bytes=storage_bytes(opt.state_dict()),peak_allocated_bytes=None,peak_reserved_bytes=None))
    return dict(records=records,device='cpu',dtype='float64',threads=1,sources={n:replay.digest(ROOT/n) for n in SOURCES})


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser();parser.add_argument('--profile',action='store_true');args=parser.parse_args()
    if args.profile:
        path=ROOT/'results/fitting_transport_training_profiles.json';assert not path.exists();replay.write(path,profile())
    else:execute(OUT,*previous.inputs())
