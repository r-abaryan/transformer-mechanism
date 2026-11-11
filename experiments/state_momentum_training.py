"""Matched training-only continuation with one first-moment intervention."""
import copy
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from state_tracking import ROOT, model_new, forward, tensors, answer, save_json
from audit_state_tracking import independent, oracle
from state_reset_exposure import encode_base, paired_reset_metrics
from state_recipe_development import evaluate
from state_training_gradients import KEYS, WEIGHTS, scalar_loss, scalar_pairs
from state_adam_momentum import scalar_step
from mqa_training import hash_state
from position_distractors import digest

DIRECTORY = ROOT/'results/state_momentum_training'
PROTOCOL = ROOT/'docs/protocols/state-momentum-training.md'
SEEDS = (3011,3019,3023)
ARMS = ('retained','clear_once')
STEPS = 2000
BATCH = 64
OBSERVATIONS = (0,1,50,200,500,1000,2000)


def inputs():
    parent = ROOT/'results/state_reset_exposure'
    old = json.loads((parent/'report.json').read_text(encoding='utf-8'))
    data = json.loads((parent/'data.json').read_text(encoding='utf-8'))
    assert digest(parent/'data.json')==old['data_sha256']
    runs = {r['seed']:r for r in old['runs'] if r['arm']=='independent'}
    assert set(runs)==set(SEEDS)
    sources = {**old['source_dependencies'],
               'experiments/state_momentum_training.py':digest(Path(__file__)),
               'experiments/state_training_gradients.py':digest(ROOT/'experiments/state_training_gradients.py'),
               'experiments/state_adam_momentum.py':digest(ROOT/'experiments/state_adam_momentum.py'),
               'results/state_reset_exposure/report.json':digest(parent/'report.json'),
               'results/state_reset_exposure/data.json':digest(parent/'data.json')}
    assert all(digest(ROOT/p)==h for p,h in old['source_dependencies'].items())
    for r in runs.values():
        for key in ('checkpoint','scores'):
            assert digest(ROOT/r[key])==r[key+'_sha256'];sources[r[key]]=r[key+'_sha256']
    support = {k:data['evaluation'][k] for k in KEYS}
    assert all(oracle(x)==answer(x[0]-4,x[1:-1])==y for rows in support.values() for x,y in rows)
    assert all(sum(y for x,y in rows)==len(rows)//2 for rows in support.values())
    return runs, data['training_base'], support, sources


def initialize(checkpoint, arm):
    model = model_new().eval();model.load_state_dict(checkpoint['model'])
    opt = torch.optim.Adam(model.parameters())
    opt.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    group = opt.param_groups[0]
    assert group['lr']==.0001 and group['betas']==(.9,.999) and group['eps']==1e-8
    assert group['weight_decay']==0 and not any(group[k] for k in ('amsgrad','maximize','decoupled_weight_decay'))
    assert all(float(v['step'])==6000 for v in opt.state.values())
    if arm=='clear_once':
        for v in opt.state.values():v['exp_avg'].zero_()
    return model,opt


def sample(encoded, gen, aug, step):
    n = 8 if step%2 else 12
    x,y,mask = encoded[n]
    ix = torch.randint(len(y),(BATCH,),generator=gen)
    flip = torch.randint(2,(BATCH,),generator=aug).bool()
    bx=x[ix].clone();long=mask[ix]
    bx[long & flip,0]=9-bx[long & flip,0]
    return n,bx,y[ix],ix,flip,long


def choose(runs):
    options = []
    for arm in ARMS:
        rows = [r for r in runs if r['arm']==arm]
        eligible = len(rows)==3 and all(r['status']=='trained' and min(m['accuracy'] for m in r['metrics'].values())>=.99 for r in rows)
        options.append(dict(arm=arm,eligible=eligible,
                            worst_accuracy=min((m['accuracy'] for r in rows for m in r.get('metrics',{}).values()),default=0.),
                            worst_loss=max((m['loss'] for r in rows for m in r.get('metrics',{}).values()),default=None)))
    eligible = [r for r in options if r['eligible']]
    selected = min(eligible,key=lambda r:(r['worst_loss'],ARMS.index(r['arm'])))['arm'] if eligible else None
    return dict(options=options,selected_for_confirmation=selected,
                rule='final >=99% accuracy in all eight training support cells in all three seeds; smallest worst-cell CE among eligible arms; retained wins exact tie')


def profile():
    torch.set_num_threads(1)
    assert not DIRECTORY.exists()
    runs,base,pools,sources = inputs()
    cp=torch.load(ROOT/runs[3011]['checkpoint'],weights_only=True)
    x,y=tensors(base['train/12/long_tail'][:BATCH]);x=x.clone();x[::2,0]=9-x[::2,0]
    records=[]
    for arm in ARMS:
        m,opt=initialize(cp,arm)
        tick=time.perf_counter();opt.zero_grad();loss=F.cross_entropy(forward(m,x)[:,-1],y)
        loss.backward();opt.step()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters())
        records.append(dict(arm=arm,full_step_seconds=time.perf_counter()-tick,optimizer_entries=len(opt.state),
                            optimizer_tensor_bytes=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))))
    DIRECTORY.mkdir()
    prof=dict(python=platform.python_version(),torch=str(torch.__version__),device='cpu',dtype='float64',threads=1,
              parameters=4080,effective_batch=BATCH,operation_length=12,input_tokens_per_example=14,
              gpu_peak_allocated_bytes=0,gpu_peak_reserved_bytes=0,cpu_peak_memory_measured=False,
              discarded_profile_steps=2,source_dependencies=sources,arms=records)
    save_json(DIRECTORY/'profile.json',prof)
    save_json(DIRECTORY/'training_data.json',dict(training_base=base,training_support=pools))
    print(json.dumps({k:v for k,v in prof.items() if k!='source_dependencies'}))


def train():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert PROTOCOL.exists() and not (DIRECTORY/'report.json').exists() and not (DIRECTORY/'checkpoints').exists()
    prof=json.loads((DIRECTORY/'profile.json').read_text(encoding='utf-8'))
    assert all(digest(ROOT/p)==h for p,h in prof['source_dependencies'].items())
    parent_runs,base,pools,sources=inputs();sources[PROTOCOL.relative_to(ROOT).as_posix()]=digest(PROTOCOL)
    encoded=encode_base(base)
    report=dict(status='running',source_dependencies=sources,profile_sha256=digest(DIRECTORY/'profile.json'),
                data_sha256=digest(DIRECTORY/'training_data.json'),steps=STEPS,batch=BATCH,
                observation_steps=list(OBSERVATIONS),seeds=list(SEEDS),selection_uses_training_only=True,
                validation_test_evaluated=False,runs=[])
    (DIRECTORY/'checkpoints').mkdir();path=DIRECTORY/'report.json';save_json(path,report)
    for si,seed in enumerate(SEEDS):
        checkpoint=torch.load(ROOT/parent_runs[seed]['checkpoint'],weights_only=True)
        archived=torch.load(ROOT/parent_runs[seed]['scores'],weights_only=True)
        for arm in (ARMS if si%2==0 else ARMS[::-1]):
            m,opt=initialize(checkpoint,arm)
            initial_hash=hash_state(m.state_dict())
            gen=torch.Generator();gen.set_state(checkpoint['generator'])
            aug=torch.Generator();aug.set_state(checkpoint['augmentation_generator'])
            stream=hashlib.sha256();coins=hashlib.sha256();counts={};curves={}
            metrics,scores=evaluate(m,pools)
            assert all(torch.equal(scores[k],archived[k]) for k in KEYS)
            run=dict(seed=seed,arm=arm,status='running',step=0,initial_model_sha256=initial_hash,
                     baseline=metrics,baseline_pairs=paired_reset_metrics(scores,pools),curve=[])
            curves[0]=scores
            run['curve'].append(dict(step=0,metrics=metrics,reset_pairs=run['baseline_pairs']))
            report['runs'].append(run);save_json(path,report);tick=time.perf_counter()
            try:
                for step in range(1,STEPS+1):
                    n,bx,by,ix,flip,long=sample(encoded,gen,aug,step)
                    stream.update(json.dumps(ix.tolist()).encode());coins.update(json.dumps(flip.tolist()).encode())
                    for initial,reset in bx[long][:,:2].tolist():
                        key=f'{n}/{initial-4}/{reset-2}';counts[key]=counts.get(key,0)+1
                    opt.zero_grad();loss=F.cross_entropy(forward(m,bx)[:,-1],by)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite loss')
                    loss.backward()
                    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters()):raise FloatingPointError('nonfinite gradient')
                    if step==1:
                        rz=independent(m,bx)[:,-1]
                        rg=torch.autograd.grad(F.cross_entropy(rz,by),list(m.parameters()))
                        error=max(float((p.grad-g).abs().max()) for p,g in zip(m.parameters(),rg))
                        assert error<=1e-10
                        bad=torch.autograd.grad(F.cross_entropy(independent(m,bx)[:,-1],1-by),list(m.parameters()))
                        negative=max(float((p.grad-g).abs().max()) for p,g in zip(m.parameters(),bad));assert negative>1e-6
                        before=[p.detach().clone() for p in m.parameters()]
                        gradient=torch.cat([p.grad.reshape(-1) for p in m.parameters()])
                        expect,em,ev=scalar_step(opt.state_dict(),gradient,[p.numel() for p in m.parameters()],False)
                    opt.step();run['step']=step
                    if not all(bool(torch.isfinite(p).all()) for p in m.parameters()):raise FloatingPointError('nonfinite parameter')
                    if step==1:
                        actual=torch.cat([(p.detach()-q).reshape(-1) for p,q in zip(m.parameters(),before)])
                        update_error=float((actual-expect).abs().max());assert update_error<=1e-12
                        moment_error=max(float((torch.cat([v[k].reshape(-1) for v in opt.state.values()])-ref).abs().max()) for k,ref in (('exp_avg',em),('exp_avg_sq',ev)))
                        assert moment_error<=1e-12
                        run['first_step_check']=dict(gradient_error=error,wrong_label_gradient_effect=negative,adam_error=update_error,moment_error=moment_error)
                    if step in OBSERVATIONS:
                        metrics,scores=evaluate(m,pools);curves[step]=scores
                        pairs=paired_reset_metrics(scores,pools)
                        run['curve'].append(dict(step=step,metrics=metrics,reset_pairs=pairs));save_json(path,report)
                        print(json.dumps(dict(seed=seed,arm=arm,step=step,worst_training=min(v['accuracy'] for v in metrics.values()),train12_disagree=metrics['train/12/long_tail_disagree']['accuracy'])),flush=True)
                run['status']='trained'
            except Exception as exc:
                run.update(status='failed',failure=repr(exc));print(json.dumps(dict(seed=seed,arm=arm,failure=repr(exc))),flush=True)
            cp=DIRECTORY/'checkpoints'/f'{seed}_{arm}.pt'
            torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),generator=gen.get_state(),augmentation_generator=aug.get_state(),steps=6000+run['step']),cp)
            sp=DIRECTORY/'checkpoints'/f'{seed}_{arm}_curves.pt';torch.save(curves,sp)
            run.update(checkpoint=cp.relative_to(ROOT).as_posix(),checkpoint_sha256=digest(cp),scores=sp.relative_to(ROOT).as_posix(),scores_sha256=digest(sp),
                       stream_sha256=stream.hexdigest(),augmentation_sha256=coins.hexdigest(),exposure_counts=counts,
                       seconds=time.perf_counter()-tick,additional_examples=run['step']*BATCH,additional_supervised_targets=run['step']*BATCH,
                       additional_input_tokens=sum(BATCH*(10 if s%2 else 14) for s in range(1,run['step']+1)))
            if run['status']=='trained':
                run['metrics']=run['curve'][-1]['metrics'];run['reset_pairs']=run['curve'][-1]['reset_pairs']
            save_json(path,report)
    assert all(digest(ROOT/p)==h for p,h in sources.items())
    report.update(status='finished',decision=choose(report['runs']),prior_hashes_unchanged=True);save_json(path,report)
    print(json.dumps(report['decision']),flush=True)


def audit():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert not (DIRECTORY/'audit.json').exists()
    r=json.loads((DIRECTORY/'report.json').read_text(encoding='utf-8'))
    assert r['status']=='finished' and len(r['runs'])==6
    assert digest(DIRECTORY/'profile.json')==r['profile_sha256'] and digest(DIRECTORY/'training_data.json')==r['data_sha256']
    assert all(digest(ROOT/p)==h for p,h in r['source_dependencies'].items())
    parents,base,pools,_=inputs();encoded=encode_base(base);cells=[];rows=[];curve_evals=0;curve_error=0.;pair_error=0.
    for seed in SEEDS:
        pair=[v for v in r['runs'] if v['seed']==seed]
        assert len(pair)==2 and {v['arm'] for v in pair}==set(ARMS)
        cp=torch.load(ROOT/parents[seed]['checkpoint'],weights_only=True)
        expected_initial=hash_state(cp['model'])
        assert all(v['initial_model_sha256']==expected_initial for v in pair)
        gen=torch.Generator();gen.set_state(cp['generator']);aug=torch.Generator();aug.set_state(cp['augmentation_generator'])
        stream=hashlib.sha256();coins=hashlib.sha256();counts={}
        for step in range(1,STEPS+1):
            n,bx,by,ix,flip,long=sample(encoded,gen,aug,step)
            assert all(oracle(x)==y for x,y in zip(bx.tolist(),by.tolist()))
            stream.update(json.dumps(ix.tolist()).encode());coins.update(json.dumps(flip.tolist()).encode())
            for initial,reset in bx[long][:,:2].tolist():
                key=f'{n}/{initial-4}/{reset-2}';counts[key]=counts.get(key,0)+1
        for run in pair:
            assert run['status']=='trained' and run['step']==2000
            assert run['stream_sha256']==stream.hexdigest() and run['augmentation_sha256']==coins.hexdigest()
            assert run['exposure_counts']==counts and run['additional_examples']==run['additional_supervised_targets']==128000
            assert run['additional_input_tokens']==1536000
            assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256'] and digest(ROOT/run['scores'])==run['scores_sha256']
            saved=torch.load(ROOT/run['checkpoint'],weights_only=True)
            assert saved['steps']==8000 and all(float(s['step'])==8000 for s in saved['optimizer']['state'].values())
            assert torch.equal(saved['generator'],gen.get_state()) and torch.equal(saved['augmentation_generator'],aug.get_state())
            assert saved['optimizer']['param_groups'][0]['lr']==.0001
            m=model_new().eval();m.load_state_dict(saved['model'])
            metrics,z=evaluate(m,pools,independent)
            history=torch.load(ROOT/run['scores'],weights_only=True);assert set(history)==set(OBSERVATIONS)
            for point in run['curve']:
                zz=history[point['step']]
                pp=scalar_pairs(zz,pools)
                for prefix in pp:
                    pair_error=max(pair_error,max(abs(v-point['reset_pairs'][prefix][k]) for k,v in pp[prefix].items()))
                for key in KEYS:
                    y=tensors(pools[key])[1];scores=zz[key]
                    ce=scalar_loss(scores,y);acc=sum(int(b>a)==truth for (a,b),truth in zip(scores.tolist(),y.tolist()))/len(y)
                    curve_error=max(curve_error,abs(ce-point['metrics'][key]['loss']),abs(acc-point['metrics'][key]['accuracy']))
                    curve_evals+=len(y)
            for key in KEYS:
                error=float((z[key]-history[2000][key]).abs().max())
                assert error<=1e-10 and torch.equal(z[key].argmax(-1),history[2000][key].argmax(-1))
                cells.append(dict(seed=seed,arm=run['arm'],group=key,count=len(z[key]),logit_error=error))
                rows.append(dict(seed=seed,arm=run['arm'],group=key,**run['metrics'][key]))
    assert max(curve_error,pair_error)<=1e-10 and choose(r['runs'])==r['decision']
    assert all(digest(ROOT/p)==h for p,h in r['source_dependencies'].items())
    out=dict(passed=True,report_sha256=digest(DIRECTORY/'report.json'),final_cells=len(cells),
             final_predictions=sum(c['count'] for c in cells),curve_predictions_reduced=curve_evals,
             max_logit_error=max(c['logit_error'] for c in cells),max_curve_metric_error=curve_error,max_pair_error=pair_error,
             argmax_changes=0,stream_replays=6,sampled_training_labels_checked=3*STEPS*BATCH,cells=cells)
    save_json(DIRECTORY/'audit.json',out)
    with (DIRECTORY/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in out.items() if k!='cells'}))


if __name__=='__main__':
    {'profile':profile,'train':train,'audit':audit}[sys.argv[1]]()
