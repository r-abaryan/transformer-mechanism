"""Matched original-budget training with a fixed centered classifier norm cap."""
import argparse
import copy
import importlib.util
import json
import math
import time
import unittest
import torch
from torch.nn import functional as F
import query_feedback_continuation as continuation
import readout_scale_control as scale

training=continuation.parent
ROOT=continuation.ROOT
OUT=ROOT/'results/readout_norm_cap_training'
PROTOCOL='docs/protocols/readout-norm-cap-training.md'
SOURCES=('experiments/readout_norm_cap_training.py','experiments/audit_readout_norm_cap_training.py','tests/test_readout_norm_cap_training.py')
SEEDS=(3461,3463)
ARMS=('control','cap')
LOADS=(2,8,16,32)
UPDATES=1000


def inputs():
    data=json.loads((ROOT/'results/delta_query_coverage/development_data.json').read_text())['pools']
    test=json.loads((training.OUT/'test_data.json').read_text())['pools']
    cases=[dict(seed=s,checkpoint=f'results/query_feedback_training/{s}_exhaustive_3000.pt',
                reference=f'results/query_feedback_continuation/{s}_exhaustive_4000.pt') for s in SEEDS]
    return cases,{**data,**test}


@torch.no_grad()
def cap_head(model,radius,arm):
    norm=scale.head_norm(model.state_dict());factor=min(1.,radius/norm) if arm=='cap' else 1.
    assert arm in ARMS and math.isfinite(radius) and radius>0
    if factor<1:
        for p in (model.output.weight,model.output.bias):
            common=p.mean(0,keepdim=True);p.copy_(common+factor*(p-common))
    return dict(raw_norm=norm,factor=factor,after_norm=scale.head_norm(model.state_dict()))


def evaluate(path,rows):
    with torch.random.fork_rng(devices=[]):model=training.previous.load_model(path).eval()
    cell=training.evaluate(model,rows)
    with torch.no_grad():
        centered=cell['logits']-cell['logits'].mean(-1,keepdim=True)
        cell['diagnostics']=dict(head_norm=scale.head_norm(model.state_dict()),value_norm=float(model.value.weight.norm()),
            beta=float(model.write_logit.sigmoid()),centered_logit_rms=float(centered.square().mean().sqrt()))
    return cell


def fixed(z,y,mask):
    good=z.argmax(-1)==y;n=int(mask.sum());loss=-z.log_softmax(-1).gather(-1,y[...,None]).squeeze(-1)
    return dict(error_count=n,error_ce=float(loss[mask].mean()) if n else None,
                rescued=int((mask&good).sum()),harmed=int((~mask&~good).sum()))


def criteria(cells,seeds):
    results=[]
    for seed in seeds:
        def get(arm,pool):return next(c for c in cells if (c['seed'],c['arm'],c['pool'])==(seed,arm,pool))['metrics']
        a,b=(get(arm,'train_32') for arm in ARMS);x=a['groups']['all'];y=b['groups']['all']
        lower=all(get('cap',f'train_{m}')['groups']['all']['accuracy']>=get('control',f'train_{m}')['groups']['all']['accuracy'] for m in (2,8,16))
        train_ce={a:sum(get(a,f'train_{m}')['groups']['all']['ce'] for m in LOADS)/4 for a in ARMS}
        val_ce={a:sum(get(a,f'validation_{m}')['groups']['all']['ce'] for m in LOADS)/4 for a in ARMS}
        val_accuracy=all(get('cap',f'validation_{m}')['groups']['all']['accuracy']>=get('control',f'validation_{m}')['groups']['all']['accuracy'] for m in LOADS)
        benefit=y['accuracy']-x['accuracy']>=.005 and y['ce']<=.95*x['ce']+1e-10 and b['complete_maps']['correct']>=a['complete_maps']['correct'] and lower and train_ce['cap']<=train_ce['control']+1e-10
        fit={arm:all(get(arm,f'train_{m}')['groups']['all']['accuracy']>=.99 for m in LOADS) for arm in ARMS}
        results.append(dict(seed=seed,train32_accuracy_gain=y['accuracy']-x['accuracy'],train32_ce_delta=y['ce']-x['ce'],
            train_equal_load_ce_delta=train_ce['cap']-train_ce['control'],lower_train_preserved=lower,
            validation_accuracy_preserved=val_accuracy,validation_equal_load_ce_delta=val_ce['cap']-val_ce['control'],
            training_benefit=benefit,validation_preserved=val_accuracy and val_ce['cap']<=val_ce['control']+1e-10,training_fit=fit))
    return dict(seeds=results,training_benefit_all=all(x['training_benefit'] for x in results),validation_preserved_all=all(x['validation_preserved'] for x in results))


def train_window(case,pools,arm,folder,updates):
    with torch.random.fork_rng(devices=[]):model,opt,start=continuation.restore(ROOT/case['checkpoint'])
    radius=scale.head_norm(model.state_dict());hr,qr,_,tables,_=continuation.streams(case['seed'],dict(pools=pools),start)
    events={m:training.parent.study.parent.tensors(pools[f'train_{m}'])[0] for m in LOADS}
    traces=[];started=time.perf_counter();targets=0
    for step in range(start+1,start+updates+1):
        load=LOADS[(step-1)%4];indices=torch.randint(len(events[load]),(64,),generator=hr);draws=torch.randint(load,(64,load),generator=qr)
        q,y,_=training.choose(*tables[load],indices,draws,'exhaustive')
        opt.zero_grad(set_to_none=True);z=training.forward_queries(model,events[load][indices],q)
        loss=F.cross_entropy(z.reshape(-1,6),y.flatten());assert torch.isfinite(loss);loss.backward()
        gradients={n:p.grad.detach().clone() for n,p in model.named_parameters()};opt.step()
        raw=copy.deepcopy(model.state_dict());projection=cap_head(model,radius,arm)
        after=copy.deepcopy(model.state_dict())
        assert all(torch.equal(v,after[k]) for k,v in raw.items() if not k.startswith('output.'))
        traces.append(dict(seed=case['seed'],arm=arm,step=step,load=load,indices=indices,draws=draws,
            loss=float(loss.detach()),gradients=gradients,raw_model=raw,after_model=after,projection=projection))
        targets+=y.numel()
        if (step-start)%250==0:print(json.dumps(dict(seed=case['seed'],arm=arm,step=step,head_norm=projection['after_norm'])),flush=True)
    model.eval();path=folder/f"{case['seed']}_{arm}_final.pt";assert not path.exists()
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=start+updates,history_rng=hr.get_state(),query_rng=qr.get_state()),path)
    tracepath=folder/f"{case['seed']}_{arm}_traces.pt";torch.save(traces,tracepath)
    return dict(seed=case['seed'],arm=arm,start=start,stop=start+updates,radius=radius,updates=updates,histories=updates*64,input_events=updates*64*32,
        answer_targets=targets,projected_updates=sum(t['projection']['factor']<1 for t in traces),
        final_head_norm=scale.head_norm(model.state_dict()),value_norm=float(model.value.weight.detach().norm()),beta=float(model.write_logit.detach().sigmoid()),
        checkpoint=path.relative_to(ROOT).as_posix(),checkpoint_sha256=training.replay.digest(path),
        traces=tracepath.relative_to(ROOT).as_posix(),traces_sha256=training.replay.digest(tracepath),seconds=time.perf_counter()-started)


def profile():
    cases,pools=inputs();records=[]
    for arm in ARMS:
        with torch.random.fork_rng(devices=[]):model,opt,start=continuation.restore(ROOT/cases[0]['checkpoint'])
        e,q,y,_=training.replay.pressure.prepare(pools['train_32'][:64],'all_queries');radius=scale.head_norm(model.state_dict())
        began=time.perf_counter();opt.zero_grad(set_to_none=True);loss=F.cross_entropy(training.forward_queries(model,e,q).flatten(0,1),y.flatten())
        loss.backward();opt.step();projection=cap_head(model,radius,arm)
        assert all(int(s['step'])==start+1 for s in opt.state.values())
        records.append(dict(arm=arm,device='cpu',dtype='float64',batch=64,events_per_history=32,answer_targets=y.numel(),
            complete_step_seconds=time.perf_counter()-began,optimizer_state_restored=True,projection=projection,
            peak_allocated_bytes=None,peak_reserved_bytes=None))
    return records


def execute(out,cases=None,pools=None,fixture=False,updates=UPDATES):
    assert not out.exists(),'Preserve executed evidence'
    if cases is None:cases,pools=inputs()
    assert updates>0 and updates%4==0
    files={p:training.replay.digest(ROOT/p) for p in (*SOURCES,PROTOCOL,'experiments/query_feedback_continuation.py','experiments/query_feedback_training.py',
        'experiments/readout_scale_control.py','experiments/association_load.py','experiments/delta_retrieval.py','experiments/audit_objective_beta_pressure.py','experiments/audit_query_feedback_training.py','experiments/audit_shared_state_feasibility.py')}
    for case in cases:
        for k in ('checkpoint','reference'):
            if k in case:files[case[k]]=training.replay.digest(ROOT/case[k])
    if not fixture:
        assert updates==1000 and [c['seed'] for c in cases]==list(SEEDS)
        for p in ('results/delta_query_coverage/development_data.json','results/query_feedback_training/test_data.json',
                  'results/readout_norm_cap_training_preflight.json','results/readout_norm_cap_training_profiles.json'):
            files[p]=training.replay.digest(ROOT/p)
        pre=json.loads((ROOT/'results/readout_norm_cap_training_preflight.json').read_text());assert pre['passed'] and pre['sources']=={p:files[p] for p in SOURCES}
        for name in ('query_feedback_continuation','readout_scale_pressure','readout_adam_transport','beta_key_transport'):
            folder=ROOT/'results'/name;old=json.loads((folder/'audit.json').read_text());assert old['passed'] and old['report_sha256']==training.replay.digest(folder/'report.json')
            for n in ('lock.json','report.json','audit.json'):
                p=folder/n;files[p.relative_to(ROOT).as_posix()]=training.replay.digest(p)
    training.replay.check_files(files);out.mkdir(parents=True);training.replay.write(out/'pools.json',dict(pools=pools))
    training.replay.write(out/'lock.json',dict(files=files,cases=cases,fixture=fixture,updates=updates,arms=list(ARMS),loads=list(LOADS),
        pool_sha256=training.replay.digest(out/'pools.json'),versions=dict(torch=torch.__version__),decision_guard=1e-10))
    scores=[];cells=[];runs=[]
    def score(case,arm,path,split,mask):
        for m in LOADS:
            pool=f'{split}_{m}';cell=evaluate(path,pools[pool]);cell.update(seed=case['seed'],arm=arm,pool=pool,checkpoint=path.relative_to(ROOT).as_posix())
            if pool=='train_32':cell['fixed']=fixed(cell['logits'],cell['labels'],mask)
            scores.append(cell);cells.append({k:v for k,v in cell.items() if k not in ('logits','labels','repeat')})
    for case in cases:
        start=ROOT/case['checkpoint'];base=evaluate(start,pools['train_32']);mask=base['logits'].argmax(-1)!=base['labels']
        for split in ('train','validation'):score(case,'baseline',start,split,mask)
        for arm in ARMS:
            run=train_window(case,pools,arm,out,updates);runs.append(run)
            if arm=='control' and 'reference' in case:
                final=torch.load(ROOT/run['checkpoint'],weights_only=True);old=torch.load(ROOT/case['reference'],weights_only=True)
                for name in ('model','optimizer','step','history_rng','query_rng'):assert training.replay.equal(final[name],old[name]),'Control historical fidelity'
                run['historical_exact']=True
            else:run['historical_exact']=None
            for split in ('train','validation'):score(case,arm,ROOT/run['checkpoint'],split,mask)
    decision=criteria(cells,[c['seed'] for c in cases]);training.replay.write(out/'pretest_decision.json',decision)
    for case in cases:
        mask=next(c for c in scores if c['seed']==case['seed'] and c['arm']=='baseline' and c['pool']=='train_32')['logits'].argmax(-1)!=next(c for c in scores if c['seed']==case['seed'] and c['arm']=='baseline' and c['pool']=='train_32')['labels']
        score(case,'baseline',ROOT/case['checkpoint'],'test',mask)
        for arm in ARMS:
            run=next(r for r in runs if r['seed']==case['seed'] and r['arm']==arm);score(case,arm,ROOT/run['checkpoint'],'test',mask)
    torch.save(scores,out/'scores.pt')
    report=dict(status='complete',runs=runs,cells=cells,decision=decision,training_updates=sum(r['updates'] for r in runs),
        optimizer_calls=sum(r['updates'] for r in runs),histories=sum(r['histories'] for r in runs),input_events=sum(r['input_events'] for r in runs),
        answer_targets=sum(r['answer_targets'] for r in runs),evaluation_targets=sum(c['labels'].numel() for c in scores),
        device='cpu',dtype='float64',batch=64,threads=1,peak_allocated_bytes=None,peak_reserved_bytes=None,test_is_exploratory=True,
        lock_sha256=training.replay.digest(out/'lock.json'),scores_sha256=training.replay.digest(out/'scores.pt'),pretest_sha256=training.replay.digest(out/'pretest_decision.json'))
    training.replay.check_files(files);training.replay.write(out/'report.json',report);print(json.dumps(decision),flush=True)


def preflight():
    spec=importlib.util.spec_from_file_location('checks',ROOT/SOURCES[-1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))
    path=ROOT/('results/readout_norm_cap_training_preflight.json' if result.wasSuccessful() else f'results/readout_norm_cap_training_preflight_failed_{time.time_ns()}.json')
    assert not path.exists();training.replay.write(path,dict(passed=result.wasSuccessful(),methods=result.testsRun,sources={p:training.replay.digest(ROOT/p) for p in SOURCES}));assert result.wasSuccessful()
    p=ROOT/'results/readout_norm_cap_training_profiles.json';assert not p.exists();training.replay.write(p,dict(records=profile(),sources={p:training.replay.digest(ROOT/p) for p in SOURCES}))


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true')
    if p.parse_args().preflight:preflight()
    else:execute(OUT)
