"""Bounded matched delta training with a squared key-overlap penalty."""
import argparse
import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import struct
import time
import unittest

import torch
from torch.nn import functional as F
import key_geometry_retention as geometry

study=geometry.study
ROOT=study.ROOT
OUT=ROOT/'results/overlap_regularization'
DEV_SEEDS=(3407,3413)
CONFIRM_SEEDS=(3433,3449,3457)
WEIGHTS=(0.,1.,10.)
STEPS,BATCH=3000,64


def dependencies():
    paths=('experiments/overlap_regularization.py','tests/test_overlap_regularization.py',
           'docs/protocols/overlap-regularization.md',*geometry.dependencies())
    return {p:study.parent.digest(ROOT/p) for p in paths}


def penalty(model):
    k=model.keys(torch.arange(32,device=model.key.weight.device))
    gram=k@k.T
    return gram[~torch.eye(32,dtype=torch.bool,device=k.device)].square().mean()


def independent_penalty(model):
    raw=model.key.weight.detach().cpu().tolist()
    keys=[[x/math.sqrt(math.fsum(v*v for v in row)) for x in row] for row in raw]
    return math.fsum(math.fsum(x*y for x,y in zip(a,b))**2
                     for i,a in enumerate(keys) for j,b in enumerate(keys) if i!=j)/(32*31)


def metrics(model,rows,all_queries=False):
    if all_queries:
        scores,_=geometry.evaluate(model,rows,'native')
        return geometry.summarize(scores,rows),scores
    result,scores=study.parent.evaluate(model,rows)
    return result['all'],scores


def audit_cell(checkpoint,rows,all_queries,recorded,scores):
    model=geometry.prior.load(checkpoint,'delta')
    if all_queries:
        error,ref=geometry.audit_cell(checkpoint,rows,'native',scores,recorded)
    else:
        e,q,_=study.parent.tensors(rows)
        with torch.no_grad():ref=study.parent.reference(model,e,q)
        error=geometry.prior.check_scores(scores,ref)
        geometry.prior.check_metrics({'all':recorded},{'all':geometry.prior.reduce_scores(ref,rows)['all']})
    return error


def select(runs):
    lookup={(r['seed'],r['weight']):r for r in runs}
    assert set(lookup)==set(itertools.product(DEV_SEEDS,WEIGHTS)) and len(runs)==len(lookup)
    baseline_fit=all(lookup[s,0.]['metrics'][f'train_{m}']['accuracy']>=.99 for s in DEV_SEEDS for m in study.LOADS)
    candidates=[]
    for w in WEIGHTS[1:]:
        fit=all(lookup[s,w]['metrics'][f'train_{m}']['accuracy']>=.99 for s in DEV_SEEDS for m in study.LOADS)
        preserve=all(lookup[s,w]['metrics'][f'validation_{m}']['accuracy']+.02>=lookup[s,0.]['metrics'][f'validation_{m}']['accuracy'] for s in DEV_SEEDS for m in (2,8))
        candidates.append(dict(weight=w,eligible=baseline_fit and fit and preserve,
            training_fit=fit,low_load_preserved=preserve,
            validation_all_query_accuracy=sum(lookup[s,w]['metrics']['all_validation_32']['all_queries']['accuracy'] for s in DEV_SEEDS)/len(DEV_SEEDS),
            validation_all_query_ce=sum(lookup[s,w]['metrics']['all_validation_32']['all_queries']['loss'] for s in DEV_SEEDS)/len(DEV_SEEDS)))
    eligible=[c for c in candidates if c['eligible']]
    chosen=min(eligible,key=lambda c:(-c['validation_all_query_accuracy'],c['validation_all_query_ce'],c['weight']))['weight'] if eligible else None
    return dict(baseline_training_fit=baseline_fit,candidates=candidates,selected_weight=chosen)


def decision(runs):
    if not runs:return dict(confirmation_executed=False)
    lookup={(r['seed'],r['arm']):r for r in runs}
    assert len(runs)==6 and set(lookup)==set(itertools.product(CONFIRM_SEEDS,('baseline','penalty')))
    fit=all(r['metrics'][f'train_{m}']['accuracy']>=.99 for r in runs for m in study.LOADS)
    gain=all(lookup[s,'penalty']['metrics']['all_test_32']['all_queries']['accuracy']-lookup[s,'baseline']['metrics']['all_test_32']['all_queries']['accuracy']>=.10 for s in CONFIRM_SEEDS)
    preserve=all(lookup[s,'penalty']['metrics'][f'test_{m}']['accuracy']+.02>=lookup[s,'baseline']['metrics'][f'test_{m}']['accuracy'] for s in CONFIRM_SEEDS for m in (2,8))
    simultaneous_fit=all(r['metrics']['all_train_32']['all_queries']['accuracy']>=.99 for r in runs)
    reliable=all(lookup[s,'penalty']['metrics']['all_test_32']['complete_maps']['accuracy']>=.95 for s in CONFIRM_SEEDS)
    return dict(confirmation_executed=True,supervised_training_fit=fit,all_seed_10_point_all_query_gain=gain,
        low_load_preservation=preserve,all_query_training_adequacy=simultaneous_fit,
        penalty_complete_map_reliability=reliable,bounded_benefit=fit and gain and preserve,
        simultaneous_retention_supported=fit and gain and preserve and simultaneous_fit and reliable)


def profiles():
    results=[]
    for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
        for weight in (0.,10.):
            model=study.pair(2026100440)['delta'].to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
            e=torch.stack([torch.tensor([[k,k%6] for k in range(32)]) for _ in range(BATCH)]).to(device)
            q=torch.arange(BATCH,device=device)%32;y=q%6
            if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad(set_to_none=True)
            ce=F.cross_entropy(model(e,q),y);reg=penalty(model)
            objective=ce+weight*reg
            objective.backward();optimizer.step()
            assert all(int(state['step'])==1 for state in optimizer.state.values())
            if device=='cuda':torch.cuda.synchronize()
            results.append(dict(device=device,weight=weight,dtype='float64',batch=BATCH,events=32,
                optimizer_state_allocated=True,objective=float(objective.detach()),
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None))
            del model,optimizer,objective,ce,reg,e,q,y
    return results


def train_run(phase,seed,weight,arm,payload,report,saved):
    model=study.pair(seed)['delta'];optimizer=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
    data={m:study.parent.tensors(payload['pools'][f'train_{m}']) for m in study.LOADS}
    rng=torch.Generator().manual_seed(seed+10000);stream=hashlib.sha256()
    run=dict(phase=phase,seed=seed,weight=weight,arm=arm,status='running',metrics={},curve=[],
        initial_sha256=study.parent.state_hash(model.state_dict()),parameters=sum(p.numel() for p in model.parameters()))
    report['runs'].append(run);study.parent.write(OUT/'report.json',report);tick=time.perf_counter()
    for step in range(1,STEPS+1):
        load=study.LOADS[(step-1)%len(study.LOADS)];e,q,y=data[load]
        idx=torch.randint(len(e),(BATCH,),generator=rng)
        stream.update(struct.pack('<I',load));stream.update(bytes(idx.view(torch.uint8).tolist()))
        optimizer.zero_grad(set_to_none=True)
        ce=F.cross_entropy(model(e[idx],q[idx]),y[idx]);reg=penalty(model)
        objective=ce+weight*reg;assert torch.isfinite(objective)
        objective.backward();optimizer.step()
        if step%1000==0:
            run['curve'].append(dict(step=step,ce=float(ce.detach()),penalty=float(reg.detach()),objective=float(objective.detach())))
            print(dict(phase=phase,seed=seed,weight=weight,step=step),flush=True)
    cp=OUT/f'{phase}_{seed}_{arm}_{weight:g}.pt';assert not cp.exists()
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=STEPS),cp)
    for pool,rows in payload['pools'].items():
        if pool.startswith('test_'):continue
        run['metrics'][pool],saved[f'{phase}_{seed}_{weight}_{pool}']=metrics(model,rows)
        if pool.endswith('_32'):
            key='all_'+pool
            run['metrics'][key],saved[f'{phase}_{seed}_{weight}_{key}']=metrics(model,rows,True)
    run.update(status='complete',updates=STEPS,examples=STEPS*BATCH,input_events=STEPS*BATCH*32,
        examples_per_load=STEPS*BATCH//4,stream_sha256=stream.hexdigest(),penalty=float(penalty(model).detach()),
        beta=float(model.write_logit.sigmoid().detach()),elapsed_seconds=time.perf_counter()-tick,
        checkpoint=cp.relative_to(ROOT).as_posix(),checkpoint_sha256=study.parent.digest(cp))
    study.parent.write(OUT/'report.json',report)


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    geometry.prior.parent_evidence()
    old=json.loads((geometry.OUT/'report.json').read_text());old_audit=json.loads((geometry.OUT/'independent_audit.json').read_text())
    assert old_audit['passed'] and old_audit['report_sha256']==study.parent.digest(geometry.OUT/'report.json') and old['source_dependencies']==geometry.dependencies()
    assert not OUT.exists();OUT.mkdir();source=dependencies()
    protected=json.loads((geometry.OUT/'preserved.json').read_text())
    for p in geometry.OUT.rglob('*'):
        if p.is_file():protected[p.relative_to(ROOT).as_posix()]=study.parent.digest(p)
    for p in ('docs/topics/attention/key-geometry-retention-results.md','docs/topics/attention/key-geometry-retention-guide.md'):
        protected[p]=study.parent.digest(ROOT/p)
    assert all(study.parent.digest(ROOT/p)==h for p,h in protected.items());study.parent.write(OUT/'preserved.json',protected)
    spec=importlib.util.spec_from_file_location('test_overlap_regularization',ROOT/'tests/test_overlap_regularization.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    study.parent.write(OUT/'preflight.json',dict(passed=result.wasSuccessful(),methods=result.testsRun,source_dependencies=source))
    assert result.wasSuccessful(),'Preflight failed; no formal training'
    profile=profiles();study.parent.write(OUT/'profiles.json',profile)
    excluded=set()
    for phase in ('development','confirmation'):
        olddata=json.loads((study.OUT/f'{phase}_data.json').read_text())
        excluded.update(study.family(r['events']) for rows in olddata['pools'].values() for r in rows)
    dev=study.generate(2026100441,exclude=excluded)
    assert all(study.family(r['events']) not in excluded for rows in dev['pools'].values() for r in rows)
    study.parent.write(OUT/'development_data.json',dev)
    study.parent.write(OUT/'lock.json',dict(source_dependencies=source,weights=WEIGHTS,development_seeds=DEV_SEEDS,confirmation_seeds=CONFIRM_SEEDS,
        steps=STEPS,batch=BATCH,profiles_sha256=study.parent.digest(OUT/'profiles.json'),preflight_sha256=study.parent.digest(OUT/'preflight.json')))
    report=dict(status='development',source_dependencies=source,device='cpu',dtype='float64',threads=1,
        runs=[],baselines=dict(development=study.audit_data(dev)),data_sha256=dict(development=study.parent.digest(OUT/'development_data.json')))
    saved={}
    for seed in DEV_SEEDS:
        for weight in WEIGHTS if seed==DEV_SEEDS[0] else WEIGHTS[::-1]:train_run('development',seed,weight,'baseline' if weight==0 else 'penalty',dev,report,saved)
    report['selection']=select(report['runs']);study.parent.write(OUT/'development_locked.json',report)
    chosen=report['selection']['selected_weight']
    if chosen is not None:
        excluded.update(study.family(r['events']) for rows in dev['pools'].values() for r in rows)
        fresh=study.generate(2026100442,confirmation=True,exclude=excluded)
        assert all(study.family(r['events']) not in excluded for rows in fresh['pools'].values() for r in rows)
        study.audit_data(dict(pools={**{'dev_'+k:v for k,v in dev['pools'].items()},**{'confirm_'+k:v for k,v in fresh['pools'].items()}}))
        study.parent.write(OUT/'confirmation_data.json',fresh)
        report['data_sha256']['confirmation']=study.parent.digest(OUT/'confirmation_data.json');report['baselines']['confirmation']=study.audit_data(fresh)
        for seed in CONFIRM_SEEDS:
            for arm,w in (('baseline',0.),('penalty',chosen)) if seed!=CONFIRM_SEEDS[1] else (('penalty',chosen),('baseline',0.)):
                train_run('confirmation',seed,w,arm,fresh,report,saved)
        study.parent.write(OUT/'pretest.json',report)
        for item in [r for r in report['runs'] if r['phase']=='confirmation']:
            model=geometry.prior.load(ROOT/item['checkpoint'],'delta')
            for pool,rows in fresh['pools'].items():
                if not pool.startswith('test_'):continue
                item['metrics'][pool],saved[f"confirmation_{item['seed']}_{item['weight']}_{pool}"]=metrics(model,rows)
                if pool=='test_32':item['metrics']['all_test_32'],saved[f"confirmation_{item['seed']}_{item['weight']}_all_test_32"]=metrics(model,rows,True)
    else:report['confirmation_skipped']='No eligible penalty under frozen fit/preservation criteria.'
    report['decision']=decision([r for r in report['runs'] if r['phase']=='confirmation'])
    assert dependencies()==source and all(study.parent.digest(ROOT/p)==h for p,h in protected.items())
    torch.save(saved,OUT/'scores.pt');report.update(status='complete',scores_sha256=study.parent.digest(OUT/'scores.pt'),protected_files_unchanged=len(protected))
    study.parent.write(OUT/'report.json',report)
    print('Bounded training complete. Run audit next.',flush=True)


def audit():
    torch.set_num_threads(1)
    r=json.loads((OUT/'report.json').read_text());lock=json.loads((OUT/'lock.json').read_text())
    assert r['status']=='complete' and r['source_dependencies']==lock['source_dependencies']==dependencies()
    assert r['scores_sha256']==study.parent.digest(OUT/'scores.pt')
    saved=torch.load(OUT/'scores.pt',weights_only=True);data={}
    for phase,h in r['data_sha256'].items():
        assert study.parent.digest(OUT/f'{phase}_data.json')==h;data[phase]=json.loads((OUT/f'{phase}_data.json').read_text())
        assert study.audit_data(data[phase])==r['baselines'][phase]
    study.audit_data(dict(pools={f'{phase}_{name}':rows for phase,p in data.items() for name,rows in p['pools'].items()}))
    oldfamilies={study.family(row['events']) for phase in ('development','confirmation') for rows in json.loads((study.OUT/f'{phase}_data.json').read_text())['pools'].values() for row in rows}
    assert all(study.family(row['events']) not in oldfamilies for p in data.values() for rows in p['pools'].values() for row in rows)
    errors,records,count,used=[],[],0,set()
    for item in r['runs']:
        assert item['status']=='complete' and item['parameters']==615 and item['updates']==STEPS and item['examples']==STEPS*BATCH
        cp=ROOT/item['checkpoint'];assert study.parent.digest(cp)==item['checkpoint_sha256']
        checkpoint=torch.load(cp,weights_only=True);assert checkpoint['step']==STEPS and all(int(state['step'])==STEPS for state in checkpoint['optimizer']['state'].values())
        model=geometry.prior.load(cp,'delta');assert abs(independent_penalty(model)-item['penalty'])<1e-12
        assert item['initial_sha256']==study.parent.state_hash(study.pair(item['seed'])['delta'].state_dict())
        generator=torch.Generator().manual_seed(item['seed']+10000);stream=hashlib.sha256()
        for step in range(STEPS):
            load=study.LOADS[step%4];idx=torch.randint(len(data[item['phase']]['pools'][f'train_{load}']),(BATCH,),generator=generator)
            stream.update(struct.pack('<I',load));stream.update(bytes(idx.view(torch.uint8).tolist()))
        assert stream.hexdigest()==item['stream_sha256']
        for name,value in item['metrics'].items():
            allq=name.startswith('all_');pool=name[4:] if allq else name
            rows=data[item['phase']]['pools'][pool];key=f"{item['phase']}_{item['seed']}_{item['weight']}_{name}"
            scores=saved[key];used.add(key);errors.append(audit_cell(cp,rows,allq,value,scores));count+=scores.numel()//6
            m=value['all_queries'] if allq else value
            records.append(dict(phase=item['phase'],seed=item['seed'],arm=item['arm'],weight=item['weight'],pool=name,**m,
                                complete_map_accuracy=value['complete_maps']['accuracy'] if allq else '',penalty=item['penalty']))
    assert used==set(saved)
    dev=[x for x in r['runs'] if x['phase']=='development'];assert select(dev)==r['selection']
    conf=[x for x in r['runs'] if x['phase']=='confirmation'];assert decision(conf)==r['decision']
    assert len(dev)==6 and len(conf)==(6 if r['selection']['selected_weight'] is not None else 0)
    for phase,seeds in (('development',DEV_SEEDS),('confirmation',CONFIRM_SEEDS)):
        for seed in seeds:
            group=[x for x in r['runs'] if x['phase']==phase and x['seed']==seed]
            if group:assert len({x['initial_sha256'] for x in group})==len({x['stream_sha256'] for x in group})==1
    protected=json.loads((OUT/'preserved.json').read_text());assert all(study.parent.digest(ROOT/p)==h for p,h in protected.items())
    assert not (OUT/'independent_audit.json').exists()
    with (OUT/'metrics.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    study.parent.write(OUT/'independent_audit.json',dict(passed=True,report_sha256=study.parent.digest(OUT/'report.json'),source_dependencies=dependencies(),
        runs=len(r['runs']),metric_cells=len(records),predictions=count,max_logit_error=max(errors),argmax_disagreements=0,
        protected_files_unchanged=len(protected),initialization_and_streams_verified=True,
        scope='Independent equations/Python losses/complete-map counts and penalty; selection/decision replay shares producer functions, preflight inverse controls. Shared weights/data/PyTorch.'))
    print('Overlap training audit passes.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('run','audit'));args=parser.parse_args()
    run() if args.phase=='run' else audit()
