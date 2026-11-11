"""Matched fixed-query versus resampled-query training at one target per history."""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import struct
import time
import unittest

import torch
from torch.nn import functional as F
import overlap_regularization as previous

study=previous.study
ROOT=study.ROOT
OUT=ROOT/'results/delta_query_coverage'
ARMS=('fixed','resampled')
DEV_SEEDS=(3461,3463)
CONFIRM_SEEDS=(3467,3469,3491)
STEPS,BATCH=3000,64


def dependencies():
    paths=('experiments/delta_query_coverage.py','tests/test_delta_query_coverage.py',
           'docs/protocols/delta-query-coverage.md',*previous.dependencies())
    return {p:study.parent.digest(ROOT/p) for p in paths}


def query_tables(rows):
    queries,labels=[],[]
    for row in rows:
        memory={}
        for k,v in row['events']:
            assert k not in memory or memory[k]==v,'Conflicting assignment'
            memory[k]=v
        assert memory[row['query']]==row['label'],'Wrong label'
        assert len(memory)==row['load'],'Wrong load'
        queries.append(sorted(memory));labels.append([memory[k] for k in sorted(memory)])
    return torch.tensor(queries),torch.tensor(labels)


def training_queries(rows,indices,slots,arm):
    assert arm in ARMS
    if arm=='fixed':return torch.tensor([rows[i]['query'] for i in indices.tolist()]),torch.tensor([rows[i]['label'] for i in indices.tolist()])
    queries,labels=query_tables(rows)
    assert int(slots.min())>=0 and int(slots.max())<queries.shape[1],'Invalid query slot'
    return queries[indices,slots],labels[indices,slots]


def expanded_rows(rows):
    q,y=query_tables(rows)
    return [dict(row,query=int(q[i,j]),label=int(y[i,j])) for i,row in enumerate(rows) for j in range(q.shape[1])]


def evaluate_all(model,rows):
    q,_=query_tables(rows);e,_,_=study.parent.tensors(rows);before=study.parent.state_hash(model.state_dict())
    with torch.no_grad():
        batches=[]
        for events,queries in zip(e.split(128),q.split(128)):
            state=model.prefill(events)
            batches.append(torch.stack([model.read(state,queries[:,j]) for j in range(queries.shape[1])],1))
        scores=torch.cat(batches)
    assert before==study.parent.state_hash(model.state_dict())
    labels=torch.tensor([r['label'] for r in expanded_rows(rows)]).reshape(scores.shape[:2])
    correct=(scores.argmax(-1)==labels).all(1)
    return dict(all_queries=study.parent.metrics(scores.reshape(-1,6),expanded_rows(rows))['all'],
                complete_maps=dict(count=len(rows),correct=int(correct.sum()),accuracy=float(correct.double().mean()))),scores


def audit_cell(checkpoint,rows,all_queries,metrics,scores):
    model=previous.geometry.prior.load(checkpoint,'delta')
    if all_queries:
        # Independently rebuild label maps and enumerate keys in sorted producer order.
        rebuilt=[]
        for row in rows:
            memory={k:v for k,v in row['events']}
            assert memory[row['query']]==row['label'] and len(memory)==row['load']
            rebuilt.extend(dict(row,query=q,label=memory[q]) for q in sorted(memory))
        e,q,_=study.parent.tensors(rebuilt)
    else:e,q,_=study.parent.tensors(rows);rebuilt=rows
    with torch.no_grad():ref=torch.cat([study.parent.reference(model,a,b) for a,b in zip(e.split(128),q.split(128))])
    error=previous.geometry.prior.check_scores(scores.reshape(-1,6),ref)
    scalar=previous.geometry.prior.reduce_scores(ref,rebuilt)['all']
    previous.geometry.prior.check_metrics({'all':metrics['all_queries'] if all_queries else metrics},{'all':scalar})
    if all_queries:
        width=rows[0]['load'];pred=ref.tolist();correct=0
        for i in range(len(rows)):
            correct+=all(max(range(6),key=lambda k:pred[i*width+j][k])==rebuilt[i*width+j]['label'] for j in range(width))
        assert metrics['complete_maps']==dict(count=len(rows),correct=correct,accuracy=correct/len(rows)),'Wrong complete-map count'
    return error


def development_gate(runs):
    expected=set(itertools.product(DEV_SEEDS,ARMS));keys=[(r['seed'],r['arm']) for r in runs]
    assert len(keys)==len(expected) and set(keys)==expected,'Incomplete development'
    fixed=all(r['metrics'][f'train_{m}']['accuracy']>=.99 for r in runs if r['arm']=='fixed' for m in study.LOADS)
    sampled=all(r['metrics'][f'all_train_{m}']['all_queries']['accuracy']>=.99 for r in runs if r['arm']=='resampled' for m in study.LOADS)
    return dict(fixed_assigned_training_fit=fixed,resampled_all_query_training_fit=sampled,confirmation_eligible=fixed and sampled)


def decision(runs):
    if not runs:return dict(confirmation_executed=False)
    lookup={(r['seed'],r['arm']):r for r in runs}
    assert len(runs)==6 and set(lookup)==set(itertools.product(CONFIRM_SEEDS,ARMS))
    fit=all(r['metrics'][f'train_{m}']['accuracy']>=.99 for r in runs if r['arm']=='fixed' for m in study.LOADS) and all(r['metrics'][f'all_train_{m}']['all_queries']['accuracy']>=.99 for r in runs if r['arm']=='resampled' for m in study.LOADS)
    gain=all(lookup[s,'resampled']['metrics']['all_test_32']['all_queries']['accuracy']-lookup[s,'fixed']['metrics']['all_test_32']['all_queries']['accuracy']>=.10 for s in CONFIRM_SEEDS)
    preserve=all(lookup[s,'resampled']['metrics'][f'all_test_{m}']['all_queries']['accuracy']+.02>=lookup[s,'fixed']['metrics'][f'all_test_{m}']['all_queries']['accuracy'] for s in CONFIRM_SEEDS for m in (2,8))
    reliability=all(lookup[s,'resampled']['metrics']['all_test_32']['complete_maps']['accuracy']>=.95 for s in CONFIRM_SEEDS)
    return dict(confirmation_executed=True,objective_training_fit=fit,all_seed_10_point_gain=gain,low_load_preservation=preserve,
                all_seed_complete_map_reliability=reliability,benefit=fit and gain and preserve,retention_supported=fit and gain and preserve and reliability)


def profiles():
    records=[]
    rows=[dict(events=[[k,k%6] for k in range(32)],query=7,label=1,load=32,group='unique') for _ in range(BATCH)]
    for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
        for arm in ARMS:
            model=study.pair(2026100443)['delta'].to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
            e,_,_=study.parent.tensors(rows);q,y=training_queries(rows,torch.arange(BATCH),torch.arange(BATCH)%32,arm)
            e,q,y=e.to(device),q.to(device),y.to(device)
            if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(e,q),y);loss.backward();optimizer.step()
            assert all(int(s['step'])==1 for s in optimizer.state.values())
            if device=='cuda':torch.cuda.synchronize()
            records.append(dict(device=device,arm=arm,batch=BATCH,events=32,dtype='float64',loss=float(loss.detach()),
                optimizer_state_allocated=True,peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None))
            del model,optimizer,loss,e,q,y
    return records


def train_run(phase,seed,arm,payload,report,saved):
    model=study.pair(seed)['delta'];optimizer=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
    data={m:study.parent.tensors(payload['pools'][f'train_{m}']) for m in study.LOADS}
    tables={m:query_tables(payload['pools'][f'train_{m}']) for m in study.LOADS}
    exposure={m:torch.zeros_like(tables[m][0]) for m in study.LOADS}
    batch_rng=torch.Generator().manual_seed(seed+10000);query_rng=torch.Generator().manual_seed(seed+20000)
    history_hash,query_hash,draw_hash=hashlib.sha256(),hashlib.sha256(),hashlib.sha256()
    item=dict(phase=phase,seed=seed,arm=arm,status='running',metrics={},curve=[],parameters=sum(p.numel() for p in model.parameters()),initial_sha256=study.parent.state_hash(model.state_dict()))
    report['runs'].append(item);study.parent.write(OUT/'report.json',report);tick=time.perf_counter()
    for step in range(1,STEPS+1):
        load=study.LOADS[(step-1)%4];e,fixed_q,fixed_y=data[load];queries,labels=tables[load]
        idx=torch.randint(len(e),(BATCH,),generator=batch_rng);slots=torch.randint(load,(BATCH,),generator=query_rng)
        q,y=(fixed_q[idx],fixed_y[idx]) if arm=='fixed' else (queries[idx,slots],labels[idx,slots])
        actual_slots=(queries[idx]==q[:,None]).long().argmax(1)
        exposure[load].index_put_((idx,actual_slots),torch.ones(BATCH,dtype=torch.long),accumulate=True)
        history_hash.update(struct.pack('<I',load));history_hash.update(bytes(idx.view(torch.uint8).tolist()))
        draw_hash.update(bytes(slots.view(torch.uint8).tolist()));query_hash.update(bytes(q.view(torch.uint8).tolist()));query_hash.update(bytes(y.view(torch.uint8).tolist()))
        optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(e[idx],q),y);assert torch.isfinite(loss)
        loss.backward();optimizer.step()
        if step%1000==0:item['curve'].append(dict(step=step,loss=float(loss.detach())));print(dict(phase=phase,seed=seed,arm=arm,step=step),flush=True)
    cp=OUT/f'{phase}_{seed}_{arm}.pt';assert not cp.exists()
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=STEPS),cp)
    for pool,rows in payload['pools'].items():
        if pool.startswith('test_'):continue
        item['metrics'][pool],saved[f'{phase}_{seed}_{arm}_{pool}']=previous.metrics(model,rows)
        item['metrics']['all_'+pool],saved[f'{phase}_{seed}_{arm}_all_{pool}']=evaluate_all(model,rows)
    torch.save(exposure,OUT/f'{phase}_{seed}_{arm}_exposure.pt')
    item.update(status='complete',updates=STEPS,examples=STEPS*BATCH,input_events=STEPS*BATCH*32,examples_per_load=STEPS*BATCH//4,
        history_stream_sha256=history_hash.hexdigest(),potential_query_draws_sha256=draw_hash.hexdigest(),actual_queries_labels_sha256=query_hash.hexdigest(),
        exposure={str(m):dict(targets=int(x.sum()),covered_history_key_pairs=int((x>0).sum()),total_history_key_pairs=x.numel(),min_targets_per_pair=int(x.min()),max_targets_per_pair=int(x.max())) for m,x in exposure.items()},
        exposure_sha256=study.parent.digest(OUT/f'{phase}_{seed}_{arm}_exposure.pt'),checkpoint=cp.relative_to(ROOT).as_posix(),checkpoint_sha256=study.parent.digest(cp),
        elapsed_seconds=time.perf_counter()-tick,beta=float(model.write_logit.sigmoid().detach()),key_overlap=previous.independent_penalty(model))
    study.parent.write(OUT/'report.json',report)


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    previous.geometry.prior.parent_evidence()
    old=json.loads((previous.OUT/'report.json').read_text());audit=json.loads((previous.OUT/'independent_audit.json').read_text())
    assert audit['passed'] and audit['report_sha256']==study.parent.digest(previous.OUT/'report.json') and old['source_dependencies']==previous.dependencies()
    assert not OUT.exists();OUT.mkdir();source=dependencies()
    protected=json.loads((previous.OUT/'preserved.json').read_text())
    for p in previous.OUT.rglob('*'):
        if p.is_file():protected[p.relative_to(ROOT).as_posix()]=study.parent.digest(p)
    for p in ('docs/topics/attention/overlap-regularization-results.md','docs/topics/attention/overlap-regularization-guide.md'):protected[p]=study.parent.digest(ROOT/p)
    assert all(study.parent.digest(ROOT/p)==h for p,h in protected.items());study.parent.write(OUT/'preserved.json',protected)
    spec=importlib.util.spec_from_file_location('test_delta_query_coverage',ROOT/'tests/test_delta_query_coverage.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    study.parent.write(OUT/'preflight.json',dict(passed=result.wasSuccessful(),methods=result.testsRun,source_dependencies=source));assert result.wasSuccessful(),'Preflight failed; no formal training'
    study.parent.write(OUT/'profiles.json',profiles())
    excluded=set()
    for folder in (study.OUT,previous.OUT):
        for phase in ('development','confirmation'):
            path=folder/f'{phase}_data.json'
            if path.exists():excluded.update(study.family(row['events']) for rows in json.loads(path.read_text())['pools'].values() for row in rows)
    dev=study.generate(2026100444,exclude=excluded);study.parent.write(OUT/'development_data.json',dev)
    assert all(study.family(row['events']) not in excluded for rows in dev['pools'].values() for row in rows)
    study.parent.write(OUT/'lock.json',dict(source_dependencies=source,steps=STEPS,batch=BATCH,seeds=DEV_SEEDS,confirmation_seeds=CONFIRM_SEEDS,
        preflight_sha256=study.parent.digest(OUT/'preflight.json'),profiles_sha256=study.parent.digest(OUT/'profiles.json')))
    report=dict(status='development',source_dependencies=source,device='cpu',dtype='float64',threads=1,runs=[],baselines=dict(development=study.audit_data(dev)),data_sha256=dict(development=study.parent.digest(OUT/'development_data.json')));saved={}
    for seed in DEV_SEEDS:
        for arm in ARMS if seed==DEV_SEEDS[0] else ARMS[::-1]:train_run('development',seed,arm,dev,report,saved)
    report['development_gate']=development_gate(report['runs']);study.parent.write(OUT/'development_locked.json',report)
    if report['development_gate']['confirmation_eligible']:
        excluded.update(study.family(row['events']) for rows in dev['pools'].values() for row in rows)
        fresh=study.generate(2026100445,confirmation=True,exclude=excluded);assert all(study.family(row['events']) not in excluded for rows in fresh['pools'].values() for row in rows)
        study.parent.write(OUT/'confirmation_data.json',fresh);report['data_sha256']['confirmation']=study.parent.digest(OUT/'confirmation_data.json');report['baselines']['confirmation']=study.audit_data(fresh)
        for seed in CONFIRM_SEEDS:
            for arm in ARMS if seed!=CONFIRM_SEEDS[1] else ARMS[::-1]:train_run('confirmation',seed,arm,fresh,report,saved)
        study.parent.write(OUT/'pretest.json',report)
        for item in [r for r in report['runs'] if r['phase']=='confirmation']:
            model=previous.geometry.prior.load(ROOT/item['checkpoint'],'delta')
            for pool,rows in fresh['pools'].items():
                if not pool.startswith('test_'):continue
                item['metrics'][pool],saved[f"confirmation_{item['seed']}_{item['arm']}_{pool}"]=previous.metrics(model,rows)
                item['metrics']['all_'+pool],saved[f"confirmation_{item['seed']}_{item['arm']}_all_{pool}"]=evaluate_all(model,rows)
    else:report['confirmation_skipped']='Objective-specific training fit failed; stop after four development runs.'
    report['decision']=decision([r for r in report['runs'] if r['phase']=='confirmation'])
    assert dependencies()==source and all(study.parent.digest(ROOT/p)==h for p,h in protected.items())
    torch.save(saved,OUT/'scores.pt');report.update(status='complete',scores_sha256=study.parent.digest(OUT/'scores.pt'),protected_files_unchanged=len(protected));study.parent.write(OUT/'report.json',report)
    print('Query-coverage budget complete. Run audit next.',flush=True)


def audit():
    torch.set_num_threads(1)
    r=json.loads((OUT/'report.json').read_text());lock=json.loads((OUT/'lock.json').read_text());assert r['status']=='complete' and r['source_dependencies']==lock['source_dependencies']==dependencies()
    assert lock['steps']==STEPS and lock['batch']==BATCH and lock['seeds']==list(DEV_SEEDS) and lock['confirmation_seeds']==list(CONFIRM_SEEDS)
    assert lock['preflight_sha256']==study.parent.digest(OUT/'preflight.json') and lock['profiles_sha256']==study.parent.digest(OUT/'profiles.json')
    preflight=json.loads((OUT/'preflight.json').read_text());assert preflight['passed'] and preflight['source_dependencies']==dependencies()
    assert r['scores_sha256']==study.parent.digest(OUT/'scores.pt');saved=torch.load(OUT/'scores.pt',weights_only=True)
    data={}
    for phase,h in r['data_sha256'].items():
        assert study.parent.digest(OUT/f'{phase}_data.json')==h;data[phase]=json.loads((OUT/f'{phase}_data.json').read_text());assert study.audit_data(data[phase])==r['baselines'][phase]
    study.audit_data(dict(pools={f'{phase}_{name}':rows for phase,p in data.items() for name,rows in p['pools'].items()}))
    oldfamilies=set()
    for folder in (study.OUT,previous.OUT):
        for phase in ('development','confirmation'):
            path=folder/f'{phase}_data.json'
            if path.exists():oldfamilies.update(study.family(row['events']) for rows in json.loads(path.read_text())['pools'].values() for row in rows)
    assert all(study.family(row['events']) not in oldfamilies for p in data.values() for rows in p['pools'].values() for row in rows)
    used,errors,records,predictions=set(),[],[],0
    for item in r['runs']:
        assert item['status']=='complete' and item['updates']==STEPS and item['parameters']==615 and item['examples']==STEPS*BATCH
        assert item['input_events']==STEPS*BATCH*32 and item['examples_per_load']==STEPS*BATCH//len(study.LOADS)
        cp=ROOT/item['checkpoint'];assert study.parent.digest(cp)==item['checkpoint_sha256'];checkpoint=torch.load(cp,weights_only=True)
        assert checkpoint['step']==STEPS and all(int(s['step'])==STEPS for s in checkpoint['optimizer']['state'].values())
        model=previous.geometry.prior.load(cp,'delta');assert item['initial_sha256']==study.parent.state_hash(study.pair(item['seed'])['delta'].state_dict())
        assert abs(item['key_overlap']-previous.independent_penalty(model))<1e-12 and item['beta']==float(model.write_logit.sigmoid().detach())
        exposure_path=OUT/f"{item['phase']}_{item['seed']}_{item['arm']}_exposure.pt";assert study.parent.digest(exposure_path)==item['exposure_sha256'];exposure=torch.load(exposure_path,weights_only=True)
        counts={m:torch.zeros_like(exposure[m]) for m in study.LOADS};g=torch.Generator().manual_seed(item['seed']+10000);qg=torch.Generator().manual_seed(item['seed']+20000)
        history_hash,query_hash,draw_hash=hashlib.sha256(),hashlib.sha256(),hashlib.sha256()
        pools={m:data[item['phase']]['pools'][f'train_{m}'] for m in study.LOADS}
        # Independent label maps and explicit scalar exposure increments, not training table/index_put.
        maps={m:[dict(row['events']) for row in rows] for m,rows in pools.items()}
        for step in range(STEPS):
            m=study.LOADS[step%4];idx=torch.randint(len(pools[m]),(BATCH,),generator=g);slots=torch.randint(m,(BATCH,),generator=qg)
            qs,ys=[],[]
            for i,slot in zip(idx.tolist(),slots.tolist()):
                row=pools[m][i];keys=sorted(maps[m][i]);q=row['query'] if item['arm']=='fixed' else keys[slot];qs.append(q);ys.append(maps[m][i][q]);counts[m][i,keys.index(q)]+=1
            history_hash.update(struct.pack('<I',m));history_hash.update(bytes(idx.view(torch.uint8).tolist()));draw_hash.update(bytes(slots.view(torch.uint8).tolist()))
            query_hash.update(bytes(torch.tensor(qs).view(torch.uint8).tolist()));query_hash.update(bytes(torch.tensor(ys).view(torch.uint8).tolist()))
        assert history_hash.hexdigest()==item['history_stream_sha256'] and draw_hash.hexdigest()==item['potential_query_draws_sha256'] and query_hash.hexdigest()==item['actual_queries_labels_sha256']
        for m in study.LOADS:
            assert torch.equal(exposure[m],counts[m])
            x=counts[m];assert item['exposure'][str(m)]==dict(targets=int(x.sum()),covered_history_key_pairs=int((x>0).sum()),total_history_key_pairs=x.numel(),min_targets_per_pair=int(x.min()),max_targets_per_pair=int(x.max()))
        for name,value in item['metrics'].items():
            allq=name.startswith('all_');pool=name[4:] if allq else name;rows=data[item['phase']]['pools'][pool];key=f"{item['phase']}_{item['seed']}_{item['arm']}_{name}"
            scores=saved[key];used.add(key);errors.append(audit_cell(cp,rows,allq,value,scores));predictions+=scores.numel()//6
            metrics=value['all_queries'] if allq else value
            records.append(dict(phase=item['phase'],seed=item['seed'],arm=item['arm'],pool=name,**metrics,complete_map_accuracy=value['complete_maps']['accuracy'] if allq else ''))
    assert used==set(saved);dev=[x for x in r['runs'] if x['phase']=='development'];conf=[x for x in r['runs'] if x['phase']=='confirmation']
    assert len(dev)==4 and development_gate(dev)==r['development_gate'];assert len(conf)==(6 if r['development_gate']['confirmation_eligible'] else 0) and decision(conf)==r['decision']
    assert len(records)==(208 if conf else 64) and predictions==(738048 if conf else 238080)
    for phase,seeds in (('development',DEV_SEEDS),('confirmation',CONFIRM_SEEDS)):
        for seed in seeds:
            pairs=[x for x in r['runs'] if x['phase']==phase and x['seed']==seed]
            if pairs:assert len(pairs)==2 and all(len({x[k] for x in pairs})==1 for k in ('initial_sha256','history_stream_sha256','potential_query_draws_sha256'))
    protected=json.loads((OUT/'preserved.json').read_text());assert all(study.parent.digest(ROOT/p)==h for p,h in protected.items())
    assert not (OUT/'independent_audit.json').exists()
    with (OUT/'metrics.csv').open('x',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    study.parent.write(OUT/'independent_audit.json',dict(passed=True,report_sha256=study.parent.digest(OUT/'report.json'),source_dependencies=dependencies(),runs=len(r['runs']),metric_cells=len(records),predictions=predictions,max_logit_error=max(errors),argmax_disagreements=0,protected_files_unchanged=len(protected),initializations_history_query_label_streams_and_exposures_verified=True,scope='Independent matrix equations and Python reductions, full query/label/exposure stream reconstruction; gate replay shares policy functions with tested inverse controls. Shared data/weights/PyTorch.'))
    print('Delta query-coverage audit passes.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('run','audit'));args=p.parse_args();run() if args.phase=='run' else audit()
