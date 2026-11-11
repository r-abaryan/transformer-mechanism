"""Matched exhaustive versus replacement-sampled multi-query supervision."""
import argparse
import csv
import importlib.util
import json
import struct
import hashlib
import time
import unittest

import torch
from torch.nn import functional as F
import query_gradient_variability as previous

replay=previous.replay
parent=replay.parent
model_source=replay.pressure.model_source
ROOT=previous.ROOT
OUT=ROOT/'results/query_feedback_training'
ARMS=('exhaustive','sampled_m')
SEEDS=(3461,3463)
LOADS=(2,8,16,32)
STEPS,BATCH=3000,64
PROBES=(0,1000,2000,3000)
SOURCE_NAMES=('experiments/query_feedback_training.py','experiments/audit_query_feedback_training.py',
              'tests/test_query_feedback_training.py')


def hash_ints(hasher,values):
    hasher.update(bytes(values.contiguous().view(torch.uint8).flatten().tolist()))


def choose(queries,labels,indices,draws,arm):
    assert arm in ARMS and draws.shape==(len(indices),queries.shape[1])
    assert int(draws.min())>=0 and int(draws.max())<queries.shape[1]
    slots=torch.arange(queries.shape[1])[None,:].expand_as(draws) if arm=='exhaustive' else draws
    return queries[indices].gather(1,slots),labels[indices].gather(1,slots),slots


def forward_queries(model,events,queries):
    state=model.prefill(events)
    return torch.stack([model.read(state,queries[:,j]) for j in range(queries.shape[1])],1)


def reduce_scores(logits,labels,repeat):
    return dict(groups=replay.pressure.reduce_logits(logits,labels,repeat),
                complete_maps=dict(count=len(labels),correct=int((logits.argmax(-1)==labels).all(1).sum()),
                    accuracy=float((logits.argmax(-1)==labels).all(1).double().mean())))


def evaluate(model,rows,batch=64):
    events,queries,labels,repeat=replay.pressure.prepare(rows,'all_queries')
    before=parent.study.parent.state_hash(model.state_dict())
    rng=torch.get_rng_state().clone()
    with torch.no_grad():
        logits=torch.cat([forward_queries(model,events[i:i+batch],queries[i:i+batch])
                          for i in range(0,len(rows),batch)])
    assert before==parent.study.parent.state_hash(model.state_dict()) and torch.equal(rng,torch.get_rng_state())
    return dict(logits=logits,labels=labels,repeat=repeat,metrics=reduce_scores(logits,labels,repeat))


def record_evaluation(model,folder,seed,arm,step,pool,rows,report,saved,checkpoint):
    cell=evaluate(model,rows)
    metadata=dict(seed=seed,arm=arm,step=step,pool=pool,checkpoint=checkpoint)
    saved.append(dict(**metadata,**cell))
    report['cells'].append(dict(**metadata,metrics=cell['metrics']))
    return cell['metrics']


def train_run(folder,seed,arm,payload,report,saved,steps=STEPS,probes=PROBES):
    model=model_source.pair(seed)['delta']
    optimizer=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
    tensors={m:parent.study.parent.tensors(payload['pools'][f'train_{m}']) for m in LOADS}
    tables={m:parent.query_tables(payload['pools'][f'train_{m}']) for m in LOADS}
    exposure={m:torch.zeros_like(tables[m][0]) for m in LOADS}
    history_rng=torch.Generator().manual_seed(seed+10000)
    query_rng=torch.Generator().manual_seed(seed+20000)
    history_hash,draw_hash,target_hash=(hashlib.sha256() for _ in range(3))
    item=dict(seed=seed,arm=arm,status='running',initial_sha256=parent.study.parent.state_hash(model.state_dict()),
              parameters=sum(p.numel() for p in model.parameters()),snapshots=[],curve=[],steps=[])
    report['runs'].append(item)

    def snapshot(step):
        cp=folder/f'{seed}_{arm}_{step}.pt';assert not cp.exists()
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step),cp)
        record=dict(step=step,checkpoint=cp.relative_to(ROOT).as_posix(),sha256=replay.digest(cp),
                    beta=float(model.write_logit.sigmoid().detach()),training={})
        for m in LOADS:
            pool=f'train_{m}'
            record['training'][str(m)]=record_evaluation(model,folder,seed,arm,step,pool,payload['pools'][pool],
                                                       report,saved,record['checkpoint'])
        item['snapshots'].append(record)
        replay.write(folder/'report.json',report)
        print(json.dumps(dict(seed=seed,arm=arm,step=step,beta=record['beta'])),flush=True)

    started=time.perf_counter()
    snapshot(0)
    target_count=0
    for step in range(1,steps+1):
        m=LOADS[(step-1)%4];events,_,_=tensors[m];queries,labels=tables[m]
        indices=torch.randint(len(events),(BATCH,),generator=history_rng)
        draws=torch.randint(m,(BATCH,m),generator=query_rng)
        q,y,slots=choose(queries,labels,indices,draws,arm)
        exposure[m].index_put_((indices[:,None].expand_as(slots).flatten(),slots.flatten()),
                              torch.ones(slots.numel(),dtype=torch.long),accumulate=True)
        history_hash.update(struct.pack('<I',m));hash_ints(history_hash,indices)
        hash_ints(draw_hash,draws);hash_ints(target_hash,q);hash_ints(target_hash,y)
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(forward_queries(model,events[indices],q).reshape(-1,6),y.flatten())
        assert torch.isfinite(loss)
        loss.backward()
        p=model.write_logit;alpha=float(p.detach());beta=float(p.sigmoid().detach());g=float(p.grad)
        optimizer.step();state=optimizer.state[p]
        item['steps'].append(dict(step=step,load=m,alpha_before=alpha,alpha_after=float(p.detach()),
            beta_before=beta,beta_after=float(p.sigmoid().detach()),gradient=g,
            exp_avg=float(state['exp_avg']),exp_avg_sq=float(state['exp_avg_sq'])))
        target_count+=y.numel()
        if step in probes:
            item['curve'].append(dict(step=step,load=m,minibatch_loss=float(loss.detach())))
            snapshot(step)
    assert steps in probes
    final=item['snapshots'][-1]['checkpoint']
    for pool,rows in payload['pools'].items():
        if pool.startswith('validation_'):
            record_evaluation(model,folder,seed,arm,steps,pool,rows,report,saved,final)
    exposure_path=folder/f'{seed}_{arm}_exposure.pt';torch.save(exposure,exposure_path)
    item.update(status='complete',updates=steps,histories=steps*BATCH,input_events=steps*BATCH*32,
        answer_targets=target_count,history_stream_sha256=history_hash.hexdigest(),potential_draws_sha256=draw_hash.hexdigest(),
        actual_queries_labels_sha256=target_hash.hexdigest(),exposure_path=exposure_path.relative_to(ROOT).as_posix(),
        exposure_sha256=replay.digest(exposure_path),exposure={str(m):dict(targets=int(x.sum()),covered=int((x>0).sum()),
            total=x.numel(),min_targets=int(x.min()),max_targets=int(x.max())) for m,x in exposure.items()},
        elapsed_seconds=time.perf_counter()-started,final_checkpoint=final,beta=float(model.write_logit.sigmoid().detach()))
    replay.write(folder/'report.json',report)


def decision(cells):
    final={(c['seed'],c['arm'],c['pool']):c['metrics'] for c in cells if c['step']==STEPS}
    assert all((s,a,f'train_{m}') in final for s in SEEDS for a in ARMS for m in LOADS)
    pairs=[]
    for seed in SEEDS:
        x,y=(final[seed,a,'train_32']['groups']['all'] for a in ARMS)
        preserve=all(final[seed,'exhaustive',f'train_{m}']['groups']['all']['accuracy']+.01>=
                     final[seed,'sampled_m',f'train_{m}']['groups']['all']['accuracy'] for m in (2,8,16))
        pairs.append(dict(seed=seed,train32_accuracy_gain=x['accuracy']-y['accuracy'],
                          train32_ce_ratio=x['ce']/y['ce'] if y['ce']>1e-12 else None,
                          preserves_lower_loads=preserve,
                          benefit=x['accuracy']-y['accuracy']>=.01 and x['ce']<=.95*y['ce'] and preserve))
    fit={a:all(final[s,a,f'train_{m}']['groups']['all']['accuracy']>=.99 for s in SEEDS for m in LOADS) for a in ARMS}
    return dict(all_seed_training_fit=fit,pairs=pairs,both_seed_fitting_benefit=all(p['benefit'] for p in pairs))


def profiles():
    rows=[dict(events=[[k,k%6] for k in range(32)],query=7,label=1,load=32,group='unique') for _ in range(BATCH)]
    events,_,_=parent.study.parent.tensors(rows);queries,labels=parent.query_tables(rows)
    records=[]
    for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
        for arm in ARMS:
            model=model_source.pair(2026100460)['delta'].to(device)
            q,y,_=choose(queries,labels,torch.arange(BATCH),torch.arange(BATCH*32).reshape(BATCH,32)%32,arm)
            e,q,y=events.to(device),q.to(device),y.to(device)
            opt=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
            if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter();opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(forward_queries(model,e,q).reshape(-1,6),y.flatten());loss.backward();opt.step()
            if device=='cuda':torch.cuda.synchronize()
            assert all(int(s['step'])==1 for s in opt.state.values())
            records.append(dict(device=device,arm=arm,histories=64,events=32,answer_targets=2048,dtype='float64',
                complete_step_seconds=time.perf_counter()-started,optimizer_state_allocated=True,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None))
            del model,opt,loss,e,q,y
    return records


def protected_files():
    files=dict(json.loads((previous.OUT/'lock.json').read_text())['files'])
    for name in ('report.json','audit.json','lock.json','metrics.csv','targets.pt'):
        p=previous.OUT/name;files[p.relative_to(ROOT).as_posix()]=replay.digest(p)
    return files


def preflight():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    spec=importlib.util.spec_from_file_location('test_query_feedback_training',ROOT/SOURCE_NAMES[-1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    record=dict(passed=result.wasSuccessful(),methods=result.testsRun,sources={n:replay.digest(ROOT/n) for n in SOURCE_NAMES})
    replay.write(ROOT/'results/query_feedback_training_preflight.json',record);assert record['passed']
    replay.write(ROOT/'results/query_feedback_training_profiles.json',dict(records=profiles(),sources=record['sources']))


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert not OUT.exists(),'Preserve completed and failed attempts'
    ready_path=ROOT/'results/query_feedback_training_preflight.json'
    profile_path=ROOT/'results/query_feedback_training_profiles.json'
    ready=json.loads(ready_path.read_text());assert ready['passed']
    hashes={n:replay.digest(ROOT/n) for n in SOURCE_NAMES};assert ready['sources']==hashes
    assert json.loads(profile_path.read_text())['sources']==hashes
    files=protected_files()
    for name in (*SOURCE_NAMES,'docs/protocols/query-feedback-training.md',ready_path.relative_to(ROOT).as_posix(),profile_path.relative_to(ROOT).as_posix()):
        files[name]=replay.digest(ROOT/name)
    replay.check_files(files)
    old=json.loads((parent.OUT/'development_data.json').read_text())
    excluded={model_source.family(r['events']) for rows in old['pools'].values() for r in rows}
    fresh=model_source.generate(2026100461,confirmation=True,exclude=excluded,
                               base_counts=dict(train=0,validation=0,test=64))
    fresh['pools']={n:rows for n,rows in fresh['pools'].items() if n.startswith('test_')}
    model_source.audit_data(fresh)
    assert all(model_source.family(r['events']) not in excluded for rows in fresh['pools'].values() for r in rows)
    OUT.mkdir();replay.write(OUT/'test_data.json',fresh)
    replay.write(OUT/'lock.json',dict(files=files,test_data_sha256=replay.digest(OUT/'test_data.json'),steps=STEPS,
                                   batch=BATCH,seeds=SEEDS,arms=ARMS,probes=PROBES))
    report=dict(status='running',device='cpu',dtype='float64',threads=1,runs=[],cells=[],development_seeds=SEEDS,
                test_reserved=True,test_is_exploratory=True,training_gpu_peak_allocated=None,training_gpu_peak_reserved=None)
    saved=[]
    for seed in SEEDS:
        for arm in ARMS if seed==SEEDS[0] else ARMS[::-1]:
            train_run(OUT,seed,arm,old,report,saved)
    report['decision']=decision(report['cells']);replay.write(OUT/'pretest_report.json',report)
    for item in report['runs']:
        model=previous.load_model(ROOT/item['final_checkpoint'])
        for pool,rows in fresh['pools'].items():
            record_evaluation(model,OUT,item['seed'],item['arm'],STEPS,pool,rows,report,saved,item['final_checkpoint'])
    torch.save(saved,OUT/'scores.pt')
    report.update(status='complete',total_updates=12000,total_histories=768000,total_input_events=24576000,
                  total_answer_targets=11136000,cells_count=len(saved),targets=sum(c['labels'].numel() for c in saved),
                  scores_sha256=replay.digest(OUT/'scores.pt'),lock_sha256=replay.digest(OUT/'lock.json'),
                  pretest_sha256=replay.digest(OUT/'pretest_report.json'))
    assert report['cells_count']==96 and report['targets']==846336
    replay.check_files(files);replay.write(OUT/'report.json',report)
    print(json.dumps(report['decision']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    if parser.parse_args().preflight:preflight()
    else:run()
