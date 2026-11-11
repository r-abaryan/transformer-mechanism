"""Bounded training-only continuation of the saved exhaustive-feedback recipe."""
import argparse
import csv
import hashlib
import importlib.util
import json
import struct
import time
import unittest

import torch
from torch.nn import functional as F
import query_feedback_training as parent

ROOT=parent.ROOT
OUT=ROOT/'results/query_feedback_continuation'
SEEDS=parent.SEEDS
START,CAP,INTERVAL,FIT=3000,12000,1000,.99
SOURCES=('experiments/query_feedback_continuation.py','experiments/audit_query_feedback_continuation.py',
         'tests/test_query_feedback_continuation.py')
PROTOCOL='docs/protocols/query-feedback-continuation.md'


def restore(path,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=parent.previous.load_model(path).to(device).train()
    optimizer=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    group=optimizer.param_groups[0]
    assert (group['lr'],group['betas'],group['eps'],group['weight_decay'],group['foreach'])==(.003,(.9,.999),1e-8,0,False)
    assert all(int(s['step'])==checkpoint['step'] for s in optimizer.state.values())
    if device=='cpu':
        assert parent.replay.equal(model.state_dict(),checkpoint['model'])
        assert parent.replay.equal(optimizer.state_dict(),checkpoint['optimizer'])
    return model,optimizer,checkpoint['step']


def streams(seed,payload,start):
    """Reconstruct discarded RNG states and hashes without updating weights."""
    history=torch.Generator().manual_seed(seed+10000)
    query=torch.Generator().manual_seed(seed+20000)
    hashes=[hashlib.sha256() for _ in range(3)]
    tables={m:parent.parent.query_tables(payload['pools'][f'train_{m}']) for m in parent.LOADS}
    exposure={m:torch.zeros_like(q) for m,(q,y) in tables.items()}
    for step in range(1,start+1):
        m=parent.LOADS[(step-1)%4]
        indices=torch.randint(len(tables[m][0]),(parent.BATCH,),generator=history)
        draws=torch.randint(m,(parent.BATCH,m),generator=query)
        q,y,slots=parent.choose(*tables[m],indices,draws,'exhaustive')
        record_stream(hashes,exposure,m,indices,draws,q,y,slots)
    return history,query,hashes,tables,exposure


def record_stream(hashes,exposure,m,indices,draws,q,y,slots):
    hashes[0].update(struct.pack('<I',m));parent.hash_ints(hashes[0],indices)
    parent.hash_ints(hashes[1],draws)
    parent.hash_ints(hashes[2],q);parent.hash_ints(hashes[2],y)
    exposure[m].index_put_((indices[:,None].expand_as(slots).flatten(),slots.flatten()),
                          torch.ones(slots.numel(),dtype=torch.long),accumulate=True)


def training_fit(training,threshold=FIT):
    assert set(training)=={str(m) for m in parent.LOADS},'Missing training load'
    for m,record in training.items():
        groups=record['groups'];all_group=groups['all']
        assert all_group['count']>0 and all_group['accuracy'] is not None
        for name in ('unique','repeat'):
            expected='unique' if int(m)==32 else 'repeat'
            assert groups[name]['count']==(all_group['count'] if name==expected else 0),'Missing relevant group'
            if name==expected:assert groups[name]['accuracy']==all_group['accuracy']
    return all(record['groups']['all']['accuracy']>=threshold for record in training.values())


def continue_run(folder,seed,path,payload,report,saved,cap=CAP,interval=INTERVAL,threshold=FIT):
    model,opt,start=restore(path)
    assert cap>start and (cap-start)%interval==0
    history,query,hashes,tables,exposure=streams(seed,payload,start)
    events={m:parent.parent.study.parent.tensors(payload['pools'][f'train_{m}'])[0] for m in parent.LOADS}
    item=dict(seed=seed,arm='exhaustive',status='running',start=start,cap=cap,interval=interval,fit_threshold=threshold,
              parent_checkpoint=path.relative_to(ROOT).as_posix(),parent_sha256=parent.replay.digest(path),
              prefix_history_sha256=hashes[0].hexdigest(),prefix_draws_sha256=hashes[1].hexdigest(),
              prefix_targets_sha256=hashes[2].hexdigest(),parameters=sum(p.numel() for p in model.parameters()),
              snapshots=[],steps=[])
    report['runs'].append(item)

    def snapshot(step):
        cp=folder/f'{seed}_exhaustive_{step}.pt';assert not cp.exists()
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,
                        history_rng=history.get_state(),query_rng=query.get_state()),cp)
        record=dict(step=step,checkpoint=cp.relative_to(ROOT).as_posix(),sha256=parent.replay.digest(cp),
                    beta=float(model.write_logit.sigmoid().detach()),training={})
        for m in parent.LOADS:
            pool=f'train_{m}'
            record['training'][str(m)]=parent.record_evaluation(model,folder,seed,'exhaustive',step,pool,
                payload['pools'][pool],report,saved,record['checkpoint'])
        record['fit']=training_fit(record['training'],threshold)
        item['snapshots'].append(record);parent.replay.write(folder/'report.json',report)
        print(json.dumps(dict(seed=seed,step=step,train32=record['training']['32']['groups']['all'],fit=record['fit'])),flush=True)
        return record['fit']

    started=time.perf_counter()
    fitted=snapshot(start)
    assert not fitted,'Already fitted: no continuation justified'
    target_count=0
    for step in range(start+1,cap+1):
        m=parent.LOADS[(step-1)%4]
        indices=torch.randint(len(events[m]),(parent.BATCH,),generator=history)
        draws=torch.randint(m,(parent.BATCH,m),generator=query)
        q,y,slots=parent.choose(*tables[m],indices,draws,'exhaustive')
        record_stream(hashes,exposure,m,indices,draws,q,y,slots)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(parent.forward_queries(model,events[m][indices],q).reshape(-1,6),y.flatten())
        assert torch.isfinite(loss)
        loss.backward();p=model.write_logit
        alpha,beta,g=float(p.detach()),float(p.sigmoid().detach()),float(p.grad)
        opt.step();state=opt.state[p]
        item['steps'].append(dict(step=step,load=m,alpha_before=alpha,alpha_after=float(p.detach()),
            beta_before=beta,beta_after=float(p.sigmoid().detach()),gradient=g,
            exp_avg=float(state['exp_avg']),exp_avg_sq=float(state['exp_avg_sq'])))
        target_count+=y.numel()
        if (step-start)%interval==0:
            fitted=snapshot(step)
            if fitted:break
    exposure_path=folder/f'{seed}_exposure.pt';assert not exposure_path.exists();torch.save(exposure,exposure_path)
    item.update(status='complete',stop=step,stop_reason='training_fit' if fitted else 'budget_cap',training_fit=fitted,
        added_updates=step-start,added_histories=(step-start)*parent.BATCH,added_input_events=(step-start)*parent.BATCH*32,
        added_answer_targets=target_count,history_stream_sha256=hashes[0].hexdigest(),potential_draws_sha256=hashes[1].hexdigest(),
        actual_queries_labels_sha256=hashes[2].hexdigest(),exposure_path=exposure_path.relative_to(ROOT).as_posix(),
        exposure_sha256=parent.replay.digest(exposure_path),exposure={str(m):dict(targets=int(x.sum()),covered=int((x>0).sum()),
            total=x.numel(),min_targets=int(x.min()),max_targets=int(x.max())) for m,x in exposure.items()},
        final_checkpoint=item['snapshots'][-1]['checkpoint'],beta=float(model.write_logit.sigmoid().detach()),
        elapsed_seconds=time.perf_counter()-started)
    parent.replay.write(folder/'report.json',report)
    return item


def profile(path,payload):
    e=parent.parent.study.parent.tensors(payload['pools']['train_32'][:64])[0]
    q,y=parent.parent.query_tables(payload['pools']['train_32'][:64]);records=[]
    for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
        model,opt,step=restore(path,device)
        events,queries,labels=e.to(device),q.to(device),y.to(device)
        if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter();opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(parent.forward_queries(model,events,queries).reshape(-1,6),labels.flatten())
        loss.backward();opt.step()
        if device=='cuda':torch.cuda.synchronize()
        assert all(int(s['step'])==step+1 for s in opt.state.values())
        records.append(dict(device=device,histories=64,events_per_history=32,answer_targets=2048,dtype='float64',
            optimizer_state_restored=True,complete_step_seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None))
        del model,opt,loss,events,queries,labels
    return records


def preflight():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    spec=importlib.util.spec_from_file_location('test_query_feedback_continuation',ROOT/SOURCES[-1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    record=dict(passed=result.wasSuccessful(),methods=result.testsRun,sources={n:parent.replay.digest(ROOT/n) for n in SOURCES})
    path=ROOT/'results/query_feedback_continuation_preflight.json'
    if not record['passed']:
        failures=list((ROOT/'results').glob('query_feedback_continuation_preflight_failed_*.json'))
        path=ROOT/f'results/query_feedback_continuation_preflight_failed_{len(failures)+1}.json'
    assert not path.exists();parent.replay.write(path,record);assert record['passed']
    old=json.loads((parent.OUT/'report.json').read_text());data=json.loads((parent.parent.OUT/'development_data.json').read_text())
    cp=next(r['final_checkpoint'] for r in old['runs'] if r['seed']==SEEDS[0] and r['arm']=='exhaustive')
    path=ROOT/'results/query_feedback_continuation_profiles.json';assert not path.exists()
    parent.replay.write(path,dict(records=profile(ROOT/cp,data),sources=record['sources']))


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert not OUT.exists(),'Preserve every completed or failed attempt'
    ready_path=ROOT/'results/query_feedback_continuation_preflight.json'
    profile_path=ROOT/'results/query_feedback_continuation_profiles.json'
    hashes={n:parent.replay.digest(ROOT/n) for n in SOURCES}
    ready=json.loads(ready_path.read_text());assert ready['passed'] and ready['sources']==hashes
    assert json.loads(profile_path.read_text())['sources']==hashes
    old=json.loads((parent.OUT/'report.json').read_text());old_audit=json.loads((parent.OUT/'audit.json').read_text())
    assert old_audit['passed'] and old_audit['report_sha256']==parent.replay.digest(parent.OUT/'report.json')
    files=dict(json.loads((parent.OUT/'lock.json').read_text())['files'])
    for p in parent.OUT.iterdir():
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=parent.replay.digest(p)
    for name in (*SOURCES,PROTOCOL,ready_path.relative_to(ROOT).as_posix(),profile_path.relative_to(ROOT).as_posix()):
        files[name]=parent.replay.digest(ROOT/name)
    parent.replay.check_files(files)
    data=json.loads((parent.parent.OUT/'development_data.json').read_text())
    OUT.mkdir();parent.replay.write(OUT/'lock.json',dict(files=files,seeds=SEEDS,start=START,cap=CAP,
        interval=INTERVAL,fit_threshold=FIT,arm='exhaustive',selection='full_training_only',heldout='previously_opened'))
    report=dict(status='running',device='cpu',dtype='float64',threads=1,runs=[],cells=[],heldout_previously_opened=True,
        validation_test_for_selection=False,training_gpu_peak_allocated=None,training_gpu_peak_reserved=None)
    saved=[]
    for seed in SEEDS:
        prior=next(r for r in old['runs'] if r['seed']==seed and r['arm']=='exhaustive')
        item=continue_run(OUT,seed,ROOT/prior['final_checkpoint'],data,report,saved)
        assert item['start']==START
        for new,previous in [('prefix_history_sha256','history_stream_sha256'),('prefix_draws_sha256','potential_draws_sha256'),
                             ('prefix_targets_sha256','actual_queries_labels_sha256')]:assert item[new]==prior[previous]
    report['all_seed_training_fit']=all(r['training_fit'] for r in report['runs'])
    parent.replay.write(OUT/'training_decision.json',report)
    # Both runs' stopping decisions are saved before any new held-out evaluation.
    fresh=json.loads((parent.OUT/'test_data.json').read_text())
    for item in report['runs']:
        model=parent.previous.load_model(ROOT/item['final_checkpoint'])
        pools={**{k:v for k,v in data['pools'].items() if k.startswith('validation_')},**fresh['pools']}
        for pool,rows in pools.items():parent.record_evaluation(model,OUT,item['seed'],'exhaustive',item['stop'],pool,rows,
            report,saved,item['final_checkpoint'])
    torch.save(saved,OUT/'scores.pt')
    report.update(status='complete',cells_count=len(saved),targets=sum(c['labels'].numel() for c in saved),
        total_added_updates=sum(r['added_updates'] for r in report['runs']),
        total_added_histories=sum(r['added_histories'] for r in report['runs']),
        total_added_input_events=sum(r['added_input_events'] for r in report['runs']),
        total_added_answer_targets=sum(r['added_answer_targets'] for r in report['runs']),
        scores_sha256=parent.replay.digest(OUT/'scores.pt'),lock_sha256=parent.replay.digest(OUT/'lock.json'),
        training_decision_sha256=parent.replay.digest(OUT/'training_decision.json'))
    parent.replay.check_files(files);parent.replay.write(OUT/'report.json',report)
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','arm','step','pool','group','count','ce','accuracy',
            'complete_maps_count','complete_maps_correct','complete_maps_accuracy']);writer.writeheader()
        for c in report['cells']:
            for group,g in c['metrics']['groups'].items():
                writer.writerow(dict(seed=c['seed'],arm=c['arm'],step=c['step'],pool=c['pool'],group=group,**g,
                    **{'complete_maps_'+k:v for k,v in c['metrics']['complete_maps'].items()}))
    print(json.dumps(dict(status=report['status'],all_seed_training_fit=report['all_seed_training_fit'],
                         added_updates=report['total_added_updates'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    if parser.parse_args().preflight:preflight()
    else:run()
