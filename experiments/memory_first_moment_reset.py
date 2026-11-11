"""Matched four-update test of one-time memory exp_avg removal."""
import argparse
import copy
import csv
import importlib.util
import json
import time
import unittest

import torch
from torch.nn import functional as F
import adam_memory_error_transport as transport

training=transport.training
ROOT=transport.ROOT
OUT=ROOT/'results/memory_first_moment_reset'
SOURCES=('experiments/memory_first_moment_reset.py','experiments/audit_memory_first_moment_reset.py',
         'tests/test_memory_first_moment_reset.py')
PROTOCOL='docs/protocols/memory-first-moment-reset.md'


def reset_first(optimizer,names):
    changed=[]
    for name,pid in zip(names,optimizer['param_groups'][0]['params']):
        if name in transport.MEMORY:optimizer['state'][pid]['exp_avg'].zero_();changed.append(name)
    assert set(changed)==set(transport.MEMORY)


def start_payload(trace):
    return dict(model=copy.deepcopy(trace['before_model']),optimizer=copy.deepcopy(trace['before_optimizer']),
        step=trace['step']-1,history_rng=trace['history_before'].clone(),query_rng=trace['query_before'].clone())


def train(path,pools,seed,arm,stop,folder,step_callback=None):
    with torch.random.fork_rng():
        model,opt,start=transport.replay_source.continuation.restore(path)
        saved=torch.load(path,weights_only=True);names=list(dict(model.named_parameters()))
        initial=copy.deepcopy(opt.state_dict())
        if arm=='reset':reset_first(initial,names);opt.load_state_dict(initial)
        else:assert arm=='control'
        hr=torch.Generator();hr.set_state(saved['history_rng']);qr=torch.Generator();qr.set_state(saved['query_rng'])
        traces=[];evaluations=[]
        for step in range(start+1,stop+1):
            load=(2,8,16,32)[(step-1)%4];rows=pools[f'train_{load}']
            events,_,_=training.parent.study.parent.tensors(rows);queries,labels=training.parent.query_tables(rows)
            hb=hr.get_state().clone();qb=qr.get_state().clone()
            indices=torch.randint(len(rows),(64,),generator=hr);draws=torch.randint(load,(64,load),generator=qr)
            q,y,_=training.choose(queries,labels,indices,draws,'exhaustive')
            _,_,_,repeat=training.replay.pressure.prepare([rows[int(i)] for i in indices],'all_queries')
            before=copy.deepcopy(model.state_dict());old=copy.deepcopy(opt.state_dict())
            began=time.perf_counter();opt.zero_grad(set_to_none=True)
            z=training.forward_queries(model,events[indices],q);loss=F.cross_entropy(z.reshape(-1,6),y.flatten());loss.backward()
            gradients={n:p.grad.detach().clone() for n,p in model.named_parameters()};opt.step();elapsed=time.perf_counter()-began
            trace=dict(seed=seed,arm=arm,start=start,step=step,load=load,indices=indices,draws=draws,queries=q,labels=y,repeat=repeat,
                logits=z.detach(),before_model=before,after_model=copy.deepcopy(model.state_dict()),before_optimizer=old,
                after_optimizer=copy.deepcopy(opt.state_dict()),gradients=gradients,names=names,history_before=hb,query_before=qb,
                history_after=hr.get_state().clone(),query_after=qr.get_state().clone(),step_seconds=elapsed)
            traces.append(trace)
            checkpoint=folder/f'{seed}_{arm}_{step}.pt';assert not checkpoint.exists()
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,history_rng=hr.get_state(),query_rng=qr.get_state()),checkpoint)
            if step_callback: evaluations.extend(step_callback(checkpoint,step))
        return traces,evaluations


def evaluate(path,rows,mask=None):
    cell=training.evaluate(training.previous.load_model(path),rows)
    if mask is not None:cell['fixed']=transport.fixed_metrics(cell['logits'],cell['labels'],mask)
    return cell


def criteria(base,immediate,final):
    results=[]
    for seed in (3461,3463):
        def get(records,arm,pool):return next(c for c in records if (c['seed'],c['arm'],c['pool'])==(seed,arm,pool))
        a,b=(get(immediate,arm,'train_32') for arm in ('control','reset'))
        immediate_delta=b['fixed']['error_ce']-a['fixed']['error_ce']
        a,b=(get(final,arm,'train_32') for arm in ('control','reset'))
        error_delta=b['fixed']['error_ce']-a['fixed']['error_ce']
        accuracy_delta=b['metrics']['groups']['all']['accuracy']-a['metrics']['groups']['all']['accuracy']
        mean={arm:sum(get(final,arm,f'train_{m}')['metrics']['groups']['all']['ce'] for m in (2,8,16,32))/4 for arm in ('control','reset')}
        equal_delta=mean['reset']-mean['control']
        results.append(dict(seed=seed,immediate_error_ce_delta=immediate_delta,final_error_ce_delta=error_delta,
            final_train32_accuracy_delta=accuracy_delta,final_equal_load_ce_delta=equal_delta,
            immediate_benefit=immediate_delta < -1e-10,
            four_update_benefit=error_delta < -1e-10 and equal_delta < -1e-10 and accuracy_delta>=0))
    return dict(seeds=results,immediate_benefit_all=all(r['immediate_benefit'] for r in results),
        four_update_benefit_all=all(r['four_update_benefit'] for r in results))


def inputs():
    traces,_=transport.inputs();starts=[t for t in traces if t['step']==11004]
    development=json.loads((training.parent.OUT/'development_data.json').read_text())['pools']
    test=json.loads((training.OUT/'test_data.json').read_text())['pools']
    return starts,{**development,**test}


def preflight():
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('test_memory_first_moment_reset',ROOT/SOURCES[-1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    record=dict(passed=result.wasSuccessful(),methods=result.testsRun,sources={n:training.replay.digest(ROOT/n) for n in SOURCES},
                profile=module.PROFILE)
    p=ROOT/'results'/('memory_first_moment_reset_preflight'+('' if record['passed'] else '_failed')+'.json')
    assert not p.exists();training.replay.write(p,record);assert record['passed']


def run():
    torch.set_num_threads(1);assert not OUT.exists(),'Preserve evidence'
    ready_path=ROOT/'results/memory_first_moment_reset_preflight.json';ready=json.loads(ready_path.read_text())
    assert ready['passed'] and ready['sources']=={n:training.replay.digest(ROOT/n) for n in SOURCES}
    files=dict(json.loads((transport.OUT/'lock.json').read_text())['files'])
    for name in ('report.json','audit.json','lock.json','metrics.csv'):
        p=transport.OUT/name;files[p.relative_to(ROOT).as_posix()]=training.replay.digest(p)
    audit=json.loads((transport.OUT/'audit.json').read_text());assert audit['passed'] and audit['report_sha256']==training.replay.digest(transport.OUT/'report.json')
    for name in (*SOURCES,PROTOCOL,ready_path.relative_to(ROOT).as_posix()):files[name]=training.replay.digest(ROOT/name)
    training.replay.check_files(files);OUT.mkdir()
    training.replay.write(OUT/'lock.json',dict(files=files,seeds=[3461,3463],start=11003,stop=11007,arms=['control','reset'],
        reset_parameters=list(transport.MEMORY),reset_field='exp_avg',batch=64,updates_per_arm=4,decision_resolution=1e-10))
    starts,pools=inputs();baseline=[];immediate=[];final=[];traces=[];bindings=[]
    original=json.loads((transport.replay_source.continuation.OUT/'report.json').read_text())
    for source in starts:
        seed=source['seed'];path=OUT/f'{seed}_start.pt';torch.save(start_payload(source),path)
        bindings.append(dict(seed=seed,path=path.relative_to(ROOT).as_posix(),sha256=training.replay.digest(path)))
        mask=None
        for split in ('train','validation','test'):
            for m in (2,8,16,32):
                pool=f'{split}_{m}';cell=evaluate(path,pools[pool]);cell.update(seed=seed,arm='baseline',step=11003,pool=pool,checkpoint=path.relative_to(ROOT).as_posix())
                if pool=='train_32':
                    mask=cell['logits'].argmax(-1)!=cell['labels'];cell['fixed']=transport.fixed_metrics(cell['logits'],cell['labels'],mask)
                baseline.append(cell)
        for arm in ('control','reset'):
            def callback(checkpoint,step):
                if step!=11004:return []
                cell=evaluate(checkpoint,pools['train_32'],mask);cell.update(seed=seed,arm=arm,step=step,pool='train_32',checkpoint=checkpoint.relative_to(ROOT).as_posix());return [cell]
            run,cells=train(path,pools,seed,arm,11007,OUT,callback);traces.extend(run);immediate.extend(cells)
            if arm=='control':
                assert training.replay.equal(run[0]['after_model'],source['after_model']) and training.replay.equal(run[0]['after_optimizer'],source['after_optimizer'])
                historical=next(r for r in original['runs'] if r['seed']==seed)
                assert [transport.replay_source.beta_record(t) for t in run]==[next(r for r in historical['steps'] if r['step']==t) for t in range(11004,11008)]
            checkpoint=OUT/f'{seed}_{arm}_11007.pt'
            for split in ('train','validation','test'):
                for m in (2,8,16,32):
                    pool=f'{split}_{m}';cell=evaluate(checkpoint,pools[pool],mask if pool=='train_32' else None)
                    cell.update(seed=seed,arm=arm,step=11007,pool=pool,checkpoint=checkpoint.relative_to(ROOT).as_posix());final.append(cell)
            print(f'Completed seed {seed} {arm}',flush=True)
    scores=baseline+immediate+final
    assert len(scores)==76 and sum(c['labels'].numel() for c in scores)==566016
    assert len(traces)==16 and sum(t['labels'].numel() for t in traces)==14848
    torch.save(scores,OUT/'scores.pt');torch.save(traces,OUT/'traces.pt')
    def records(cells):return [{k:v for k,v in c.items() if k not in ('logits','labels','repeat')} for c in cells]
    report=dict(status='complete',optimizer_calls=16,intervention_updates=8,control_reproduction_updates=8,new_post12000_updates=0,
        device='cpu',dtype='float64',profile=ready['profile'],gpu_peak_allocated_bytes=0,gpu_peak_reserved_bytes=0,
        histories=1024,input_events=32768,answer_targets=14848,batch=64,updates_per_arm=4,mean_targets_per_update=928,
        starts=bindings,baseline=records(baseline),immediate=records(immediate),final=records(final),criteria=criteria(baseline,immediate,final),
        scores_sha256=training.replay.digest(OUT/'scores.pt'),traces_sha256=training.replay.digest(OUT/'traces.pt'),
        lock_sha256=training.replay.digest(OUT/'lock.json'),checkpoints={p.relative_to(ROOT).as_posix():training.replay.digest(p) for p in OUT.glob('*.pt') if p.name not in ('scores.pt','traces.pt')})
    training.replay.check_files(files);training.replay.write(OUT/'report.json',report)
    fields=('seed','arm','step','pool','group','count','accuracy','ce')
    with (OUT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for c in scores:
            for group,metric in c['metrics']['groups'].items():w.writerow(dict(**{k:c[k] for k in ('seed','arm','step','pool')},group=group,**metric))
    print(json.dumps(report['criteria']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true')
    if p.parse_args().preflight:preflight()
    else:run()
