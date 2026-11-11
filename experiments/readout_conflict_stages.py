"""Three saved training stages, four fresh searches and cached endpoints."""
import argparse
import csv
import importlib.util
import json
import time
import unittest
import torch
import shared_readout_recovery as recovery

original=recovery.original
ROOT=recovery.ROOT
OUT=ROOT/'results/readout_conflict_stages'
PROTOCOL='docs/protocols/readout-conflict-training-stages.md'
SOURCES=('experiments/readout_conflict_stages.py','experiments/audit_readout_conflict_stages.py',
         'tests/test_readout_conflict_stages.py')


def cases():
    result=[]
    for seed in (3461,3463):
        for step in (0,3000,12000):
            directory='query_feedback_continuation' if step==12000 else 'query_feedback_training'
            result.append(dict(seed=seed,step=step,checkpoint=f'results/{directory}/{seed}_exhaustive_{step}.pt',
                               data='results/delta_query_coverage/development_data.json',pools=[f'train_{m}' for m in original.LOADS],cached=step==12000))
    return result


def flat_cells(bundles):
    cells=[]
    for bundle in bundles:
        case=bundle['case']
        for block in bundle['source']['blocks']:
            cells.append(dict(seed=case['seed'],step=case['step'],pool=block['pool'],arm='native',metrics=block['native_metrics']))
        for c in bundle['recovery']['cells']:
            cells.append(dict(step=case['step'],**c))
    return cells


def write_csv(path,cells):
    flat=[]
    for cell in cells:
        row={k:v for k,v in cell.items() if k!='metrics'}
        for group,stats in cell['metrics'].items():
            if group=='complete_maps':row[group]=stats
            else:
                for k,v in stats.items():row[group+'_'+k]=v
        flat.append(row)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)


@torch.no_grad()
def evaluate(case):
    checkpoint=torch.load(ROOT/case['checkpoint'],weights_only=True);assert checkpoint['step']==case['step']
    payload=json.loads((ROOT/case['data']).read_text(encoding='utf-8-sig'))
    with torch.random.fork_rng(devices=[]):
        model=original.prior.historical.parent.previous.load_model(ROOT/case['checkpoint']).eval()
    before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    source=dict(seed=case['seed'],blocks=[original.features(model,payload['pools'][p],p) for p in case['pools']])
    with torch.random.fork_rng(devices=[]):
        recovered=recovery.evaluate(source,case)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()) and torch.equal(rng,torch.get_rng_state())
    return dict(case=case,source=source,recovery=recovered)


def execute(out,selected=None,fixture=False):
    assert not out.exists(),'Preserve executed evidence'
    selected=cases() if selected is None else selected;prior=original.prior
    files={p:prior.digest(ROOT/p) for p in (*SOURCES,PROTOCOL,'experiments/shared_readout_feasibility.py',
        'experiments/audit_shared_readout_feasibility.py','experiments/shared_readout_recovery.py','experiments/audit_shared_readout_recovery.py',
        'experiments/query_feedback_training.py','experiments/association_load.py','requirements-analysis.txt')}
    for case in selected:
        for field in ('checkpoint','data'):files[case[field]]=prior.digest(ROOT/case[field])
    if not fixture:
        ready=ROOT/'results/readout_conflict_stages_preflight.json';pre=json.loads(ready.read_text());assert pre['passed']
        assert pre['sources']=={p:prior.digest(ROOT/p) for p in SOURCES};files[ready.relative_to(ROOT).as_posix()]=prior.digest(ready)
        for name in ('query_feedback_training','query_feedback_continuation','shared_readout_feasibility','shared_readout_recovery'):
            folder=ROOT/'results'/name;old=json.loads((folder/'audit.json').read_text());assert old['passed'] and old['report_sha256']==prior.digest(folder/'report.json')
            for filename in ('lock.json','report.json','audit.json'):
                p=folder/filename;files[p.relative_to(ROOT).as_posix()]=prior.digest(p)
        for folder in (original.OUT,recovery.OUT):
            for filename in ('witnesses.pt','metrics.csv'):
                p=folder/filename;files[p.relative_to(ROOT).as_posix()]=prior.digest(p)
    prior.check_files(files);out.mkdir(parents=True)
    prior.write(out/'lock.json',dict(files=files,cases=selected,fixture=fixture,options=recovery.OPTIONS,versions=prior.versions()))
    native_cache=None;recovery_cache=None
    if any(c['cached'] for c in selected):
        native_cache=torch.load(original.OUT/'witnesses.pt',weights_only=True);recovery_cache=torch.load(recovery.OUT/'witnesses.pt',weights_only=True)
    bundles=[];start=time.perf_counter()
    for case in selected:
        if case['cached']:
            source=next(b for b in native_cache if b['seed']==case['seed']);rec=next(b for b in recovery_cache if b['seed']==case['seed'])
            bundle=dict(case=case,source=source,recovery=rec)
        else:bundle=evaluate(case)
        bundles.append(bundle);torch.save(bundles,out/'stages.pt')
        print(json.dumps(dict(seed=case['seed'],step=case['step'],cached=case['cached'],outcome=bundle['recovery']['record']['outcome'])),flush=True)
    cells=flat_cells(bundles);write_csv(out/'metrics.csv',cells)
    report=dict(status='complete',summary=[dict(seed=b['case']['seed'],step=b['case']['step'],cached=b['case']['cached'],**b['recovery']['record']) for b in bundles],
                cells=cells,new_solver_calls=sum(b['recovery']['record']['solver_calls'] for b in bundles if not b['case']['cached']),
                fresh_native_queries=sum(block['labels'].numel() for b in bundles if not b['case']['cached'] for block in b['source']['blocks']),
                reported_native_queries=sum(block['labels'].numel() for b in bundles for block in b['source']['blocks']),
                histories=sum(len(block['labels']) for b in bundles for block in b['source']['blocks']),training_updates=0,optimizer_calls=0,
                fresh_native_prefills=2*sum(len(b['source']['blocks']) for b in bundles if not b['case']['cached']),seconds=time.perf_counter()-start,
                lock_sha256=prior.digest(out/'lock.json'),stages_sha256=prior.digest(out/'stages.pt'),metrics_sha256=prior.digest(out/'metrics.csv'))
    prior.check_files(files);prior.write(out/'report.json',report)


def preflight():
    spec=importlib.util.spec_from_file_location('checks',ROOT/SOURCES[-1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))
    p=ROOT/'results/readout_conflict_stages_preflight.json'
    if not result.wasSuccessful():p=ROOT/f'results/readout_conflict_stages_preflight_failed_{time.time_ns()}.json'
    assert not p.exists();original.prior.write(p,dict(passed=result.wasSuccessful(),methods=result.testsRun,sources={p:original.prior.digest(ROOT/p) for p in SOURCES}))
    assert result.wasSuccessful()


if __name__=='__main__':
    torch.set_num_threads(1);parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    if parser.parse_args().preflight:preflight()
    else:execute(OUT)
