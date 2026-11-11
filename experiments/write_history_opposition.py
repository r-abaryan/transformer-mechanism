"""History/position decomposition of saved trained-model value gradients."""
import argparse
import csv
import importlib.util
import json
import time
import unittest
import torch
import write_credit_assignment as primary

ROOT=primary.ROOT
OUT=ROOT/'results/write_history_opposition'
PROTOCOL='docs/protocols/write-history-opposition.md'
SOURCES=('experiments/write_history_opposition.py','experiments/audit_write_history_opposition.py',
         'tests/test_write_history_opposition.py')
GUARD=1e-14


def sign(x):return 'negative' if x < -GUARD else 'positive' if x > GUARD else 'unresolved'


def reduction(events,value_ids,supervised,seed,group):
    assert events.dtype==torch.float64 and events.ndim==3
    h,t,v=events.shape;assert value_ids.shape==(h,t) and supervised.shape==(h,)
    assert bool(torch.isfinite(events).all()) and bool(((value_ids>=0)&(value_ids<6)).all())
    per_history=torch.zeros(h,6,v,dtype=torch.float64)
    per_history.scatter_add_(1,value_ids[:,:,None].expand(-1,-1,v),events)
    event_energy=events.square().sum((1,2));map_energy=per_history.square().sum((1,2))
    within=map_energy-event_energy
    positions=torch.zeros(t,6,v,dtype=torch.float64)
    for step in range(t):positions[step].index_add_(0,value_ids[:,step],events[:,step])
    base=float(event_energy.sum());position_energy=float(positions.square().sum());joint=float(positions.sum(0).square().sum())
    w=float(within.sum());p=position_energy-base;c=joint-position_energy;b=c-w
    rows=[dict(seed=seed,group=group,history=i,supervised=bool(supervised[i]),event_energy=float(event_energy[i]),
               map_energy=float(map_energy[i]),within=float(within[i]),sign=sign(float(within[i]))) for i in range(h)]
    summary=dict(seed=seed,group=group,histories=h,positions=t,supervised_histories=int(supervised.sum()),
        event_energy=base,position_energy=position_energy,joint_energy=joint,within=w,
        between_temporal=b,between_same_position=p,temporal_cross=c,
        within_over_event=w/base if base else None,between_temporal_over_event=b/base if base else None,
        between_same_position_over_event=p/base if base else None,
        supervised_negative=sum(r['supervised'] and r['sign']=='negative' for r in rows),
        supervised_positive=sum(r['supervised'] and r['sign']=='positive' for r in rows),
        supervised_unresolved=sum(r['supervised'] and r['sign']=='unresolved' for r in rows),
        unsupervised_histories=sum(not r['supervised'] for r in rows))
    return summary,rows,positions


def load(source):
    lock=json.loads((source/'lock.json').read_text());report=json.loads((source/'report.json').read_text())
    audit=json.loads((source/'audit.json').read_text())
    assert audit['passed'] and audit['report_sha256']==primary.prior.digest(source/'report.json')
    assert report['lock_sha256']==primary.prior.digest(source/'lock.json')
    assert report['gradients_sha256']==primary.prior.digest(source/'gradients.pt')
    primary.prior.check_files(lock['files'])
    bundles=torch.load(source/'gradients.pt',weights_only=True);assert len(bundles)==len(lock['cases'])
    return lock,report,bundles


def execute(source,out,fixture=False):
    assert not out.exists(),'Preserve evidence'
    lock,report,bundles=load(source);assert fixture==lock['fixture']
    files=dict(lock['files'])
    for p in (*SOURCES,PROTOCOL):files[p]=primary.prior.digest(ROOT/p)
    for name in ('lock.json','report.json','audit.json','gradients.pt'):
        p=source/name;files[p.relative_to(ROOT).as_posix()]=primary.prior.digest(p)
    if not fixture:
        pre=ROOT/'results/write_history_opposition_preflight.json';passed=json.loads(pre.read_text())
        assert passed['passed'] and passed['sources']=={p:primary.prior.digest(ROOT/p) for p in SOURCES}
        files[pre.relative_to(ROOT).as_posix()]=primary.prior.digest(pre)
        p=source/'cross_terms.json';files[p.relative_to(ROOT).as_posix()]=primary.prior.digest(p)
    primary.prior.check_files(files);out.mkdir(parents=True)
    primary.prior.write(out/'lock.json',dict(source=source.relative_to(ROOT).as_posix(),fixture=fixture,files=files,
        cases=lock['cases'],dtype='float64',threads=torch.get_num_threads(),guard=GUARD))
    start=time.perf_counter();summary=[];history=[]
    for case,bundle in zip(lock['cases'],bundles):
        assert case['seed']==bundle['seed']
        rows=json.loads((ROOT/case['data']).read_text(encoding='utf-8-sig'))['pools'][case['pool']]
        ids=torch.tensor([[v for k,v in row['events']] for row in rows])
        for group in primary.GROUPS:
            g=bundle['gradients'][group];s,r,positions=reduction(g['events'],ids,bundle['masks'][group].any(1),case['seed'],group)
            torch.testing.assert_close(positions,g['positions'],atol=1e-12,rtol=1e-12)
            torch.testing.assert_close(positions.sum(0),g['embedding'],atol=1e-12,rtol=1e-12)
            summary.append(s);history.extend(r)
        print(json.dumps(summary[-3:]),flush=True)
    with (out/'histories.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    result=dict(status='complete',fixture=fixture,summary=summary,history_rows=len(history),cohorts=len(summary),
        model_evaluations=0,backward_calls=0,optimizer_calls=0,training_updates=0,
        seconds=time.perf_counter()-start,lock_sha256=primary.prior.digest(out/'lock.json'),
        histories_sha256=primary.prior.digest(out/'histories.csv'))
    primary.prior.check_files(files);primary.prior.write(out/'report.json',result)


def preflight():
    spec=importlib.util.spec_from_file_location('test_write_history_opposition',ROOT/SOURCES[-1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if result.wasSuccessful():
        path=ROOT/'results/write_history_opposition_preflight.json';assert not path.exists()
        primary.prior.write(path,dict(passed=True,methods=result.testsRun,sources={p:primary.prior.digest(ROOT/p) for p in SOURCES},
            initialized_fixture_primary_forward_cases=1,optimizer_calls=0))
    assert result.wasSuccessful()


if __name__=='__main__':
    torch.set_num_threads(1);parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    if parser.parse_args().preflight:preflight()
    else:execute(primary.OUT,OUT)
