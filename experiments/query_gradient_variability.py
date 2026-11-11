"""Exact finite-pool alpha-gradient variation at immutable replay snapshots."""
import argparse
import csv
import importlib.util
import json
import time
import unittest

import torch
import retention_learning_replay as replay
import audit_objective_beta_pressure as tangent

ROOT=replay.ROOT
OUT=ROOT/'results/query_gradient_variability'
BATCH=64
SOURCE_NAMES=('experiments/query_gradient_variability.py','experiments/audit_query_gradient_variability.py',
              'tests/test_query_gradient_variability.py')


def statistics(gradients, slots):
    assert gradients.ndim==2 and gradients.shape[0]==len(slots)
    assert torch.isfinite(gradients).all()
    mean=gradients.mean()
    history_means=gradients.mean(1)
    total=float((gradients-mean).square().mean())
    between=float((history_means-mean).square().mean())
    within=float((gradients-history_means[:,None]).square().mean())
    assert abs(total-between-within)<=1e-11+1e-9*total
    one_se=(total/BATCH)**.5;all_se=(between/BATCH)**.5
    assigned=gradients.gather(1,slots[:,None]).flatten()
    return dict(mean=float(mean),population_variance=total,history_variance=between,query_variance=within,
                query_variance_fraction=within/total if total else None,
                one_query_batch_se=one_se,all_query_batch_se=all_se,
                all_to_one_se_ratio=all_se/one_se if one_se else None,
                one_query_snr=abs(float(mean))/one_se if one_se else None,
                all_query_snr=abs(float(mean))/all_se if all_se else None,
                mean_pressure=replay.pressure.pressure(float(mean)),
                assigned_mean=float(assigned.mean()),
                assigned_variance=float((assigned-assigned.mean()).square().mean()),
                targets=gradients.numel(),histories=gradients.shape[0],queries_per_history=gradients.shape[1])


def evaluate(model, rows, batch=64):
    events,queries,labels,repeat=tangent.inputs(rows,'all_queries')
    slots=(queries==torch.tensor([r['query'] for r in rows])[:,None]).long().argmax(1)
    outputs=[];gradients=[]
    before=replay.parent.study.parent.state_hash(model.state_dict())
    rng=torch.get_rng_state().clone()
    with torch.no_grad():
        beta=model.write_logit.sigmoid()
        for first in range(0,len(rows),batch):
            e,q,y=events[first:first+batch],queries[first:first+batch],labels[first:first+batch]
            logits,derivative=tangent.forward_tangent(model,e,q)
            probabilities=logits.softmax(-1)
            probabilities.scatter_add_(-1,y[...,None],-torch.ones_like(y[...,None],dtype=torch.float64))
            g=(probabilities*derivative).sum(-1)*beta*(1-beta)
            gradients.append(g);outputs.append(logits)
    assert before==replay.parent.study.parent.state_hash(model.state_dict())
    assert torch.equal(rng,torch.get_rng_state())
    g=torch.cat(gradients);logits=torch.cat(outputs)
    return dict(gradients=g,logits=logits,labels=labels,repeat=repeat,assigned_slots=slots,
                statistics=statistics(g,slots),metrics=replay.pressure.reduce_logits(logits,labels,repeat))


def load_model(checkpoint):
    model=replay.pressure.model_source.LoadReader('delta').eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
    return model


def preflight():
    spec=importlib.util.spec_from_file_location('test_query_gradient_variability',ROOT/'tests/test_query_gradient_variability.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    record=dict(passed=result.wasSuccessful(),methods=result.testsRun,
                sources={n:replay.digest(ROOT/n) for n in SOURCE_NAMES})
    replay.write(ROOT/'results/query_gradient_variability_preflight.json',record)
    assert record['passed']


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert not OUT.exists(),'Preserve completed and failed attempts'
    preflight_path=ROOT/'results/query_gradient_variability_preflight.json'
    ready=json.loads(preflight_path.read_text());assert ready['passed']
    assert ready['sources']=={n:replay.digest(ROOT/n) for n in SOURCE_NAMES}
    parent_report=json.loads((replay.OUT/'report.json').read_text())
    parent_audit=json.loads((replay.OUT/'audit.json').read_text())
    assert parent_audit['passed'] and parent_audit['report_sha256']==replay.digest(replay.OUT/'report.json')
    files=dict(json.loads((replay.OUT/'lock.json').read_text())['files'])
    for name in ('report.json','audit.json','lock.json','final_scores.pt'):
        path=replay.OUT/name;files[path.relative_to(ROOT).as_posix()]=replay.digest(path)
    for run in parent_report['runs']:
        path=replay.OUT/f"trajectory_{run['seed']}_{run['arm']}.json"
        files[path.relative_to(ROOT).as_posix()]=replay.digest(path)
        for snapshot in json.loads(path.read_text())['snapshots']:
            files[snapshot['checkpoint']]=snapshot['sha256']
    for name in (*SOURCE_NAMES,'docs/protocols/query-gradient-variability.md',preflight_path.relative_to(ROOT).as_posix()):
        files[name]=replay.digest(ROOT/name)
    replay.check_files(files)
    payload=json.loads((replay.parent.OUT/'development_data.json').read_text())
    OUT.mkdir()
    replay.write(OUT/'lock.json',dict(files=files,batch=64,steps=list(replay.PROBES),loads=list(replay.pressure.LOADS)))
    report=dict(status='running',device='cpu',dtype='float64',threads=1,training_updates=0,
                parameter='write_logit',batch=64,cells=[])
    saved=[];tick=time.perf_counter()
    for run in parent_report['runs']:
        trace=json.loads((replay.OUT/f"trajectory_{run['seed']}_{run['arm']}.json").read_text())
        for snapshot in trace['snapshots']:
            cp=ROOT/snapshot['checkpoint'];model=load_model(cp)
            old=torch.load(cp,weights_only=True)
            for load in replay.pressure.LOADS:
                cell=evaluate(model,payload['pools'][f'train_{load}'])
                all_old=next(c for c in old['cells'] if c['load']==load and c['objective']=='all_queries')
                assigned_old=next(c for c in old['cells'] if c['load']==load and c['objective']=='assigned')
                tangent.close(cell['logits'],all_old['logits'])
                tangent.close(cell['statistics']['mean'],float(all_old['gradients']['write_logit']),True)
                tangent.close(cell['statistics']['assigned_mean'],float(assigned_old['gradients']['write_logit']),True)
                record=dict(seed=run['seed'],arm=run['arm'],step=snapshot['step'],load=load,
                            checkpoint=snapshot['checkpoint'])
                saved.append(dict(**record,**cell))
                report['cells'].append(dict(**record,statistics=cell['statistics'],metrics=cell['metrics']))
        replay.write(OUT/'report.json',report)
        print(json.dumps(dict(seed=run['seed'],arm=run['arm'],cells=len(saved))),flush=True)
    assert len(saved)==112 and sum(c['statistics']['targets'] for c in saved)==1247232
    torch.save(saved,OUT/'targets.pt')
    rows=[]
    for c in report['cells']:
        for group,metrics in c['metrics'].items():
            rows.append(dict(seed=c['seed'],arm=c['arm'],step=c['step'],load=c['load'],group=group,
                             **metrics,**c['statistics']))
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report.update(status='complete',cells_count=112,targets=1247232,csv_rows=len(rows),
                  elapsed_seconds=time.perf_counter()-tick,targets_sha256=replay.digest(OUT/'targets.pt'),
                  lock_sha256=replay.digest(OUT/'lock.json'),csv_sha256=replay.digest(OUT/'metrics.csv'))
    replay.check_files(files);replay.write(OUT/'report.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    if parser.parse_args().preflight:preflight()
    else:run()
