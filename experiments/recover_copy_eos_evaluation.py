"""Evaluation-only recovery; preserve the six completed EOS training runs."""
import csv
import json
from pathlib import Path

import torch

import audit_copy_eos_training as auditor
from copy_eos_training import ROOT,DIRECTORY as PARENT,DATA,PROTOCOL,ARMS,evaluate,decision
from copy_length_exposure import evaluate as group_aware_evaluate,metrics as group_aware_metrics
from encoder_decoder_copy import metrics as legacy_metrics
from encoder_decoder_contract import Model
from position_distractors import digest
from recurrent_retrieval import write

TARGET=PARENT/'evaluation_recovery'
CORRECTION=ROOT/'docs/protocols/copy-eos-evaluation-recovery.md'


def main():
    assert CORRECTION.exists() and not TARGET.exists()
    original=json.loads((PARENT/'pretest.json').read_text(encoding='utf-8'))
    failed=json.loads((PARENT/'report.json').read_text(encoding='utf-8'))
    assert original['status']=='all_training_finished_before_test' and len(original['runs'])==6
    assert all(r['status']=='trained' and r['step']==6000 for r in original['runs'])
    assert [r['checkpoint_sha256'] for r in failed['runs']]==[r['checkpoint_sha256'] for r in original['runs']]
    hashes=dict(original['source_dependencies'])
    for path in (PARENT/'report.json',PARENT/'pretest.json',PARENT/'criterion_controls.json',Path(__file__),CORRECTION):
        hashes[path.relative_to(ROOT).as_posix()]=digest(path)
    for run in original['runs']:hashes[run['checkpoint']]=run['checkpoint_sha256']
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    # Demonstrate the actual failing boundary with known, correct predictions.
    rows=[[3,4,5,6,7,8,9,10,3,4],[4,3,5,6,7,8,9,10,4,3]];source=torch.tensor(rows)
    labels=torch.tensor([row+[2] for row in rows]);z=torch.full((2,11,16),-10.,dtype=torch.float64)
    z.scatter_(2,labels[:,:,None],10.);generated=[row+[2] for row in rows]
    try:legacy_metrics(z,source,generated)
    except AssertionError:failure_detected=True
    else:raise AssertionError('Legacy missing-group failure was not reproduced.')
    fixed=group_aware_metrics(z,rows,generated)
    assert set(fixed)=={'all','repeated'} and all(m['exact_accuracy']==1 for m in fixed.values())
    TARGET.mkdir();(TARGET/'pretest.json').write_bytes((PARENT/'pretest.json').read_bytes())
    write(TARGET/'evaluation_failure.json',dict(original_metric_failure_reproduced=failure_detected,
        corrected_boundary_passed=True,missing_group='distinct at length10',training_updates_added=0,
        preserved_hashes=hashes,recovery_protocol=CORRECTION.relative_to(ROOT).as_posix()))
    report=original;report.update(status='evaluation_recovery_running',source_dependencies=hashes,
        training_report=(PARENT/'report.json').relative_to(ROOT).as_posix(),training_updates_added=0,
        recovery_protocol=CORRECTION.relative_to(ROOT).as_posix(),ordinary_evaluator='copy_length_exposure.evaluate')
    payload=json.loads(DATA.read_text(encoding='utf-8'))
    write(TARGET/'report.json',report)
    for run in report['runs']:
        assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256']
        cp=torch.load(ROOT/run['checkpoint'],weights_only=True)
        assert cp['step']==6000 and all(float(s['step'])==6000 for s in cp['optimizer']['state'].values())
        model=Model();model.load_state_dict(cp['model']);saved={}
        for pool,rows in payload['pools'].items():
            met,ev=evaluate(model,rows,ARMS[run['arm']])
            if pool in run['metrics']:assert met==run['metrics'][pool]
            run['metrics'][pool]=met;saved[pool]=ev
        run['ordinary_stopping']={pool:group_aware_evaluate(model,payload['pools'][pool])[0]
            for pool in ('train_6','test_6','test_8','test_10')}
        path=TARGET/f"{run['seed']}_{run['arm']}_scores.pt";torch.save(saved,path)
        run.update(scores=path.relative_to(ROOT).as_posix(),scores_sha256=digest(path))
        write(TARGET/'report.json',report)
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    with (TARGET/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        rows=[dict(seed=r['seed'],arm=r['arm'],pool=pool,group=g,**m) for r in report['runs'] for pool,gg in r['metrics'].items() for g,m in gg.items()]
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report.update(status='complete',decision=decision(report['runs']),metrics_sha256=digest(TARGET/'metrics.csv'),prior_hashes_unchanged=True)
    write(TARGET/'report.json',report)
    # Only the audit artifact location changes; its numerical/decision routines are reused unchanged.
    auditor.DIRECTORY=TARGET;auditor.main()
    print(json.dumps(report['decision']),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()
