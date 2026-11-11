"""Post-run independent scalar/decision audit, no training or result rewriting."""
import csv
import json
import math
from pathlib import Path

import torch
import delta_retrieval as study


def reduce_scores(scores, rows):
    result = {}
    for group in ('all', *study.GROUPS):
        selected = [(values, row) for values, row in zip(scores.tolist(), rows)
                    if group == 'all' or row['group'] == group]
        if not selected:
            continue
        losses=[];correct=0
        for values,row in selected:
            memory={}
            for key,value in row['events']:memory[key]=value
            y=memory[row['query']]
            assert y==row['label']
            peak=max(values)
            losses.append(peak+math.log(math.fsum(math.exp(v-peak) for v in values))-values[y])
            correct += max(range(6),key=lambda i:values[i])==y
        result[group]=dict(count=len(selected),correct=correct,accuracy=correct/len(selected),
                           loss=math.fsum(losses)/len(selected))
    return result


def check_metrics(recorded, independent):
    assert recorded.keys()==independent.keys()
    for group,cell in independent.items():
        saved=recorded[group]
        assert saved['count']==cell['count'] and saved['correct']==cell['correct']
        assert abs(saved['accuracy']-cell['accuracy'])<1e-12
        assert abs(saved['loss']-cell['loss'])<1e-10


def independent_decision(runs):
    out={}
    for arm in study.ARMS:
        rows=[r for r in runs if r['phase']=='confirmation' and r['arm']==arm]
        ready={r['seed'] for r in rows}==set(study.CONFIRM_SEEDS) and all(r['status']=='complete' for r in rows)
        train=[];heldout=[]
        for row in rows:
            for pool,groups in row['metrics'].items():
                target=train if pool.startswith('train_') else heldout
                target.extend(groups[g]['correct']/groups[g]['count'] for g in study.GROUPS)
        out[arm]=dict(training_fit=ready and min(train)>=.99,reliability=ready and min(heldout)>=.95)
    preserved=True
    for seed in study.CONFIRM_SEEDS:
        pair={r['arm']:r for r in runs if r['phase']=='confirmation' and r['seed']==seed}
        for pool in ('test_4','test_8','test_16','test_32'):
            for group in study.GROUPS:
                a=pair['delta']['metrics'][pool][group];b=pair['softmax']['metrics'][pool][group]
                preserved &= a['correct']/a['count']+0.02+1e-12>=b['correct']/b['count']
    out['quality_preservation']=bool(preserved)
    return out


def main():
    torch.set_num_threads(1)
    path=study.OUT/'report.json';report=json.loads(path.read_text());assert report['status']=='complete'
    assert report['source_dependencies']==study.dependencies()
    prior=json.loads((study.OUT/'prior_hashes.json').read_text())
    assert all(study.digest(study.ROOT/p)==h for p,h in prior.items())
    data={phase:json.loads((study.OUT/f'{phase}_data.json').read_text()) for phase in ('development','confirmation')}
    assert all(study.digest(study.OUT/f'{phase}_data.json')==h for phase,h in report['data_sha256'].items())
    study.audit_data(dict(pools={f'{phase}_{p}':rows for phase,d in data.items() for p,rows in d['pools'].items()}))
    cells=[];csv_rows=[];max_error=0.;predictions=0
    for run in report['runs']:
        assert run['status']=='complete' and run['updates']==3000
        cp=study.ROOT/run['checkpoint'];assert study.digest(cp)==run['checkpoint_sha256']
        saved=torch.load(cp,weights_only=True);assert saved['step']==3000
        assert all(state['step'].item()==3000 for state in saved['optimizer']['state'].values())
        model=study.Reader(run['arm']);model.load_state_dict(saved['model'])
        for pool,rows in data[run['phase']]['pools'].items():
            e,q,_=study.tensors(rows)
            with torch.no_grad():
                native=torch.cat([model(a,b) for a,b in zip(e.split(128),q.split(128))])
                other=torch.cat([study.reference(model,a,b) for a,b in zip(e.split(128),q.split(128))])
            error=(native-other).abs().max().item();max_error=max(error,max_error)
            assert error<1e-9 and torch.equal(native.argmax(-1),other.argmax(-1))
            reduced=reduce_scores(other,rows);check_metrics(run['metrics'][pool],reduced)
            cells.append(dict(phase=run['phase'],seed=run['seed'],arm=run['arm'],recipe=run['recipe'],pool=pool,predictions=len(rows),max_logit_error=error))
            for group,value in reduced.items():
                csv_rows.append(dict(phase=run['phase'],seed=run['seed'],arm=run['arm'],recipe=run['recipe'],pool=pool,group=group,**value))
            predictions+=len(rows)
    decision=independent_decision(report['runs']);assert decision==report['confirmation_decision']
    # Wrong reported scalar and wrong reliability decision must be rejected.
    cell=report['runs'][0]['metrics']['train_4']
    wrong=json.loads(json.dumps(cell));wrong['all']['loss']+=.01
    try:check_metrics(wrong,cell)
    except AssertionError:pass
    else:raise AssertionError('Incorrect loss not detected')
    altered=json.loads(json.dumps(report['confirmation_decision']));altered['softmax']['reliability']=True
    assert altered!=decision
    for seed in (*study.DEV_SEEDS,*study.CONFIRM_SEEDS):
        for phase in ('development','confirmation'):
            subset=[r for r in report['runs'] if r['phase']==phase and r['seed']==seed]
            if subset:
                assert len({r['shared_sha256'] for r in subset})==1
                assert len({r['stream_sha256'] for r in subset})==1
    assert report['benchmark']['passed'] is False and 'no benchmark executed' in report['benchmark']['reason']
    csv_path=study.OUT/'metrics.csv';assert not csv_path.exists()
    with csv_path.open('x',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    target=study.OUT/'independent_audit.json';assert not target.exists()
    study.write(target,dict(passed=True,report_sha256=study.digest(path),audit_source_sha256=study.digest(Path(__file__)),
                            cells=cells,predictions=predictions,metric_cells=len(csv_rows),max_logit_error=max_error,
                            argmax_changes=0,optimizer_steps_verified=True,common_initialization_and_streams_match=True,
                            wrong_scalar_and_decision_detected=True,prior_files_unchanged=len(prior),decision=decision,
                            scope='Post-run scalar reduction uses Python math; matrix reference imported from study. No GPU benchmark or independent stack claim.'))
    print(json.dumps(dict(passed=True,predictions=predictions,metric_cells=len(csv_rows),max_logit_error=max_error,prior_files_unchanged=len(prior))))


if __name__=='__main__':main()
