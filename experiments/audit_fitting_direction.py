"""Independent matrix gradients/directions plus native engine and saved-score replay."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import torch
import fitting_direction as study
import audit_full_pool_fitting as numerical
from audit_shared_state_feasibility import compare


def check_run(case,run,pools,lock):
    # Native control uses the unmodified installed step. Intervention replay
    # shares its adapter, but its directions are checked independently as -g.
    if run['arm']=='lbfgs':result=numerical.check_lbfgs(case,run,pools,lock)
    else:
        with (patch.object(numerical.study,'make_lbfgs',lambda model:study.DirectionLBFGS(model.parameters(),'negative_gradient')),
              patch.object(numerical,'two_loop',lambda g,state:-g)):
            result=numerical.check_lbfgs(case,run,pools,lock)
    trace=torch.load(study.ROOT/run['trace'],weights_only=True);names=list(trace['calls'][0]['gradients']);records=[]
    for i,step in enumerate(trace['steps']):
        state=next(iter(step['optimizer']['state'].values()));first=trace['calls'][step['first']]
        g=numerical.flat(first['gradients'],names);before=numerical.flat(step['before'],names);after=numerical.flat(step['after'],names)
        if torch.equal(before,after):
            records.append(dict(index=i+1,first=step['first'],last=step['last'],moved=False));continue
        d=state['d'];t=float(state['t']);gtd=float(g.dot(d))
        trials=trace['calls'][step['first']+1:step['last']]
        matches=[c for c in trials if float((numerical.flat(c['model'],names)-after).abs().max())<=1e-12]
        assert matches,'Accepted trial missing';accepted=matches[-1]
        ga=numerical.flat(accepted['gradients'],names)
        sufficient=accepted['loss']<=first['loss']+1e-4*t*gtd
        curvature=abs(float(ga.dot(d)))<=-.9*gtd
        cosine=float((-g).dot(d)/(g.norm()*d.norm()))
        records.append(dict(index=i+1,first=step['first'],last=step['last'],moved=True,step_length=t,
                            gradient_norm=float(g.norm()),direction_norm=float(d.norm()),cosine_to_negative_gradient=cosine,
                            objective_before=first['loss'],objective_after=accepted['loss'],
                            sufficient_decrease=sufficient,curvature=curvature,strong_wolfe=sufficient and curvature))
    result.update(seed=case['seed'],arm=run['arm'],steps=records)
    return result


def audit_worker(args):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    return check_run(*args)


def audit(out,persist=True):
    replay=study.replay;report=json.loads((out/'report.json').read_text());lock=json.loads((out/'lock.json').read_text())
    assert report['status']=='complete';replay.check_files(lock['files'])
    assert (lock['torch'],lock['native_hash'],lock['adapted_hash'])==(torch.__version__,study.NATIVE_HASH,study.ADAPTED_HASH)
    for key,file in (('scores_sha256','scores.pt'),('lock_sha256','lock.json'),('pretest_sha256','pretest_decision.json')):
        assert report[key]==replay.digest(out/file)
    assert lock['pools_sha256']==replay.digest(out/'pools.pt');pools=torch.load(out/'pools.pt',weights_only=True)
    if not lock['fixture']:
        cases,original=study.inputs();assert cases==lock['cases'] and pools==original
        assert (lock['budget'],lock['interval'])==(128,32)
    for rows in pools.values():numerical.reference.independent.check_rows(rows)
    assert [(r['seed'],r['arm']) for r in report['runs']]==[(c['seed'],a) for c in lock['cases'] for a in study.ARMS]
    jobs=[]
    for case in lock['cases']:
        for arm in study.ARMS:
            run=next(r for r in report['runs'] if (r['seed'],r['arm'])==(case['seed'],arm))
            assert replay.digest(study.ROOT/run['trace'])==run['trace_sha256']
            for snap in run['snapshots']:assert replay.digest(study.ROOT/snap['checkpoint'])==snap['sha256']
            n=run['closure_evaluations'];h=sum(len(pools[f'train_{m}']) for m in study.prior.LOADS)*n
            answers=sum(len(pools[f'train_{m}'])*m for m in study.prior.LOADS)*n
            assert (run['histories'],run['input_events'],run['answer_targets'])==(h,h*32,answers)
            jobs.append((case,run,pools,lock))
    if lock['fixture']:checks=list(map(audit_worker,jobs))
    else:
        with ProcessPoolExecutor(max_workers=3) as executor:checks=list(executor.map(audit_worker,jobs))
    pairs=[]
    historical=json.loads((study.SOURCE/'report.json').read_text()) if not lock['fixture'] else None
    for case in lock['cases']:
        a,b=[next(r for r in report['runs'] if (r['seed'],r['arm'])==(case['seed'],arm)) for arm in study.ARMS]
        x,y=[torch.load(study.ROOT/r['trace'],weights_only=True) for r in (a,b)]
        assert replay.equal(x['steps'][0],y['steps'][0]),'First-step null control'
        changed=any(not torch.equal(p['optimizer']['state'][0]['d'],q['optimizer']['state'][0]['d']) for p,q in zip(x['steps'][1:],y['steps'][1:]))
        native_match=None
        if historical:
            old=next(r for r in historical['runs'] if (r['seed'],r['arm'])==(case['seed'],'lbfgs'))
            before=torch.load(study.ROOT/old['final_checkpoint'],weights_only=True)
            now=torch.load(study.ROOT/a['final_checkpoint'],weights_only=True)
            assert replay.equal(before,now),'Historical native control endpoint'
            assert replay.equal(x,torch.load(study.ROOT/old['trace'],weights_only=True)),'Historical native control trials'
            native_match=True
        pairs.append(dict(seed=case['seed'],first_step_identical=True,later_directions_differ=changed,historical_native_exact=native_match))
    scores=torch.load(out/'scores.pt',weights_only=True)
    expected={(r['seed'],r['arm'],s['step'],f'train_{m}') for r in report['runs'] for s in r['snapshots'] for m in study.prior.LOADS}
    expected|={(r['seed'],r['arm'],r['snapshots'][-1]['step'],f'{split}_{m}') for r in report['runs'] for split in ('validation','test') for m in study.prior.LOADS}
    ids=[(c['seed'],c['arm'],c['step'],c['pool']) for c in scores]
    assert len(ids)==len(expected) and set(ids)==expected,'Missing evaluation group'
    assert len(report['cells'])==len(scores);error=0.;targets=0
    for cell,saved in zip(scores,report['cells']):
        compare(saved,{k:v for k,v in cell.items() if k not in ('logits','labels','repeat')})
        with torch.random.fork_rng(devices=[]):check=numerical.reference.audit_cell(study.ROOT/cell['checkpoint'],pools[cell['pool']],cell)
        error=max(error,check['max_logit_error']);targets+=check['targets']
    for run in report['runs']:
        for snap in run['snapshots']:
            metrics={str(m):next(c['metrics'] for c in scores if (c['seed'],c['arm'],c['step'],c['pool'])==(run['seed'],run['arm'],snap['step'],f'train_{m}')) for m in study.prior.LOADS}
            compare(metrics,snap['training']);assert snap['fit']==study.prior.fit(metrics)
            assert snap['perfect']==all(v['groups']['all']['accuracy']==1 for v in metrics.values())
    choice=study.paired_decision(report['runs']);compare(choice,report['decision']);compare(choice,json.loads((out/'pretest_decision.json').read_text()))
    assert targets==report['evaluation_targets']
    for field in ('optimizer_calls','answer_targets','histories','input_events'):assert report[field]==sum(r[field] for r in report['runs'])
    assert report['gradient_evaluations']==sum(r['closure_evaluations'] for r in report['runs'])
    result=dict(passed=True,report_sha256=replay.digest(out/'report.json'),protected_files=len(lock['files']),cells=len(scores),
                evaluation_targets=targets,checks=checks,pair_controls=pairs,max_logit_error=error,
                max_gradient_error=max(c['max_gradient_error'] for c in checks),max_direction_error=max(c['max_direction_error'] for c in checks))
    if persist:
        assert not (out/'audit.json').exists();replay.write(out/'audit.json',result)
    return result


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser();parser.add_argument('out',nargs='?',default=str(study.OUT));args=parser.parse_args()
    result=audit(study.ROOT/args.out);print(json.dumps({k:v for k,v in result.items() if k not in ('checks',)}),flush=True)
