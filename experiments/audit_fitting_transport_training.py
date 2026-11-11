"""Native replay, independent matrix gradients and dense block directions."""
import argparse
import json
from unittest.mock import patch

import torch
import train_fitting_transport as study
import audit_full_pool_fitting as numerical
from audit_shared_state_feasibility import compare


def dense_direction(g,state,names,shapes):
    h=study.transport.dense_inverse(state,len(g));mask=torch.zeros(len(g),dtype=torch.bool);offset=0
    for name in names:
        size=shapes[name].numel()
        if name in ('write_logit','key.weight'):mask[offset:offset+size]=True
        offset+=size
    assert offset==len(g) and set(names)=={'write_logit','key.weight','value.weight','output.weight','output.bias'}
    return -(h@(g*mask))*mask-(h@(g*~mask))*~mask


def check_run(case,run,pools,lock):
    if run['arm']=='lbfgs':result=numerical.check_lbfgs(case,run,pools,lock)
    else:
        trace=torch.load(study.ROOT/run['trace'],weights_only=True);shapes=trace['calls'][0]['gradients'];names=list(shapes)
        with (patch.object(numerical.study,'make_lbfgs',lambda model:study.make_optimizer(model,'block_diagonal')),
              patch.object(numerical,'two_loop',lambda g,state:dense_direction(g,state,names,shapes))):
            result=numerical.check_lbfgs(case,run,pools,lock)
    trace=torch.load(study.ROOT/run['trace'],weights_only=True);names=list(trace['calls'][0]['gradients']);steps=[]
    for i,step in enumerate(trace['steps']):
        state=next(iter(step['optimizer']['state'].values()));first=trace['calls'][step['first']]
        before=numerical.flat(step['before'],names);after=numerical.flat(step['after'],names)
        if torch.equal(before,after):steps.append(dict(call=i+1,moved=False));continue
        g=numerical.flat(first['gradients'],names);d=state['d'];t=float(state['t']);gtd=float(g.dot(d))
        matches=[c for c in trace['calls'][step['first']+1:step['last']] if float((numerical.flat(c['model'],names)-after).abs().max())<=1e-12]
        assert matches,'Accepted trial missing';accepted=matches[-1];ga=numerical.flat(accepted['gradients'],names)
        sufficient=accepted['loss']<=first['loss']+1e-4*t*gtd
        curvature=abs(float(ga.dot(d)))<=-.9*gtd
        steps.append(dict(call=i+1,moved=True,strong_wolfe=sufficient and curvature,descent=gtd<0,
                          objective_before=first['loss'],objective_after=accepted['loss']))
    result.update(seed=run['seed'],arm=run['arm'],steps=steps);return result


def audit(out,persist=True):
    lock=json.loads((out/'lock.json').read_text());report=json.loads((out/'report.json').read_text());study.replay.check_files(lock['files'])
    assert report['status']=='complete' and (lock['torch'],lock['native_hash'],lock['adapted_hash'])==(torch.__version__,study.previous.NATIVE_HASH,study.ADAPTED_HASH)
    for key,name in [('scores_sha256','scores.pt'),('lock_sha256','lock.json'),('pretest_sha256','pretest_decision.json')]:assert report[key]==study.replay.digest(out/name)
    assert lock['pools_sha256']==study.replay.digest(out/'pools.pt');pools=torch.load(out/'pools.pt',weights_only=True)
    if not lock['fixture']:
        cases,old=study.previous.inputs();assert cases==lock['cases'] and old==pools and (lock['budget'],lock['interval'])==(128,32)
    for rows in pools.values():numerical.reference.independent.check_rows(rows)
    assert [(r['seed'],r['arm']) for r in report['runs']]==[(c['seed'],a) for c in lock['cases'] for a in study.ARMS]
    checks=[]
    for case in lock['cases']:
        for arm in study.ARMS:
            run=next(r for r in report['runs'] if (r['seed'],r['arm'])==(case['seed'],arm))
            assert study.replay.digest(study.ROOT/run['trace'])==run['trace_sha256']
            for snap in run['snapshots']:assert study.replay.digest(study.ROOT/snap['checkpoint'])==snap['sha256']
            histories=sum(len(pools[f'train_{m}']) for m in study.previous.prior.LOADS)*run['closure_evaluations']
            targets=sum(len(pools[f'train_{m}'])*m for m in study.previous.prior.LOADS)*run['closure_evaluations']
            assert (run['histories'],run['input_events'],run['answer_targets'])==(histories,histories*32,targets)
            checks.append(check_run(case,run,pools,lock))
            print(json.dumps(dict(audited_seed=case['seed'],arm=arm,gradients=run['closure_evaluations'])),flush=True)
    controls=[];oldreport=json.loads((study.previous.OUT/'report.json').read_text()) if not lock['fixture'] else None
    for case in lock['cases']:
        a,b=[next(r for r in report['runs'] if (r['seed'],r['arm'])==(case['seed'],arm)) for arm in study.ARMS]
        x,y=[torch.load(study.ROOT/r['trace'],weights_only=True) for r in [a,b]]
        assert study.replay.equal(x['steps'][0],y['steps'][0]),'First-step null'
        changed=any(not torch.equal(p['optimizer']['state'][0]['d'],q['optimizer']['state'][0]['d']) for p,q in zip(x['steps'][1:],y['steps'][1:]))
        exact=None
        if oldreport:
            old=next(r for r in oldreport['runs'] if (r['seed'],r['arm'])==(case['seed'],'lbfgs'))
            assert study.replay.equal(x,torch.load(study.ROOT/old['trace'],weights_only=True)),'Historical native trace'
            assert study.replay.equal(torch.load(study.ROOT/a['final_checkpoint'],weights_only=True),torch.load(study.ROOT/old['final_checkpoint'],weights_only=True)),'Historical native endpoint'
            exact=True
        controls.append(dict(seed=case['seed'],first_step_identical=True,later_directions_differ=changed,historical_native_exact=exact))
    scores=torch.load(out/'scores.pt',weights_only=True)
    expected={(r['seed'],r['arm'],s['step'],f'train_{m}') for r in report['runs'] for s in r['snapshots'] for m in study.previous.prior.LOADS}
    expected|={(r['seed'],r['arm'],r['snapshots'][-1]['step'],f'{split}_{m}') for r in report['runs'] for split in ('validation','test') for m in study.previous.prior.LOADS}
    ids=[(c['seed'],c['arm'],c['step'],c['pool']) for c in scores];assert len(ids)==len(expected) and set(ids)==expected,'Missing evaluation group'
    assert len(report['cells'])==len(scores);maximum=0.;targets=0
    for cell,saved in zip(scores,report['cells']):
        compare(saved,{k:v for k,v in cell.items() if k not in ('logits','labels','repeat')})
        with torch.random.fork_rng(devices=[]):check=numerical.reference.audit_cell(study.ROOT/cell['checkpoint'],pools[cell['pool']],cell)
        maximum=max(maximum,check['max_logit_error']);targets+=check['targets']
    for run in report['runs']:
        for snap in run['snapshots']:
            metrics={str(m):next(c['metrics'] for c in scores if (c['seed'],c['arm'],c['step'],c['pool'])==(run['seed'],run['arm'],snap['step'],f'train_{m}')) for m in study.previous.prior.LOADS}
            compare(metrics,snap['training']);assert snap['fit']==study.previous.prior.fit(metrics)
            assert snap['perfect']==all(v['groups']['all']['accuracy']==1 for v in metrics.values())
    choice=study.decision(report['runs']);compare(choice,report['decision']);compare(choice,json.loads((out/'pretest_decision.json').read_text()))
    for pair in choice['pairs']:
        seed=pair['seed'];errors={}
        for arm in study.ARMS:
            final=next(r for r in report['runs'] if (r['seed'],r['arm'])==(seed,arm))['snapshots'][-1]['step']
            selected=[c for c in scores if (c['seed'],c['arm'],c['step'])==(seed,arm,final) and c['pool'].startswith('train_')]
            errors[arm]=sum(int(c['logits'].argmax(-1).ne(c['labels']).sum()) for c in selected)
        assert (pair['full']['errors'],pair['block_diagonal']['errors'])==(errors['lbfgs'],errors['block_diagonal'])
        assert pair['removal_increases_errors']==(errors['block_diagonal']>errors['lbfgs'])
    assert targets==report['evaluation_targets']
    for field in ('optimizer_calls','histories','input_events','answer_targets'):assert report[field]==sum(r[field] for r in report['runs'])
    assert report['gradient_evaluations']==sum(r['closure_evaluations'] for r in report['runs'])
    result=dict(passed=True,report_sha256=study.replay.digest(out/'report.json'),protected_files=len(lock['files']),cells=len(scores),evaluation_targets=targets,
                checks=checks,pair_controls=controls,max_logit_error=maximum,max_gradient_error=max(c['max_gradient_error'] for c in checks),max_direction_error=max(c['max_direction_error'] for c in checks))
    if persist:
        assert not (out/'audit.json').exists();study.replay.write(out/'audit.json',result)
    return result


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser();parser.add_argument('out',nargs='?',default=str(study.OUT));args=parser.parse_args()
    result=audit(study.ROOT/args.out);print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
