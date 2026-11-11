"""Matrix gradients, scalar Adam, L-BFGS oracle replay and complete metrics."""
import copy
import inspect
import json
from pathlib import Path
import torch
import torch.optim.lbfgs as lbfgs_source
import full_pool_fitting as study
import audit_query_feedback_training as reference
from audit_shared_state_feasibility import compare


def independent_gradient(model,table,indices=None,load=None):
    result={n:torch.zeros_like(p) for n,p in model.named_parameters()};loss=0.
    for m in (study.LOADS if load is None else (load,)):
        e,q,y,_=table[m]
        if indices is not None:e,q,y=e[indices],q[indices],y[indices]
        z,_=reference.independent.forward_tangent(model,e,q)
        part=(z.logsumexp(-1)-z.gather(-1,y[...,None]).squeeze(-1)).mean()/(4 if load is None else 1)
        g=torch.autograd.grad(part,tuple(model.parameters()));loss+=float(part.detach())
        for (name,_),value in zip(model.named_parameters(),g):result[name]+=value.detach()
    return loss,result


def check_gradient(model,table,record,indices=None,load=None):
    loss,expected=independent_gradient(model,table,indices,load);assert abs(loss-record['loss'])<=1e-9,'Training loss'
    assert set(record['gradients'])==set(expected)
    return max(reference.independent.close(record['gradients'][n],v,gradient=True) for n,v in expected.items())


def flat(values,names):return torch.cat([values[n].flatten() for n in names])


def two_loop(g,state):
    """Independent inverse-Hessian direction using the stored secant pairs."""
    q=-g.clone();alphas=[]
    for y,s,rho in zip(reversed(state['old_dirs']),reversed(state['old_stps']),reversed(state['ro'])):
        a=rho*torch.dot(s,q);alphas.append(a);q=q-a*y
    r=q*state['H_diag']
    for y,s,rho,a in zip(state['old_dirs'],state['old_stps'],state['ro'],reversed(alphas)):
        r=r+s*(a-rho*torch.dot(y,r))
    return r


def check_lbfgs(case,run,pools,lock):
    model,_,start=study.load(study.ROOT/case['checkpoint']);opt=study.make_lbfgs(model)
    table={m:reference.independent.inputs(pools[f'train_{m}'],'all_queries') for m in study.LOADS}
    trace=torch.load(study.ROOT/run['trace'],weights_only=True);calls=trace['calls'];used=0;maximum=0.;direction_error=0.
    states={0:copy.deepcopy(model.state_dict())}
    for step in trace['steps']:
        assert step['first']==used and study.prior.training.replay.equal(model.state_dict(),step['before'])
        opt.param_groups[0]['max_eval']=min(26,lock['budget']-used-1)
        assert lock['budget']-used>=2
        def closure():
            nonlocal used,maximum
            record=calls[used];assert study.prior.training.replay.equal(model.state_dict(),record['model']),'Trial parameter mismatch'
            maximum=max(maximum,check_gradient(model,table,record))
            for n,p in model.named_parameters():p.grad=record['gradients'][n].clone()
            used+=1
            return torch.tensor(record['loss'],dtype=torch.float64)
        opt.step(closure)
        assert used==step['last'] and study.prior.training.replay.equal(model.state_dict(),step['after'])
        assert study.prior.training.replay.equal(opt.state_dict(),step['optimizer']),'L-BFGS state replay'
        state=next(iter(opt.state.values()));g=flat(calls[step['first']]['gradients'],list(dict(model.named_parameters())))
        if 'd' in state and float(g.abs().max())>1e-10:
            want=two_loop(g,state);err=float((want-state['d']).abs().max());assert err<=1e-9*(1+float(want.abs().max())),'Two-loop direction'
            direction_error=max(direction_error,err)
            expected=flat(step['before'],list(dict(model.named_parameters())))+state['t']*state['d']
            actual=flat(step['after'],list(dict(model.named_parameters())))
            # A directional-tolerance exit can retain cached direction without movement.
            if float(g.dot(state['d']))<=-1e-12:assert float((actual-expected).abs().max())<=1e-11,'Accepted displacement'
            for y,s,rho in zip(state['old_dirs'],state['old_stps'],state['ro']):assert abs(float(rho*y.dot(s))-1)<=1e-9,'Secant reciprocal'
        states[used]=copy.deepcopy(model.state_dict())
    assert used==len(calls)==run['closure_evaluations']<=lock['budget'] and lock['budget']-used<2
    assert run['optimizer_calls']==len(trace['steps'])
    final=torch.load(study.ROOT/run['final_checkpoint'],weights_only=True)
    assert study.prior.training.replay.equal(final['model'],model.state_dict()) and study.prior.training.replay.equal(final['optimizer'],opt.state_dict())
    for snap in run['snapshots']:assert snap['step'] in states and study.prior.training.replay.equal(states[snap['step']],torch.load(study.ROOT/snap['checkpoint'],weights_only=True)['model'])
    return dict(max_gradient_error=maximum,max_direction_error=direction_error)


def check_adam(case,run,pools,lock,closure_count):
    model,opt,start=study.load(study.ROOT/case['checkpoint']);optimizer=copy.deepcopy(opt.state_dict());group=optimizer['param_groups'][0]
    names=list(dict(model.named_parameters()));state=copy.deepcopy(model.state_dict());table={m:reference.independent.inputs(pools[f'train_{m}'],'all_queries') for m in study.LOADS}
    hr=torch.Generator().manual_seed(case['seed']+10000);qr=torch.Generator().manual_seed(case['seed']+20000)
    for step in range(start):
        m=study.LOADS[step%4];torch.randint(len(table[m][0]),(64,),generator=hr);torch.randint(m,(64,m),generator=qr)
    trace=torch.load(study.ROOT/run['trace'],weights_only=True);assert len(trace)==closure_count*lock['ratio']==run['optimizer_calls']
    max_gradient=max_update=0.;targets=0;snapshots={s['step']:s for s in run['snapshots']}
    for index,record in enumerate(trace):
        m=study.LOADS[(start+index)%4];ids=torch.randint(len(table[m][0]),(64,),generator=hr);draws=torch.randint(m,(64,m),generator=qr)
        assert record['load']==m and torch.equal(record['indices'],ids) and torch.equal(record['draws'],draws),'Adam stream'
        model.load_state_dict(state);max_gradient=max(max_gradient,check_gradient(model,table,record,ids,m))
        new={};b1,b2=group['betas']
        for name,pid in zip(names,group['params']):
            g=record['gradients'][name];old=optimizer['state'].get(pid,{})
            count=int(old['step'])+1 if old else 1
            first=b1*old.get('exp_avg',torch.zeros_like(g))+(1-b1)*g
            second=b2*old.get('exp_avg_sq',torch.zeros_like(g))+(1-b2)*g.square()
            wanted=state[name]-group['lr']*(first/(1-b1**count))/((second/(1-b2**count)).sqrt()+group['eps'])
            err=float((wanted-record['after'][name]).abs().max());assert err<=1e-11,'Scalar Adam update';max_update=max(max_update,err)
            new[name]=record['after'][name];optimizer['state'][pid]=dict(step=torch.tensor(float(count)),exp_avg=first,exp_avg_sq=second)
        state=new;targets+=64*m
        if index+1 in snapshots:
            cp=torch.load(study.ROOT/snapshots[index+1]['checkpoint'],weights_only=True)
            assert study.prior.training.replay.equal(state,cp['model']) and cp['step']==start+index+1
        if (index+1)%1536==0:print(json.dumps(dict(audit_seed=case['seed'],arm='adam',updates=index+1)),flush=True)
    final=torch.load(study.ROOT/run['final_checkpoint'],weights_only=True)
    assert study.prior.training.replay.equal(state,final['model']) and final['optimizer']['param_groups']==optimizer['param_groups']
    for pid,want in optimizer['state'].items():
        got=final['optimizer']['state'][pid];assert int(got['step'])==int(want['step'])
        for n in ('exp_avg','exp_avg_sq'):assert float((got[n]-want[n]).abs().max())<=1e-11,'Adam final moments'
    assert targets==run['answer_targets']
    return dict(max_gradient_error=max_gradient,max_update_error=max_update)


def audit(out,persist=True):
    replay=study.prior.training.replay;report=json.loads((out/'report.json').read_text());lock=json.loads((out/'lock.json').read_text())
    assert report['status']=='complete';replay.check_files(lock['files']);assert lock['torch']==torch.__version__
    assert lock['lbfgs_implementation_sha256']==replay.digest(Path(inspect.getsourcefile(lbfgs_source)))
    for field,file in (('scores_sha256','scores.pt'),('lock_sha256','lock.json'),('pretest_sha256','pretest_decision.json')):assert report[field]==replay.digest(out/file)
    assert lock['pools_sha256']==replay.digest(out/'pools.pt');pools=torch.load(out/'pools.pt',weights_only=True)
    if not lock['fixture']:
        cases,original=study.inputs();assert lock['cases']==cases and pools==original
        assert (lock['budget'],lock['ratio'],lock['interval'])==(128,48,32)
    for rows in pools.values():reference.independent.check_rows(rows)
    expected_runs=[(c['seed'],a) for c in lock['cases'] for a in ('lbfgs','adam')];assert [(r['seed'],r['arm']) for r in report['runs']]==expected_runs
    checks=[]
    for case in lock['cases']:
        a,b=[next(r for r in report['runs'] if (r['seed'],r['arm'])==(case['seed'],arm)) for arm in ('lbfgs','adam')]
        for run in (a,b):
            assert replay.digest(study.ROOT/run['trace'])==run['trace_sha256']
            for s in run['snapshots']:assert replay.digest(study.ROOT/s['checkpoint'])==s['sha256']
        checks.append(check_lbfgs(case,a,pools,lock));checks.append(check_adam(case,b,pools,lock,a['closure_evaluations']))
        for r in (a,b):
            count=r['closure_evaluations'] if r['arm']=='lbfgs' else r['optimizer_calls']
            h=sum(len(pools[f'train_{m}']) for m in study.LOADS)*count if r['arm']=='lbfgs' else count*64
            targets=sum(len(pools[f'train_{m}'])*m for m in study.LOADS)*count if r['arm']=='lbfgs' else sum(64*study.LOADS[(r['source_step']+i)%4] for i in range(count))
            assert (r['histories'],r['input_events'],r['answer_targets'])==(h,h*32,targets)
        if not lock['fixture']:assert (a['answer_targets'],a['histories'])==(b['answer_targets'],b['histories'])
    scores=torch.load(out/'scores.pt',weights_only=True)
    expected={(r['seed'],r['arm'],s['step'],f'train_{m}') for r in report['runs'] for s in r['snapshots'] for m in study.LOADS}
    expected|={(r['seed'],r['arm'],r['snapshots'][-1]['step'],f'{split}_{m}') for r in report['runs'] for split in ('validation','test') for m in study.LOADS}
    ids=[(c['seed'],c['arm'],c['step'],c['pool']) for c in scores];assert len(ids)==len(expected) and set(ids)==expected
    assert len(report['cells'])==len(scores);targets=0;logit_error=0.
    for cell,record in zip(scores,report['cells']):
        compare(record,{k:v for k,v in cell.items() if k not in ('logits','labels','repeat')})
        with torch.random.fork_rng(devices=[]):check=reference.audit_cell(study.ROOT/cell['checkpoint'],pools[cell['pool']],cell)
        targets+=check['targets'];logit_error=max(logit_error,check['max_logit_error'])
    choice={}
    for arm in ('lbfgs','adam'):
        fits=[];perfects=[]
        for run in [r for r in report['runs'] if r['arm']==arm]:
            for s in run['snapshots']:
                metrics={str(m):next(c['metrics'] for c in scores if (c['seed'],c['arm'],c['step'],c['pool'])==(run['seed'],arm,s['step'],f'train_{m}')) for m in study.LOADS}
                compare(s['training'],metrics)
                fitted=all(metrics[str(m)]['groups']['all']['accuracy']>=.99 for m in study.LOADS)
                perfect=all(metrics[str(m)]['groups']['all']['accuracy']==1 for m in study.LOADS)
                assert s['fit']==fitted and s['perfect']==perfect
            fits.append(run['snapshots'][-1]['fit']);perfects.append(run['snapshots'][-1]['perfect'])
        choice[arm]=dict(all_seed_fit=all(fits),all_seed_perfect=all(perfects))
    compare(report['decision'],choice);compare(json.loads((out/'pretest_decision.json').read_text()),choice)
    assert report['evaluation_targets']==targets
    for field in ('optimizer_calls','answer_targets','histories','input_events'):assert report[field]==sum(r[field] for r in report['runs'])
    assert report['gradient_evaluations']==sum(r['closure_evaluations'] if r['arm']=='lbfgs' else r['optimizer_calls'] for r in report['runs'])
    result=dict(passed=True,report_sha256=replay.digest(out/'report.json'),cells=len(scores),evaluation_targets=targets,
        optimizer_calls=report['optimizer_calls'],gradient_evaluations=report['gradient_evaluations'],answer_targets=report['answer_targets'],
        max_gradient_error=max(c['max_gradient_error'] for c in checks),max_adam_error=max(c.get('max_update_error',0.) for c in checks),
        max_direction_error=max(c.get('max_direction_error',0.) for c in checks),max_logit_error=logit_error,protected_files=len(lock['files']))
    if persist:
        assert not (out/'audit.json').exists();replay.write(out/'audit.json',result)
    return result


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);print(json.dumps(audit(study.OUT)),flush=True)
