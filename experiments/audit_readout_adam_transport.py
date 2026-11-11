"""Independent streams, matrix gradients and scalar Adam for every coordinate."""
import copy
import json
import math

import torch
import audit_readout_scale_pressure as objective
import readout_adam_transport as study


def scalar_adam(trace):
    before=trace['before_model'];after=trace['after_model'];old=trace['before_optimizer'];new=trace['after_optimizer']
    group=old['param_groups'][0];assert group==new['param_groups'][0]
    assert group['weight_decay']==0 and not group['amsgrad'] and not group['maximize']
    b1,b2=group['betas'];lr=group['lr'];eps=group['eps'];stored={};current={};errors=[]
    assert set(trace['names'])==before.keys()==after.keys()==trace['gradients'].keys()
    for name,pid in zip(trace['names'],group['params']):
        p=before[name];g=trace['gradients'][name];initial=old['state'].get(pid,{})
        t=int(initial['step'])+1 if initial else 1
        assert int(new['state'][pid]['step'])==t
        means=initial['exp_avg'].flatten().tolist() if initial else [0.]*p.numel()
        squares=initial['exp_avg_sq'].flatten().tolist() if initial else [0.]*p.numel()
        expected_m=[];expected_v=[];expected_p=[];ds=[];dc=[]
        for value,gradient,m,v in zip(p.flatten().tolist(),g.flatten().tolist(),means,squares):
            first=b1*m+(1-b1)*gradient;second=b2*v+(1-b2)*gradient*gradient
            den=math.sqrt(second/(1-b2**t))+eps
            a=-lr*b1*m/((1-b1**t)*den);b=-lr*(1-b1)*gradient/((1-b1**t)*den)
            expected_m.append(first);expected_v.append(second);expected_p.append(value+a+b);ds.append(a);dc.append(b)
        for values,actual in ((expected_m,new['state'][pid]['exp_avg']),
                              (expected_v,new['state'][pid]['exp_avg_sq']),(expected_p,after[name])):
            error=max(abs(a-b) for a,b in zip(values,actual.flatten().tolist()));assert error<=1e-11,'Scalar Adam disagreement';errors.append(error)
        if name.startswith('output.'):
            stored[name]=ds;current[name]=dc
    def rows(state):
        return [w+[b] for w,b in zip(state['output.weight'].tolist(),state['output.bias'].tolist())]
    a=rows(before);b=rows(after);width=len(a[0]);ma=[math.fsum(r[j] for r in a)/6 for j in range(width)]
    mb=[math.fsum(r[j] for r in b)/6 for j in range(width)]
    c=[[v-ma[j] for j,v in enumerate(r)] for r in a]
    d=[[(b[i][j]-mb[j])-c[i][j] for j in range(width)] for i in range(6)]
    norm2=math.fsum(x*x for r in c for x in r);norm=math.sqrt(norm2)
    def head(values):
        ws=values['output.weight'];bs=values['output.bias'];w=width-1
        return [ws[i*w:(i+1)*w]+[bs[i]] for i in range(6)]
    def dot(values):return math.fsum(x*y for rx,ry in zip(c,values) for x,y in zip(rx,ry))
    raw=dot(rows(trace['gradients']));rho=dot(d)/norm2
    new_norm=math.sqrt(math.fsum((b[i][j]-mb[j])**2 for i in range(6) for j in range(width)))
    perpendicular=math.sqrt(math.fsum((d[i][j]-rho*c[i][j])**2 for i in range(6) for j in range(width)))/norm
    identity=new_norm**2-norm2-(2*dot(d)+math.fsum(x*x for r in d for x in r))
    result=dict(raw_g=raw,sgd_rho=-lr*raw/norm2,adam_rho=rho,stored_rho=dot(head(stored))/norm2,
        current_rho=dot(head(current))/norm2,before_norm=norm,after_norm=new_norm,
        relative_norm_change=new_norm/norm-1,perpendicular_ratio=perpendicular,norm_identity_error=identity)
    assert abs(result['stored_rho']+result['current_rho']-rho)<=1e-11 and abs(identity)<=1e-9
    return result,max(errors)


def matrix_gradients(trace,rows):
    e,q,y,repeat=objective.prior.previous.independent.inputs(rows,'all_queries')
    assert torch.equal(q,trace['queries']) and torch.equal(y,trace['labels']) and torch.equal(repeat,trace['repeat']),'Wrong labels/order'
    with torch.random.fork_rng():
        model=study.training.model_source.LoadReader('delta');model.load_state_dict(trace['before_model'])
        assert list(dict(model.named_parameters()))==trace['names']
        z,_=objective.prior.previous.independent.forward_tangent(model,e,q)
        logp=z-z.logsumexp(-1,keepdim=True);loss=-logp.gather(-1,y[...,None]).mean()
        grads=torch.autograd.grad(loss,tuple(model.parameters()))
    error=float((z.detach()-trace['logits']).abs().max());assert error<=1e-9,'Matrix logits disagree'
    assert torch.equal(z.argmax(-1),trace['logits'].argmax(-1))
    gradient_errors=[]
    for name,g in zip(trace['names'],grads):
        actual=trace['gradients'][name];delta=(actual-g).abs()
        assert bool((delta<=1e-9+1e-7*g.abs()).all()),'Matrix gradient disagreement'
        gradient_errors.append(float(delta.max()))
    return max(gradient_errors),error


def audit_cycle(path,pools,traces,original=None):
    checkpoint=torch.load(path,weights_only=True);start=checkpoint['step'];seed=traces[0]['seed']
    assert len(traces)==4 and [t['step'] for t in traces]==list(range(start+1,start+5))
    assert study.training.replay.equal(traces[0]['before_model'],checkpoint['model'])
    assert study.training.replay.equal(traces[0]['before_optimizer'],checkpoint['optimizer'])
    hr=torch.Generator().manual_seed(seed+10000);qr=torch.Generator().manual_seed(seed+20000)
    for step in range(1,start+1):
        load=(2,8,16,32)[(step-1)%4]
        torch.randint(len(pools[f'train_{load}']),(64,),generator=hr)
        torch.randint(load,(64,load),generator=qr)
    assert torch.equal(hr.get_state(),checkpoint['history_rng']) and torch.equal(qr.get_state(),checkpoint['query_rng']),'Wrong archived RNG'
    checks=[]
    for i,t in enumerate(traces):
        load=(2,8,16,32)[(t['step']-1)%4];assert t['load']==load and t['start']==start and t['seed']==seed
        assert torch.equal(t['history_before'],hr.get_state()) and torch.equal(t['query_before'],qr.get_state()),'Wrong stream before'
        indices=torch.randint(len(pools[f'train_{load}']),(64,),generator=hr);draws=torch.randint(load,(64,load),generator=qr)
        assert torch.equal(indices,t['indices']) and torch.equal(draws,t['draws']),'Wrong sampled stream'
        assert torch.equal(t['history_after'],hr.get_state()) and torch.equal(t['query_after'],qr.get_state()),'Wrong stream after'
        if i:
            assert study.training.replay.equal(t['before_model'],traces[i-1]['after_model'])
            assert study.training.replay.equal(t['before_optimizer'],traces[i-1]['after_optimizer'])
        rows=[pools[f'train_{load}'][int(j)] for j in indices]
        gradient_error,logit_error=matrix_gradients(t,rows);expected,adam_error=scalar_adam(t)
        objective.prior.compare({k:t['record'][k] for k in expected},expected)
        groups=objective.reconstruct(dict(logits=t['logits'],labels=t['labels'],repeat=t['repeat']))
        objective.prior.compare(t['record']['groups'],groups)
        assert abs(t['record']['loss']-groups['all']['ce'])<=1e-9
        assert abs(expected['raw_g']-groups['all']['mean_g'])<=1e-9
        if original is not None:
            pid=t['before_optimizer']['param_groups'][0]['params'][0];state=t['after_optimizer']['state'][pid]
            before=t['before_model']['write_logit'];after=t['after_model']['write_logit']
            beta=dict(step=t['step'],load=load,alpha_before=float(before),alpha_after=float(after),
                beta_before=float(before.sigmoid()),beta_after=float(after.sigmoid()),gradient=float(t['gradients']['write_logit']),
                exp_avg=float(state['exp_avg']),exp_avg_sq=float(state['exp_avg_sq']))
            assert beta==original[i],'Historical beta log mismatch'
        checks.append(dict(max_gradient_error=gradient_error,max_logit_error=logit_error,max_adam_error=adam_error))
    return checks


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);out=study.OUT;replay=study.training.replay
    report=json.loads((out/'report.json').read_text());lock=json.loads((out/'lock.json').read_text())
    replay.check_files(lock['files']);assert report['status']=='complete'
    assert report['lock_sha256']==replay.digest(out/'lock.json') and report['traces_sha256']==replay.digest(out/'traces.pt')
    traces=torch.load(out/'traces.pt',weights_only=True);assert report['cells']==[t['record'] for t in traces]
    pools,cycles=study.inputs();checks=[]
    assert report['cycles']==[dict(seed=s,start=t,checkpoint=p.relative_to(study.ROOT).as_posix()) for s,t,p,original in cycles]
    assert [(t['seed'],t['start'],t['step']) for t in traces]==[(s,start,step) for s in (3461,3463) for start in (3000,11000) for step in range(start+1,start+5)]
    for seed,start,path,original in cycles:
        selected=[t for t in traces if (t['seed'],t['start'])==(seed,start)]
        checks.extend(audit_cycle(path,pools,selected,original))
    assert report['replayed_optimizer_calls']==16 and report['new_training_budget_updates']==0
    assert report['answer_targets']==sum(t['labels'].numel() for t in traces)==14848
    assert report['histories']==sum(len(t['indices']) for t in traces)==1024 and report['input_events']==32768
    replay.check_files(lock['files'])
    result=dict(passed=True,report_sha256=replay.digest(out/'report.json'),replayed_optimizer_calls=16,new_training_budget_updates=0,
        parameter_coordinates=16*615,answer_targets=14848,historical_beta_logs_exact=True,
        max_gradient_error=max(c['max_gradient_error'] for c in checks),max_logit_error=max(c['max_logit_error'] for c in checks),
        max_adam_error=max(c['max_adam_error'] for c in checks),protected_files=len(lock['files']))
    path=out/'audit.json';assert not path.exists();replay.write(path,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
