"""Independent matrix error gradients/conditions and scalar Adam projections."""
import json
import math

import torch
import adam_memory_error_transport as study
import audit_readout_adam_transport as adam_audit
import audit_query_feedback_training as reference
import audit_feedback_loss_decomposition as prior


def fixed(z,y,mask):
    values=[prior.scalar(score,label) for score,label in zip(z.reshape(-1,6).tolist(),y.flatten().tolist())]
    errors=mask.flatten().tolist();n=sum(errors)
    return dict(error_count=n,error_ce=math.fsum(v['ce'] for v,m in zip(values,errors) if m)/n if n else None,
        originally_wrong_now_correct=sum(m and v['correct'] for v,m in zip(values,errors)),
        originally_correct_now_wrong=sum(not m and not v['correct'] for v,m in zip(values,errors)))


def scalar_parts(trace):
    _,error=adam_audit.scalar_adam(trace)
    group=trace['before_optimizer']['param_groups'][0];b1,b2=group['betas'];lr=group['lr'];eps=group['eps']
    parts={p:{} for p in ('actual','stored','current','sgd')}
    for n,pid in zip(trace['names'],group['params']):
        shape=trace['before_model'][n].shape;old=trace['before_optimizer']['state'].get(pid,{})
        new=trace['after_optimizer']['state'][pid];step=int(new['step'])
        m=old['exp_avg'].flatten().tolist() if old else [0.]*trace['gradients'][n].numel()
        g=trace['gradients'][n].flatten().tolist();v=new['exp_avg_sq'].flatten().tolist()
        actual=[b-a for a,b in zip(trace['before_model'][n].flatten().tolist(),trace['after_model'][n].flatten().tolist())]
        stored=[];current=[]
        for a,b,c in zip(m,g,v):
            den=math.sqrt(c/(1-b2**step))+eps
            stored.append(-lr*b1*a/((1-b1**step)*den));current.append(-lr*(1-b1)*b/((1-b1**step)*den))
        for p,values in (('actual',actual),('stored',stored),('current',current),('sgd',[-lr*x for x in g])):
            parts[p][n]=torch.tensor(values,dtype=torch.float64).reshape(shape)
    return parts,error


def scalar_projections(gradient,parts):
    records={}
    for block,names in study.BLOCKS.items():
        values={p:math.fsum(x*y for n in names for x,y in zip(gradient[n].flatten().tolist(),d[n].flatten().tolist())) for p,d in parts.items()}
        def sign(v):return 'unresolved' if abs(v)<=1e-10 else ('higher_error_ce' if v>0 else 'lower_error_ce')
        records[block]=dict(**values,actual_direction=sign(values['actual']),sgd_direction=sign(values['sgd']))
        assert abs(values['actual']-values['stored']-values['current'])<=1e-11
    return records


def audit_cell(trace,rows,saved,record,batch=128):
    e,q,y,repeat=reference.independent.inputs(rows,'all_queries');before=saved['conditions']['before']
    assert torch.equal(before['labels'],y) and torch.equal(before['repeat'],repeat), 'Labels/order disagree'
    states={'before':trace['before_model'],
        'memory_after':{n:trace['after_model'][n] if n in study.MEMORY else v for n,v in trace['before_model'].items()},
        'full_after':trace['after_model']}
    gradients={n:torch.zeros_like(v) for n,v in trace['before_model'].items()};zs={};max_logit=0.
    for name,state in states.items():
        model=study.model_from(state);chunks=[]
        with torch.no_grad():
            for i in range(0,len(rows),batch):chunks.append(reference.independent.forward_tangent(model,e[i:i+batch],q[i:i+batch])[0])
        z=torch.cat(chunks);zs[name]=z;data=saved['conditions'][name]
        assert torch.equal(data['labels'],y) and torch.equal(data['repeat'],repeat), 'Condition labels disagree'
        max_logit=max(max_logit,reference.independent.close(data['logits'],z));assert torch.equal(data['logits'].argmax(-1),z.argmax(-1))
        metrics=reference.metrics(z,y,repeat);reference.check_metrics(data['metrics'],metrics);reference.check_metrics(record['conditions'][name]['metrics'],metrics)
    mask=zs['before'].argmax(-1)!=y;assert torch.equal(before['error_mask'],mask), 'Error mask disagrees'
    count=int(mask.sum());model=study.model_from(trace['before_model']);params=dict(model.named_parameters())
    if count:
        for i in range(0,len(rows),batch):
            z,_=reference.independent.forward_tangent(model,e[i:i+batch],q[i:i+batch])
            losses=-(z-z.logsumexp(-1,keepdim=True)).gather(-1,y[i:i+batch,...,None]).squeeze(-1)
            gs=torch.autograd.grad(losses[mask[i:i+batch]].sum()/count,tuple(params.values()))
            for n,g in zip(params,gs):gradients[n]+=g.detach()
    max_gradient=0.
    for n,g in gradients.items():max_gradient=max(max_gradient,reference.independent.close(before['gradients'][n],g,gradient=True))
    for name,z in zs.items():
        expected=fixed(z,y,mask);prior.compare(saved['conditions'][name]['fixed'],expected);prior.compare(record['conditions'][name]['fixed'],expected)
    parts,adam_error=scalar_parts(trace)
    for p,d in parts.items():
        for n,v in d.items():assert float((saved['parts'][p][n]-v).abs().max())<=1e-11, 'Adam movement term disagrees'
    projections=scalar_projections(gradients,parts);scalar_error=prior.compare(record['blocks'],projections)
    probe=record['probe'];norm=math.sqrt(math.fsum(x*x for n in study.MEMORY for x in parts['actual'][n].flatten().tolist()))
    assert abs(probe['norm']-norm)<=1e-12 and probe['skipped']==(norm<=1e-10 or count==0)
    probe_error=0.;probe_answers=0
    if not probe['skipped']:
        expected=projections['memory']['actual']/norm
        assert abs(probe['analytic']-expected)<=1e-9+1e-7*abs(expected),'Probe projection/sign disagrees'
        assert [c['h'] for c in probe['checks']]==[1e-5,5e-6]
        for c in probe['checks']:
            derivative=(c['plus_loss']-c['minus_loss'])/(2*c['h']);assert derivative==c['derivative']
            delta=abs(derivative-expected);assert delta<=1e-7+1e-4*abs(expected),'Probe slope disagrees'
            probe_error=max(probe_error,delta)
        probe_answers=4*y.numel()
    return dict(max_logit_error=max_logit,max_gradient_error=max_gradient,max_adam_error=adam_error,
        max_scalar_error=scalar_error,max_probe_error=probe_error,probe_answers=probe_answers)


def main():
    torch.set_num_threads(1);out=study.OUT;replay=study.training.replay
    report=json.loads((out/'report.json').read_text());lock=json.loads((out/'lock.json').read_text());replay.check_files(lock['files'])
    assert report['lock_sha256']==replay.digest(out/'lock.json') and report['scores_sha256']==replay.digest(out/'scores.pt')
    assert report['status']=='complete' and report['training_updates']==report['optimizer_calls']==0
    assert report['pool']=='train_32' and report['device']=='cpu' and report['dtype']=='float64'
    source_report=json.loads((study.replay_source.OUT/'report.json').read_text())
    assert source_report['traces_sha256']==replay.digest(study.replay_source.OUT/'traces.pt')
    traces,rows=study.inputs();saved=torch.load(out/'scores.pt',weights_only=True)
    ids=[(s,a,t) for s in (3461,3463) for a in (3000,11000) for t in range(a+1,a+5)]
    assert [(r['seed'],r['start'],r['step']) for r in report['cells']]==ids
    assert len(rows)==768 and all(r['load']==32 and len(r['events'])==32 for r in rows)
    errors=[]
    assert len(traces)==len(saved)==len(report['cells'])==report['reused_replay_steps']==16
    for trace,data,record in zip(traces,saved,report['cells']):
        assert data['record']==record
        for k in ('seed','start','step','load'):assert record[k]==trace[k]
        errors.append(audit_cell(trace,rows,data,record))
    assert sum(d['conditions']['before']['labels'].numel() for d in saved)==report['baseline_answers']==393216
    assert report['condition_answers']==3*report['baseline_answers']==1179648
    assert sum(e['probe_answers'] for e in errors)==report['probe_answers']<=1572864
    replay.check_files(lock['files']);result=dict(passed=True,report_sha256=replay.digest(out/'report.json'),cells=16,
        baseline_answers=393216,condition_answers=1179648,probe_answers=report['probe_answers'],
        **{k:max(e[k] for e in errors) for k in errors[0] if k!='probe_answers'},protected_files=len(lock['files']),optimizer_calls=0)
    p=out/'audit.json';assert not p.exists();replay.write(p,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
