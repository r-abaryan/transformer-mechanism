"""Independent scalar radial derivatives and matrix head-gradient projections."""
import json
import math

import torch
import audit_feedback_loss_decomposition as prior
import audit_readout_scale_control as head_audit
import readout_scale_pressure as study


def scalar(z,y):
    high=max(z);weights=[math.exp(v-high) for v in z];den=math.fsum(weights)
    return dict(ce=high-z[y]+math.log(den),g=math.fsum(w*(v-z[y]) for w,v in zip(weights,z))/den,
        correct=z.index(high)==y)


def reconstruct(cell):
    logits=cell['logits'].reshape(-1,6).tolist();ys=cell['labels'].flatten().tolist();rs=cell['repeat'].flatten().tolist()
    assert len(logits)==len(ys)==len(rs)
    values=[dict(**scalar(z,y),repeat=r) for z,y,r in zip(logits,ys,rs)];groups={}
    for group in ('all','unique','repeat'):
        scope=[v for v in values if group=='all' or v['repeat']==(group=='repeat')];n=len(scope);parts={}
        for name,correct in (('correct',True),('error',False)):
            chosen=[v for v in scope if v['correct']==correct];count=len(chosen)
            parts[name]=dict(count=count,ce=math.fsum(v['ce'] for v in chosen)/count if count else None,
                mean_g=math.fsum(v['g'] for v in chosen)/count if count else None,
                contribution=math.fsum(v['g'] for v in chosen)/n if n else 0.)
        groups[group]=dict(count=n,accuracy=sum(v['correct'] for v in scope)/n if n else None,
            ce=math.fsum(v['ce'] for v in scope)/n if n else None,
            mean_g=math.fsum(v['g'] for v in scope)/n if n else None,partitions=parts)
    return groups


def matrix_gradient(path,rows,cell,batch=64):
    events,q,y,repeat=prior.previous.independent.inputs(rows,'all_queries')
    assert torch.equal(y,cell['labels']) and torch.equal(repeat,cell['repeat']),'Wrong labels/order'
    model=study.training.previous.load_model(path)
    before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('output.'))
    parameters=(model.output.weight,model.output.bias);grads=[torch.zeros_like(p) for p in parameters];zs=[]
    for i in range(0,len(rows),batch):
        z,_=prior.previous.independent.forward_tangent(model,events[i:i+batch],q[i:i+batch])
        logp=z-z.logsumexp(-1,keepdim=True)
        loss=-logp.gather(-1,y[i:i+batch,...,None]).sum()/y.numel()
        for j,g in enumerate(torch.autograd.grad(loss,parameters)):grads[j]+=g.detach()
        zs.append(z.detach())
    actual=torch.cat(zs);error=float((actual-cell['logits']).abs().max())
    assert error<=1e-9 and torch.equal(actual.argmax(-1),cell['logits'].argmax(-1)),'Matrix logits disagree'
    radial=math.fsum(float((g*(p.detach()-p.detach().mean(0,keepdim=True))).sum()) for g,p in zip(grads,parameters))
    norm=head_audit.norm(model.state_dict())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()) and torch.equal(rng,torch.get_rng_state())
    return dict(radial_g=radial,head_norm=norm,max_logit_error=error)


def audit_cell(path,rows,source,record):
    expected=reconstruct(source);error=prior.compare(record['groups'],expected)
    matrix=matrix_gradient(path,rows,source);g=expected['all']['mean_g']
    for value in (matrix['radial_g'],record['head_gradient_g'],record['autograd_g']):
        delta=abs(value-g);assert delta<=1e-9+1e-7*abs(g),'Head gradient disagreement';error=max(error,delta)
    assert abs(record['head_norm']-matrix['head_norm'])<=1e-9
    for key in ('0.0001','5e-05'):
        assert abs(record['finite_differences'][key]-g)<=1e-8+1e-5*abs(g),'Finite difference disagreement'
    return dict(max_scalar_error=error,max_matrix_logit_error=matrix['max_logit_error'])


def objective(records):
    assert len(records)==4 and {r['load'] for r in records}=={2,8,16,32}
    return dict(ce=math.fsum(r['groups']['all']['ce'] for r in records)/4,
        mean_g=math.fsum(r['groups']['all']['mean_g'] for r in records)/4,
        correct_contribution=math.fsum(r['groups']['all']['partitions']['correct']['contribution'] for r in records)/4,
        error_contribution=math.fsum(r['groups']['all']['partitions']['error']['contribution'] for r in records)/4)


def main():
    torch.set_num_threads(1);out=study.OUT;replay=study.training.replay
    report=json.loads((out/'report.json').read_text());lock=json.loads((out/'lock.json').read_text())
    replay.check_files(lock['files']);assert report['lock_sha256']==replay.digest(out/'lock.json')
    assert report['status']=='complete' and report['training_updates']==0 and report['split']=='train'
    pairs=study.inputs();ids=[(s,t,m) for s in (3461,3463) for t in range(3000,12001,1000) for m in (2,8,16,32)]
    assert [(r['seed'],r['step'],r['load']) for r in report['cells']]==ids and len(pairs)==80
    errors=[]
    for (cell,rows),record in zip(pairs,report['cells']):
        assert (record['seed'],record['step'],record['load'])==(cell['seed'],cell['step'],int(cell['pool'].split('_')[1]))
        assert record['checkpoint']==cell['checkpoint']
        cp=torch.load(study.ROOT/cell['checkpoint'],weights_only=True);assert cp['step']==cell['step']
        assert record['max_logit_error']<=1e-9
        errors.append(audit_cell(study.ROOT/cell['checkpoint'],rows,cell,record))
    assert [(r['seed'],r['step']) for r in report['objectives']]==[(s,t) for s in (3461,3463) for t in range(3000,12001,1000)]
    for record in report['objectives']:
        cells=[c for c in report['cells'] if (c['seed'],c['step'])==(record['seed'],record['step'])]
        assert len({c['head_norm'] for c in cells})==1 and record['head_norm']==cells[0]['head_norm']
        prior.compare({k:record[k] for k in ('ce','mean_g','correct_contribution','error_contribution')},objective(cells))
    assert report['cells_count']==80 and report['evaluated_answers']==sum(c['labels'].numel() for c,rows in pairs)==890880
    replay.check_files(lock['files'])
    result=dict(passed=True,report_sha256=replay.digest(out/'report.json'),cells=80,objectives=20,
        evaluated_answers=890880,training_updates=0,max_scalar_error=max(e['max_scalar_error'] for e in errors),
        max_matrix_logit_error=max(e['max_matrix_logit_error'] for e in errors),protected_files=len(lock['files']))
    path=out/'audit.json';assert not path.exists();replay.write(path,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
