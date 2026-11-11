"""Independent matrix replay, forward write basis and scalar cohort reductions."""
import json
import torch
import write_interference_stages as study
import audit_shared_readout_feasibility as proof


@torch.no_grad()
def independent(model,rows,endpoint):
    queries=[];labels=[];positions=[]
    for row in rows:
        mapping={};where={}
        for t,(key,value) in enumerate(row['events']):
            assert key not in mapping,'Repeated key outside contract'
            mapping[key]=value;where[key]=t+1
        assert mapping[row['query']]==row['label'],'Assigned label'
        keys=sorted(mapping);queries.append(keys);labels.append([mapping[k] for k in keys]);positions.append([where[k] for k in keys])
    q=torch.tensor(queries);y=torch.tensor(labels);pos=torch.tensor(positions)
    h,m=q.shape;e=torch.tensor([r['events'] for r in rows]);tmax=e.shape[1];assert tmax==m
    raw=model.key.weight;keys=raw/raw.square().sum(-1,keepdim=True).clamp_min(1e-24).sqrt()
    k=keys[e[...,0]];v=model.value.weight[e[...,1]];beta=model.write_logit.sigmoid();eye=torch.eye(16,dtype=torch.float64)
    state=torch.zeros(h,16,8,dtype=torch.float64);basis=torch.zeros(h,16,tmax,dtype=torch.float64);frames=[]
    for t in range(tmax+1):
        if t:
            a=eye-beta*k[:,t-1,:,None]*k[:,t-1,None,:]
            state=a@state+beta*k[:,t-1,:,None]*v[:,t-1,None,:]
            basis=a@basis;basis[:,:,t-1]=beta*k[:,t-1]
        frames.append((keys[q]@state)@model.output.weight.T+model.output.bias)
    z=torch.stack(frames);coef=keys[q]@basis;i=torch.arange(h)[:,None];j=torch.arange(m)[None,:];post=z[pos,i,j]
    rivals=z[-1].clone();rivals.scatter_(-1,y[...,None],-torch.inf);c=rivals.argmax(-1)
    w=model.output.weight[y]-model.output.weight[c]
    parts=coef*(w@v.transpose(1,2));target=parts.gather(-1,(pos-1)[...,None]).squeeze(-1)
    early=torch.zeros_like(target);late=torch.zeros_like(target)
    for t in range(tmax):
        early+=parts[:,:,t]*(t+1<pos);late+=parts[:,:,t]*(t+1>pos)
    bias=model.output.bias[y]-model.output.bias[c]
    fixedmargin=z[-1].gather(-1,y[...,None]).squeeze(-1)-z[-1].gather(-1,c[...,None]).squeeze(-1)
    torch.testing.assert_close(target+early+late+bias,fixedmargin,atol=1e-9,rtol=1e-9)
    immediate=beta*keys[q].square().sum(-1);atwrite=immediate*(w*v[i,pos-1]).sum(-1)
    good=z.argmax(-1)==y;harm=torch.zeros_like(endpoint);rescue=torch.zeros_like(endpoint)
    for t in range(1,tmax+1):
        harm|=(t>pos)&good[t-1]&~good[t];rescue|=(t>pos)&~good[t-1]&good[t]
    def margin(scores):
        other=scores.clone();other.scatter_(-1,y[...,None],-torch.inf)
        return scores.gather(-1,y[...,None]).squeeze(-1)-other.max(-1).values
    pm=margin(post);fm=margin(z[-1])
    fields=dict(position=pos,competitor=c,post_correct=post.argmax(-1)==y,final_correct=good[-1],later_harm=harm,later_rescue=rescue,
        post_margin=pm,final_margin=fm,margin_change=fm-pm,target_at_write=atwrite,target_final=target,target_change=target-atwrite,
        earlier_contribution=early,later_contribution=late,bias=bias,target_coefficient_ratio=coef.gather(-1,(pos-1)[...,None]).squeeze(-1)/immediate)
    return q,y,z,coef,fields


def reduce(fields,endpoint,seed,step):
    complete=endpoint.all(-1);masks=dict(all=torch.ones_like(endpoint),endpoint_wrong=~endpoint,
        correct_in_incomplete=endpoint&~complete[:,None],correct_in_complete=endpoint&complete[:,None]);result=[]
    for group in ('all','endpoint_wrong','correct_in_incomplete','correct_in_complete'):
        for position in [None,*range(1,int(fields['position'].max())+1)]:
            mask=masks[group].clone()
            if position is not None:mask&=fields['position']==position
            n=int(torch.count_nonzero(mask));r=dict(seed=seed,step=step,group=group,position=position,count=n)
            for name,value in fields.items():
                if name in ('position','competitor'):continue
                x=value.masked_select(mask)
                if x.dtype==torch.bool:r[name+'_count']=int(torch.count_nonzero(x))
                else:
                    r[name+'_mean']=float(x.sum()/n) if n else None
                    if name in ('margin_change','target_change','later_contribution'):
                        r[name+'_negative']=int(torch.count_nonzero(x < -1e-9));r[name+'_positive']=int(torch.count_nonzero(x > 1e-9))
                        r[name+'_unresolved']=n-r[name+'_negative']-r[name+'_positive']
            result.append(r)
    return result


@torch.no_grad()
def audit(out,persist=True):
    prior=study.prior;lock=json.loads((out/'lock.json').read_text());report=json.loads((out/'report.json').read_text());prior.check_files(lock['files'])
    assert report['status']=='complete' and all(report[k]==0 for k in ('training_updates','optimizer_calls','autograd_calls','solver_calls'))
    assert lock['guard']==1e-9 and lock['groups']==list(study.GROUPS) and lock['versions']==prior.versions()
    assert report['lock_sha256']==prior.digest(out/'lock.json') and report['traces_sha256']==prior.digest(out/'traces.pt')
    bundles=torch.load(out/'traces.pt',weights_only=True);assert len(bundles)==len(lock['cases'])
    cached=None if lock['fixture'] else torch.load(study.stages.OUT/'stages.pt',weights_only=True)
    rows_out=[];stages_out=[];queryrows=outputs=writes=0;error=0.
    for case,b in zip(lock['cases'],bundles):
        assert (case['seed'],case['step'])==(b['seed'],b['step'])
        rows=json.loads((study.ROOT/case['data']).read_text(encoding='utf-8-sig'))['pools']['train_32']
        checkpoint=torch.load(study.ROOT/case['checkpoint'],weights_only=True);assert checkpoint['step']==case['step']
        with torch.random.fork_rng(devices=[]):model=prior.historical.parent.previous.load_model(study.ROOT/case['checkpoint']).eval()
        if lock['fixture']:
            _,y,_,_,_,native=proof.independent_features(model,rows);endpoint=native.argmax(-1)==y
        else:
            saved=next(x for x in cached if x['case']['seed']==case['seed'] and x['case']['step']==12000)
            block=next(x for x in saved['source']['blocks'] if x['pool']=='train_32');endpoint=block['native_logits'].argmax(-1)==block['labels']
        assert torch.equal(endpoint,b['endpoint_correct'])
        q,y,z,coef,fields=independent(model,rows,endpoint)
        assert torch.equal(q,b['queries']) and torch.equal(y,b['labels']) and torch.equal(z.argmax(-1),b['logits'].argmax(-1))
        for actual,expected in ((b['logits'],z),(b['coefficients'],coef)):
            torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-9);error=max(error,float((actual-expected).abs().max()))
        assert set(fields)==set(b['fields'])
        for name,expected in fields.items():
            actual=b['fields'][name]
            if expected.dtype in (torch.bool,torch.int64):assert torch.equal(actual,expected),name
            else:torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-9)
        summary=reduce(fields,endpoint,case['seed'],case['step']);proof.ref.compare(b['summary'],summary);rows_out.extend(summary)
        metrics=proof.reduced(z[-1],y,torch.zeros_like(y,dtype=torch.bool));proof.ref.compare(b['metrics'],metrics)
        assert b['beta']==float(model.write_logit.sigmoid())
        stages_out.append(dict(seed=case['seed'],step=case['step'],beta=b['beta'],metrics=metrics))
        if cached is not None:
            saved=next(x for x in cached if x['case']['seed']==case['seed'] and x['case']['step']==case['step'])
            block=next(x for x in saved['source']['blocks'] if x['pool']=='train_32')
            torch.testing.assert_close(z[-1],block['native_logits'],atol=1e-9,rtol=1e-9)
            assert torch.equal(z[-1].argmax(-1),block['native_logits'].argmax(-1))
            proof.ref.compare(metrics,block['native_metrics']);assert y.shape==(768,32) and len(z)==33
        queryrows+=y.numel();outputs+=len(z)*y.numel();writes+=(len(z)-1)*len(y)
    proof.ref.compare(report['summary'],rows_out);proof.ref.compare(report['stages'],stages_out)
    assert (report['query_rows'],report['answer_outputs'],report['replay_writes'])==(queryrows,outputs,writes)
    prior.check_files(lock['files']);result=dict(passed=True,report_sha256=prior.digest(out/'report.json'),stages=len(bundles),
        summary_rows=len(rows_out),query_rows=queryrows,answer_outputs=outputs,replay_writes=writes,max_equation_error=error,protected_files=len(lock['files']),training_updates=0)
    if persist:assert not (out/'audit.json').exists();prior.write(out/'audit.json',result)
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    torch.set_num_threads(1);audit(study.OUT)
