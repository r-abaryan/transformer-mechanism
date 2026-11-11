"""Forward basis effects, reverse matrix gradients and chronological cross terms."""
import csv
import json
import torch
import single_query_write_opposition as study
import audit_shared_state_feasibility as ref


@torch.no_grad()
def independent(model,rows,bundle):
    e=torch.tensor([r['events'] for r in rows]);ids=e[...,1];orders=[];ys=[]
    for row in rows:
        mapping={}
        for key,value in row['events']:
            assert key not in mapping or mapping[key]==value;mapping[key]=value
        assert mapping[row['query']]==row['label'];order=sorted(mapping);orders.append(order);ys.append([mapping[q] for q in order])
    queries=torch.tensor(orders);labels=torch.tensor(ys)
    assert torch.equal(queries,bundle['queries']) and torch.equal(labels,bundle['labels'])
    key=model.key.weight.detach();key=key/key.square().sum(-1).sqrt().clamp_min(1e-12)[:,None]
    k=key[e[...,0]];q=key[queries];beta=float(model.write_logit.sigmoid());h,m,dkey=q.shape;t=e.shape[1]
    # Forward propagation of all injected write directions, rather than reverse q.
    effects=torch.zeros(h,t,dkey,dtype=torch.float64)
    for step in range(t):
        dot=(effects*k[:,step,None]).sum(-1)
        effects-=beta*dot[...,None]*k[:,step,None];effects[:,step]=beta*k[:,step]
    coefficients=q@effects.transpose(1,2)
    z=(coefficients@model.value.weight.detach()[ids])@model.output.weight.detach().T+model.output.bias.detach()
    torch.testing.assert_close(z,bundle['logits'],atol=1e-9,rtol=1e-9)
    good=z.argmax(-1)==labels;masks=dict(correct=good,error=~good,total=torch.ones_like(good))
    assert masks.keys()==bundle['masks'].keys()
    for name in masks:assert torch.equal(masks[name],bundle['masks'][name])
    probabilities=torch.exp(z-torch.logsumexp(z,dim=-1,keepdim=True));probabilities.scatter_add_(-1,labels[...,None],-torch.ones_like(labels[...,None],dtype=torch.float64))
    read_grad=(probabilities@model.output.weight.detach())/(4*labels.numel())
    state_grad=q[:,:,:,None]*read_grad[:,:,None,:];reversed_events=[]
    for step in reversed(range(t)):
        contracted=torch.einsum('hd,hmdv->hmv',k[:,step],state_grad)
        reversed_events.append(beta*contracted)
        state_grad-=beta*k[:,step,None,:,None]*contracted[:,:,None,:]
    g=torch.stack(list(reversed(reversed_events)),dim=2)
    for name,mask in masks.items():torch.testing.assert_close((g*mask[:,:,None,None]).sum(1),bundle['gradients'][name]['events'],atol=1e-12,rtol=1e-12)
    prefix=torch.zeros(h,m,6,g.shape[-1],dtype=torch.float64);coef_prefix=torch.zeros(h,m,6,dtype=torch.float64)
    energy=torch.zeros(h,m,dtype=torch.float64);within=torch.zeros_like(energy);geometry=torch.zeros_like(energy)
    pos=torch.zeros(h,m,6,dtype=torch.bool);neg=pos.clone();i=torch.arange(h)[:,None];j=torch.arange(m)[None,:]
    mixed={name:0. for name in masks};cross={name:0. for name in masks}
    for step in range(t):
        value=ids[:,step,None];current=g[:,:,step];past=prefix[i,j,value];a=coefficients[:,:,step];past_a=coef_prefix[i,j,value]
        energy+=current.square().sum(-1);within+=2*(current*past).sum(-1);geometry+=2*a*past_a
        for name,mask in masks.items():
            product=((current*mask[:,:,None]).sum(1)*(past*mask[:,:,None]).sum(1)).sum()
            single=(current*past*mask[:,:,None]).sum();mixed[name]+=2*float(product);cross[name]+=2*float(product-single)
        prefix[i,j,value]+=current;coef_prefix[i,j,value]+=a
        pos[i,j,value]|=a>1e-12;neg[i,j,value]|=a < -1e-12
    joint=prefix.square().sum((2,3));opposite=(pos&neg).any(-1);records=[]
    for a in range(h):
        for b in range(m):
            w=float(within[a,b]);geo=float(geometry[a,b])
            records.append(dict(seed=bundle['seed'],history=a,query=int(queries[a,b]),slot=b,target=int(labels[a,b]),
                correct=bool(good[a,b]),event_energy=float(energy[a,b]),joint_energy=float(joint[a,b]),within=w,
                sign='negative' if w < -1e-14 else 'positive' if w > 1e-14 else 'unresolved',geometry=geo,
                geometry_sign='negative' if geo < -1e-12 else 'positive' if geo > 1e-12 else 'unresolved',
                opposite_same_value=bool(opposite[a,b])))
    summaries=[]
    for name,mask in masks.items():
        selected=[r for r in records if bool(mask[r['history'],r['slot']])]
        s=dict(seed=bundle['seed'],group=name,queries=len(selected),individual_event_energy=float(energy[mask].sum()),
            single_query_within=float(within[mask].sum()),mixed_query_within=mixed[name],cross_query_within=cross[name],
            opposite_same_value=sum(r['opposite_same_value'] for r in selected))
        for label in ('negative','positive','unresolved'):
            s['weighted_'+label]=len([r for r in selected if r['sign']==label]);s['geometry_'+label]=len([r for r in selected if r['geometry_sign']==label])
        summaries.append(s)
    examples=[]
    for row in records:
        if row['correct'] or row['sign']!='negative':continue
        a,b=row['history'],row['slot'];pair=next(((u,v) for u in range(t) for v in range(u+1,t)
            if ids[a,u]==ids[a,v] and float(coefficients[a,b,u]*coefficients[a,b,v]) < -1e-12),None)
        if pair:
            u,v=pair;examples=[dict(seed=bundle['seed'],history=a,query=row['query'],target=row['target'],winner=int(z[a,b].argmax()),
                within=row['within'],geometry=row['geometry'],first_step=u+1,second_step=v+1,value_id=int(ids[a,u]),
                first_coefficient=float(coefficients[a,b,u]),second_coefficient=float(coefficients[a,b,v]),
                gradient_dot=float(g[a,b,u]@g[a,b,v]))];break
    return coefficients,read_grad,g,records,summaries,examples


def audit(out,persist=True):
    report=json.loads((out/'report.json').read_text());lock=json.loads((out/'lock.json').read_text())
    for field,name in (('lock_sha256','lock.json'),('factors_sha256','factors.pt'),('queries_sha256','queries.csv')):
        assert report[field]==study.primary.prior.digest(out/name)
    study.primary.prior.check_files(lock['files']);source=study.ROOT/lock['source'];up_lock,_,bundles=study.history.load(source)
    assert lock['cases']==up_lock['cases'];factors=torch.load(out/'factors.pt',weights_only=True);assert len(factors)==len(bundles)
    records=[];summaries=[];examples=[];sensitivities=0;maximum=0.
    for case,bundle,factor in zip(lock['cases'],bundles,factors):
        assert factor['seed']==bundle['seed']==case['seed']
        rows=json.loads((study.ROOT/case['data']).read_text(encoding='utf-8-sig'))['pools'][case['pool']]
        model=study.primary.prior.historical.parent.previous.load_model(study.ROOT/case['checkpoint']).eval()
        a,d,g,r,s,example=independent(model,rows,bundle)
        for field,expected in (('coefficients',a),('read_gradient',d)):
            actual=factor[field];assert actual.shape==expected.shape;maximum=max(maximum,float((actual-expected).abs().max()))
            torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        for field in ('queries','labels'):assert torch.equal(factor[field],bundle[field])
        assert factor['masks'].keys()==bundle['masks'].keys()
        for name in factor['masks']:assert torch.equal(factor['masks'][name],bundle['masks'][name])
        torch.testing.assert_close(factor['coefficients'][...,None]*factor['read_gradient'][:,:,None],g,atol=1e-12,rtol=1e-12)
        records.extend(r);summaries.extend(s);examples.extend(example);sensitivities+=a.numel()
    ref.compare(report['summary'],summaries);ref.compare(report['examples'],examples)
    if not lock['fixture']:
        old=json.loads((study.history.OUT/'report.json').read_text())
        for current,previous in zip(summaries,old['summary']):
            assert (current['seed'],current['group'])==(previous['seed'],previous['group'])
            ref.close(current['mixed_query_within'],previous['within'],scale=1e-12)
        assert (len(records),sensitivities,len(summaries))==(2*768*32,2*768*32*32,2*3)
    with (out/'queries.csv').open(newline='',encoding='utf-8') as f:actual=list(csv.DictReader(f))
    assert len(actual)==len(records)==report['query_rows']
    for row,expected in zip(actual,records):
        assert row.keys()==expected.keys()
        for key,value in expected.items():
            if isinstance(value,float):ref.close(float(row[key]),value,scale=1e-12)
            else:assert row[key]==str(value),key
    assert report['query_event_sensitivities']==sensitivities and report['cohorts']==len(summaries)
    assert report['coefficient_logit_reconstructions']==len(factors)
    assert report['recurrent_prefills']==report['autograd_calls']==report['optimizer_calls']==report['training_updates']==0
    study.primary.prior.check_files(lock['files']);result=dict(passed=True,report_sha256=study.primary.prior.digest(out/'report.json'),
        cohorts=len(summaries),query_rows=len(records),query_event_sensitivities=sensitivities,
        max_factor_error=maximum,protected_files=len(lock['files']))
    if persist:
        assert not (out/'audit.json').exists();study.primary.prior.write(out/'audit.json',result)
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':torch.set_num_threads(1);audit(study.OUT)
