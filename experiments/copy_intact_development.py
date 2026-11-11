"""Observe intact copy checkpoints; never modify activations or parameters."""
import json
import math
import uuid
import torch
from copy_learning_trajectory import ROOT,PARENT,MODELS,summarize,audit_logits
from copy_position_sites import make_model,forward,digest,write
from copy_rope_model import matrix_rotation

OUT=ROOT/'results/copy_intact_development'
SNAP=ROOT/'results/copy_learning_trajectory'


def check_cross(actual,reconstructed):
    error=float((actual-reconstructed).abs().max());assert error<=1e-9;return error


def trace(model,rows):
    x=torch.tensor(rows);prefix=torch.tensor([[1]+r[:4] for r in rows]);y=x[:,4]
    d=model.base.decoder;cache={};hooks=[]
    def post(name):return lambda m,a,o:cache.update({name:o.detach().clone()})
    def pre(name):return lambda m,a:cache.update({name:a[0].detach().clone()})
    for module,name in ((model.base.encoder.norm2,'memory'),(d.norm1,'pre_cross'),(d.norm2,'cross_norm'),(d.linear2,'ffn'),(d.norm3,'final')):
        hooks.append(module.register_forward_hook(post(name)))
    for module,name in ((d.norm2,'cross_residual'),(d.norm3,'ffn_residual')):hooks.append(module.register_forward_pre_hook(pre(name)))
    try:
        with torch.no_grad():z=forward(model,x,prefix)[0][:,-1]
    finally:
        for hook in hooks:hook.remove()
    with torch.no_grad():
        rz=forward(model,x,prefix,reference=True)[0][:,-1];error=float((z-rz).abs().max());assert error<=1e-9 and torch.equal(z.argmax(-1),rz.argmax(-1))
        layer=d.multihead_attn;mem=cache['memory'];qinput=cache['pre_cross'];h=layer.num_heads;w=layer.embed_dim//h
        q,k,v=[(a@ww.T+b).reshape(len(a),a.shape[1],h,w).transpose(1,2) for a,ww,b in zip((qinput,mem,mem),layer.in_proj_weight.chunk(3),layer.in_proj_bias.chunk(3))]
        s=matrix_rotation(q,torch.arange(5))@matrix_rotation(k,torch.arange(mem.shape[1])).transpose(-1,-2)/math.sqrt(w)
        a=s.softmax(-1);mixed=(a@v).transpose(1,2).reshape_as(qinput)@layer.out_proj.weight.T+layer.out_proj.bias
        error=max(error,check_cross(cache['cross_residual'],qinput+mixed))
        cache['cross']=mixed
        assert torch.allclose(cache['ffn_residual'],cache['cross_norm']+cache['ffn'],atol=1e-9,rtol=0)
        cache={name:(t if name=='memory' else t[:,-1]) for name,t in cache.items()}
        cache.update(logits=z,labels=y,attention=a[:,:,-1],values=v,
            readout=model.base.output.weight[2]-model.base.output.weight[y])
        cache['projected_values']=torch.stack([v[:,j]@layer.out_proj.weight[:,j*w:(j+1)*w].T for j in range(h)],1)
    return cache,error


def distribution(v):
    v=v.double().flatten();values=v.tolist()
    result=dict(mean=float(v.mean()),minimum=float(v.min()),maximum=float(v.max()))
    assert abs(result['mean']-sum(values)/len(values))<=1e-9
    return result


def score(short,long,rows):
    assert torch.equal(short['labels'],long['labels']);assert torch.equal(short['pre_cross'],long['pre_cross'])
    groups={}
    for group in ('all','distinct','repeated'):
        ids=[i for i,r in enumerate(rows) if group=='all' or (len(set(r[:6]))==6)==(group=='distinct')]
        if not ids:groups[group]=None;continue
        stats={}
        for name in ('pre_cross','cross','cross_residual','cross_norm','ffn','ffn_residual','final'):
            stats[name]=distribution((long[name][ids]-short[name][ids]).norm(dim=-1))
        stats['encoder_common']=distribution((long['memory'][ids,:6]-short['memory'][ids,:6]).norm(dim=-1))
        stats['encoder_end']=distribution((long['memory'][ids,-1]-short['memory'][ids,-1]).norm(dim=-1))
        delta=long['final'][ids]-short['final'][ids];readout=short['readout'][ids];y=short['labels'][ids]
        zlong,zshort=long['logits'][ids],short['logits'][ids]
        observed=(zlong[:,2]-zlong.gather(1,y[:,None]).squeeze(1))-(zshort[:,2]-zshort.gather(1,y[:,None]).squeeze(1))
        identity=(readout*delta).sum(-1);assert torch.allclose(observed,identity,atol=1e-9,rtol=0)
        scalar=[sum(x*y for x,y in zip(w,d)) for w,d in zip(readout.tolist(),delta.tolist())]
        assert max(abs(a-b) for a,b in zip(scalar,observed.tolist()))<=1e-9
        denom=readout.norm(dim=-1)*delta.norm(dim=-1);valid=denom>1e-12
        final=dict(extension_margin=distribution(observed),readout_norm=distribution(readout.norm(dim=-1)),cosine=distribution(identity[valid]/denom[valid]) if valid.any() else None,undefined_cosines=int((~valid).sum()))
        attention={}
        for name,e in (('short',short),('long',long)):
            a=e['attention'][ids];v=e['projected_values'][ids];heads=[]
            for head in range(a.shape[1]):
                aa=a[:,head];heads.append(dict(target_mass=distribution(aa[:,4]),end_mass=distribution(aa[:,-1]),
                    suffix_mass=distribution(aa[:,6:-1].sum(-1)),entropy=distribution(-(aa*aa.clamp_min(1e-300).log()).sum(-1)),
                    target_value_norm=distribution(v[:,head,4].norm(dim=-1)),end_value_norm=distribution(v[:,head,-1].norm(dim=-1))))
            attention[name]=dict(heads=heads,cross_output_norm=distribution(e['cross'][ids].norm(dim=-1)))
        groups[group]=dict(count=len(ids),distances=stats,final=final,attention=attention)
    for e in (short,long):
        metrics=summarize(e['logits'],e['labels'],rows,True);audit_logits(e['logits'],e['labels'],rows,metrics,True)
    return groups


def preflight():
    torch.manual_seed(47);m=make_model('rope').eval();folder=ROOT/'results/test_runs'/('intact_'+uuid.uuid4().hex);folder.mkdir(parents=True)
    torch.save(m.state_dict(),folder/'initial.pt');m.load_state_dict(torch.load(folder/'initial.pt',weights_only=True))
    rows=[[3]*10,list(range(3,9))+[3]*4];s,_=trace(m,[r[:6] for r in rows]);l,_=trace(m,rows);score(s,l,rows)
    null=score(s,s,rows);assert null['all']['final']['extension_margin']['mean']==0 and null['all']['final']['undefined_cosines']==2
    assert score({k:v[:1] for k,v in s.items()},{k:v[:1] for k,v in l.items()},rows[:1])['distinct'] is None
    errors=0
    try:check_cross(s['cross_residual'],s['cross_residual']+.01)
    except AssertionError:errors+=1
    metrics=summarize(s['logits'],s['labels'],rows,True)
    for mode in ('logit','missing'):
        z=s['logits'].clone();mm=dict(metrics)
        if mode=='logit':z[0,2]+=1
        else:mm.pop('distinct')
        try:audit_logits(z,s['labels'],rows,mm,True)
        except AssertionError:errors+=1
    assert errors==3
    return dict(passed=True,initialized_saved_trace=True,reference_and_scalar_audit=True,null_and_absent_groups=True,three_intentional_errors_rejected=True)


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists();write(OUT/'preflight.json',preflight())
    protected=json.loads((ROOT/'results/copy_routing_development/plan.json').read_text())['protected']
    for p in ('experiments/copy_intact_development.py','docs/protocols/copy-intact-development.md','results/copy_intact_development/preflight.json'):
        protected[p]=digest(ROOT/p)
    assert all(digest(ROOT/p)==v for p,v in protected.items())
    write(OUT/'plan.json',dict(models=MODELS,stages=[500,1000],paired_predictions=2048,protected=protected))
    rows=json.loads((PARENT/'data.json').read_text())['pools']['test_10'];assert len(rows)==128
    runs=[];maxerror=0.;artifacts={}
    for seed,arm in MODELS:
        for step in (500,1000):
            m=make_model('rope').eval();m.load_state_dict(torch.load(SNAP/f'{seed}_{arm}_{step}.pt',weights_only=True)['model'])
            s,e=trace(m,[r[:6] for r in rows]);l,f=trace(m,rows);maxerror=max(maxerror,e,f)
            old=torch.load(SNAP/f'{seed}_{arm}_{step}_logits.pt',weights_only=True)
            for n,t in ((6,s),(10,l)):
                assert torch.equal(t['labels'],old[f'paired_{n}']['labels'])
                error=float((t['logits']-old[f'paired_{n}']['logits']).abs().max());maxerror=max(maxerror,error)
                assert error<=1e-9 and torch.equal(t['logits'].argmax(-1),old[f'paired_{n}']['logits'].argmax(-1))
            p=OUT/f'{seed}_{arm}_{step}.pt';torch.save(dict(short=s,long=l),p);ev=torch.load(p,weights_only=True)
            groups=score(ev['short'],ev['long'],rows);assert all(v is not None for v in groups.values());artifacts[p.name]=digest(p)
            runs.append(dict(seed=seed,arm=arm,step=step,groups=groups));print(seed,arm,step,groups['all']['final'],flush=True)
    assert all(digest(ROOT/p)==v for p,v in protected.items())
    write(OUT/'report.json',dict(runs=runs,training_updates=0,plan_sha256=digest(OUT/'plan.json'),artifact_sha256=artifacts,audit=dict(passed=True,max_error=maxerror,prediction_differences=0,protected_files=len(protected),paired_predictions=2048)))


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()
