"""Fixed long encoder, restricted cross-attention keys at the fifth decision."""
import json
import math
import uuid
import torch
from copy_position_sites import ROOT,make_model,forward,digest,write
from copy_site_failure import MODELS,PARENT
from copy_rope_model import attention,rotate,matrix_rotation
from audit_kv_cache_equivalence import norm

OUT=ROOT/'results/copy_added_read_mask'


def mix(scores,v,masked,reference=False):
    n=scores.shape[-1]-1;keep=torch.tensor([j<6 or j==n for j in range(n+1)])
    if reference:
        ids=torch.where(keep)[0] if masked else torch.arange(n+1)
        s=scores[...,ids];ex=(s-s.max(-1,keepdim=True).values).exp()
        return (ex/ex.sum(-1,keepdim=True))@v[:,:,ids]
    s=scores.masked_fill(~keep,-torch.inf) if masked else scores
    return s.softmax(-1)@v


def cross(m,x,memory,masked,ref=False):
    h=m.num_heads;d=m.embed_dim//h
    q,k,v=[(a@w.T+b).reshape(len(a),a.shape[1],h,d).transpose(1,2) for a,w,b in zip((x,memory,memory),m.in_proj_weight.chunk(3),m.in_proj_bias.chunk(3))]
    rot=matrix_rotation if ref else rotate
    scores=rot(q,torch.arange(x.shape[1]))@rot(k,torch.arange(memory.shape[1])).transpose(-1,-2)/math.sqrt(d)
    out=mix(scores,v,masked,ref).transpose(1,2).reshape_as(x)
    return out@m.out_proj.weight.T+m.out_proj.bias


def producer(model,content,prefix,masked):
    cache={};handles=[];b=model.base;d=b.decoder
    handles.append(b.encoder.norm2.register_forward_hook(lambda m,a,o:cache.update(memory=o)))
    handles.append(d.norm1.register_forward_hook(lambda m,a,o:cache.update(x=o)))
    def change(m,args):
        if not masked:return args
        return (cache['x']+cross(d.multihead_attn,cache['x'],cache['memory'],True),)
    handles.append(d.norm2.register_forward_pre_hook(change))
    try:z=forward(model,content,prefix)[0][:,-1]
    finally:
        for h in handles:h.remove()
    return z,cache


def reference(model,content,prefix,masked):
    b=model.base;e,d=b.encoder,b.decoder
    source=torch.tensor([r+[2] for r in content.tolist()]);s=b.source(source)*math.sqrt(8);t=b.target(prefix)*math.sqrt(8)
    ep,dp=torch.arange(source.shape[1]),torch.arange(prefix.shape[1]);sp=torch.zeros_like(source,dtype=torch.bool);tp=torch.zeros_like(prefix,dtype=torch.bool)
    def ff(l,x):return (x@l.linear1.weight.T+l.linear1.bias).clamp_min(0)@l.linear2.weight.T+l.linear2.bias
    memory=norm(s+attention(e.self_attn,s,s,s,sp,ep,ep,reference=True),e.norm1);memory=norm(memory+ff(e,memory),e.norm2)
    x=norm(t+attention(d.self_attn,t,t,t,tp,dp,dp,True,True),d.norm1)
    x=norm(x+cross(d.multihead_attn,x,memory,masked,True),d.norm2);x=norm(x+ff(d,x),d.norm3)
    return b.output(x)[:,-1]


def metrics(a,b,y,rows):
    groups={}
    for group in ('all','distinct_prefix','repeated_prefix'):
        ids=[i for i,r in enumerate(rows) if group=='all' or (len(set(r[:6]))==6)==(group=='distinct_prefix')]
        if not ids:groups[group]=None;continue
        za,zb,yy=a[ids],b[ids],y[ids];pa,pb=za.argmax(-1),zb.argmax(-1);ga,gb=pa==yy,pb==yy
        ma=za[:,2]-za.gather(1,yy[:,None]).squeeze(1);mb=zb[:,2]-zb.gather(1,yy[:,None]).squeeze(1)
        m=dict(count=len(ids),native_correct=int(ga.sum()),masked_correct=int(gb.sum()),native_eos=int((pa==2).sum()),masked_eos=int((pb==2).sum()),rescues=int((~ga&gb).sum()),harms=int((ga&~gb).sum()),native_margin=float(ma.mean()),masked_margin=float(mb.mean()),margin_change=float((mb-ma).mean()))
        ap=[max(range(len(v)),key=v.__getitem__) for v in za.tolist()];bp=[max(range(len(v)),key=v.__getitem__) for v in zb.tolist()];labels=yy.tolist()
        assert m['rescues']==sum(p!=t and q==t for p,q,t in zip(ap,bp,labels))
        assert m['harms']==sum(p==t and q!=t for p,q,t in zip(ap,bp,labels))
        delta=[(q[2]-q[t])-(p[2]-p[t]) for p,q,t in zip(za.tolist(),zb.tolist(),labels)]
        assert abs(sum(delta)/len(delta)-m['margin_change'])<=1e-9
        groups[group]=m
    return groups


def evaluate(model,rows):
    x=torch.tensor(rows);prefix=torch.tensor([[1]+r[:4] for r in rows]);y=x[:,4];zs=[];caches=[];error=0.
    assert prefix.shape[1]==5 and torch.equal(prefix[:,1:],x[:,:4])
    with torch.no_grad():
        for masked in (False,True):
            z,cache=producer(model,x,prefix,masked);rz=reference(model,x,prefix,masked)
            err=float((z-rz).abs().max());assert err<=1e-9 and torch.equal(z.argmax(-1),rz.argmax(-1));error=max(error,err);zs.append(z);caches.append(cache)
    for key in ('memory','x'):assert torch.equal(caches[0][key],caches[1][key])
    return dict(native=zs[0],masked=zs[1],labels=y),metrics(*zs,y,rows),error


def preflight():
    torch.manual_seed(97);model=make_model('rope').eval();folder=ROOT/'results/test_runs'/('added_reads_'+uuid.uuid4().hex);folder.mkdir(parents=True)
    p=folder/'initialized.pt';torch.save(model.state_dict(),p);model.load_state_dict(torch.load(p,weights_only=True))
    rows=[[3,4,5,6,7,8,9,10,3,4],[4]*10];evaluate(model,rows)
    short=[r[:6] for r in rows];ev,_,_=evaluate(model,short);assert torch.equal(ev['native'],ev['masked'])
    scores=torch.zeros(1,2,1,11,dtype=torch.float64);v=torch.randn(1,2,11,4,dtype=torch.float64);base=mix(scores,v,True)
    missing_mask_detected=[]
    for sign in (-1,1):
        s=scores.clone();vv=v.clone();s[...,6:10]=sign*100;vv[:,:,6:10]+=sign*100
        assert torch.equal(mix(s,vv,True),base)
        missing_mask_detected.append(not torch.allclose(mix(s,vv,False),base))
    assert any(missing_mask_detected)
    marker=v.clone();marker[:,:,10]+=1;assert not torch.equal(mix(scores,marker,True),base)
    a=torch.zeros(2,16,dtype=torch.float64);a[0,2]=3;a[1,3]=3;b=a.flip(0);y=torch.tensor([3,3]);m=metrics(a,b,y,rows)['all'];assert m['rescues']==1 and m['harms']==1
    b[0,2]=3;assert metrics(a,b,y,rows)['all']['masked_eos']==2
    assert metrics(a[1:],b[1:],y[1:],rows[1:])['distinct_prefix'] is None
    return dict(passed=True,initialized_reload=True,zero_mask_identity=True,both_sign_perturbations=True,missing_mask_rejected=True,marker_retained=True,synthetic_metrics=True)


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'report.json').exists();write(OUT/'preflight.json',preflight())
    parent=json.loads((PARENT/'report.json').read_text());rows=json.loads((PARENT/'data.json').read_text())['pools']['test_10'];assert len(rows)==128
    old=torch.load(ROOT/'results/copy_prefix_extension/evidence.pt',weights_only=True)
    protected=json.loads((ROOT/'results/copy_prefix_extension/plan.json').read_text())['protected']
    for p in (ROOT/'experiments/copy_added_read_mask.py',ROOT/'docs/protocols/copy-added-read-mask.md',OUT/'preflight.json',ROOT/'results/copy_prefix_extension/evidence.pt',ROOT/'results/copy_prefix_extension/report.json'):
        protected[str(p.relative_to(ROOT))]=digest(p)
    assert all(digest(ROOT/p)==v for p,v in protected.items())
    write(OUT/'plan.json',dict(models=MODELS,position=5,masked_indices=[6,7,8,9],retained_eos_index=10,protected=protected))
    cells=[];saved={};err=0.
    for seed,arm in MODELS:
        run=next(r for r in parent['confirmation'] if r['seed']==seed and r['arm']==arm);model=make_model('rope').eval();model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        ev,groups,error=evaluate(model,rows);archive=old[f'{seed}_{arm}_5']['long'];ae=float((ev['native']-archive).abs().max());assert ae<=1e-9 and torch.equal(ev['native'].argmax(-1),archive.argmax(-1));err=max(err,error,ae)
        cells.append(dict(seed=seed,arm=arm,groups=groups));saved[f'{seed}_{arm}']=ev;print(json.dumps(dict(seed=seed,arm=arm,**groups['all'])),flush=True)
    recovery=cells[0]['groups']['all']['masked_correct']/128>=.95
    preservation=all((g['native_correct']-g['masked_correct'])/g['count']<=.05 for c in cells[1:] for g in c['groups'].values() if g)
    assert all(c['groups'][g] is not None for c in cells for g in ('all','distinct_prefix','repeated_prefix'))
    torch.save(saved,OUT/'evidence.pt');assert all(digest(ROOT/p)==v for p,v in protected.items())
    write(OUT/'report.json',dict(cells=cells,recovery=recovery,control_preservation=preservation,supported=recovery and preservation,training_updates=0,plan_sha256=digest(OUT/'plan.json'),evidence_sha256=digest(OUT/'evidence.pt'),audit=dict(passed=True,max_logit_error=err,prediction_differences=0,paired_cells=4,selected_predictions_per_implementation=1024,encoder_and_pre_cross_states_identical=True,protected_files=len(protected)),limitation='Opened pools, correct prefixes, one next-token row; masking renormalizes attention; shared reference helpers.'))


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()
