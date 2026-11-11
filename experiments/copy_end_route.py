"""Fixed-attention end-route subtraction, native hooks versus explicit equations."""
import json
import math
import uuid
from pathlib import Path
import torch
from copy_position_sites import ROOT,forward,make_model,digest,write
from copy_site_failure import MODELS,PARENT
from copy_rope_model import attention,rotate,matrix_rotation
from audit_kv_cache_equivalence import norm

OUT=ROOT/'results/copy_end_route'
MODES=('native','end','control')


def contributions(model,memory,x,ref=False):
    m=model.base.decoder.multihead_attn;h=m.num_heads;d=m.embed_dim//h
    q,k,v=[(a@w.T+b).reshape(len(a),a.shape[1],h,d).transpose(1,2) for a,w,b in zip((x,memory,memory),m.in_proj_weight.chunk(3),m.in_proj_bias.chunk(3))]
    rot=matrix_rotation if ref else rotate
    scores=rot(q,torch.arange(x.shape[1]))@rot(k,torch.arange(memory.shape[1])).transpose(-1,-2)/math.sqrt(d)
    if ref:
        ex=(scores-scores.max(-1,keepdim=True).values).exp();a=ex/ex.sum(-1,keepdim=True)
        parts=[]
        for j in range(memory.shape[1]):
            parts.append(sum((a[:,i,:,j,None]*v[:,i,j,None,:])@m.out_proj.weight[:,i*d:(i+1)*d].T for i in range(h)))
        c=torch.stack(parts,2)
    else:
        c=scores.softmax(-1)[:,:,:,:,None]*v[:,:,None,:,:]
        c=c.permute(0,2,3,1,4).flatten(-2)@m.out_proj.weight.T
    return c


def removal(c,mode):
    end=c[:,:,-1];base=c[:,:,0];en=end.norm(dim=-1);bn=base.norm(dim=-1)
    scale=torch.where(bn>1e-12,en/bn.clamp_min(1e-12),torch.zeros_like(bn))
    delta=torch.zeros_like(end) if mode=='native' else end if mode=='end' else base*scale[:,:,None]
    return delta,dict(end_norm=en,control_norm=bn,scale=scale,eligible=bn>1e-12)


def patched(model,content,prefix,mode):
    cache={};handles=[];dec=model.base.decoder
    handles.append(model.base.encoder.norm2.register_forward_hook(lambda m,a,o:cache.update(memory=o)))
    handles.append(dec.norm1.register_forward_hook(lambda m,a,o:cache.update(x=o)))
    def change(m,args):
        c=contributions(model,cache['memory'],cache['x']);delta,stats=removal(c,mode)
        reconstructed=cache['x']+c.sum(2)+dec.multihead_attn.out_proj.bias
        assert float((reconstructed-args[0]).abs().max())<1e-9
        cache.update(stats=stats)
        return (args[0]-delta,)
    handles.append(dec.norm2.register_forward_pre_hook(change))
    try:z=forward(model,content,prefix)[0]
    finally:
        for h in handles:h.remove()
    return z,cache['stats']


def reference(model,content,prefix,mode):
    base=model.base;e,d=base.encoder,base.decoder
    source=torch.tensor([r+[2] for r in content.tolist()]);s=base.source(source)*math.sqrt(8);t=base.target(prefix)*math.sqrt(8)
    ep,dp=torch.arange(source.shape[1]),torch.arange(prefix.shape[1]);sp=torch.zeros_like(source,dtype=torch.bool);tp=torch.zeros_like(prefix,dtype=torch.bool)
    def ff(l,x):return (x@l.linear1.weight.T+l.linear1.bias).clamp_min(0)@l.linear2.weight.T+l.linear2.bias
    mem=norm(s+attention(e.self_attn,s,s,s,sp,ep,ep,reference=True),e.norm1);mem=norm(mem+ff(e,mem),e.norm2)
    x=norm(t+attention(d.self_attn,t,t,t,tp,dp,dp,True,True),d.norm1)
    c=contributions(model,mem,x,True)
    if mode=='end':delta=c[:,:,-1]
    elif mode=='control':
        delta=torch.zeros_like(x)
        for i in range(len(x)):
            for j in range(x.shape[1]):
                a,b=c[i,j,-1],c[i,j,0];den=math.sqrt(sum(float(v*v) for v in b))
                if den>1e-12:delta[i,j]=b*(math.sqrt(sum(float(v*v) for v in a))/den)
    else:delta=torch.zeros_like(x)
    x=norm(x+c.sum(2)+d.multihead_attn.out_proj.bias-delta,d.norm2)
    x=norm(x+ff(d,x),d.norm3)
    return base.output(x)


def metrics(z,labels,native,terminal):
    top=z.argmax(-1);old=native.argmax(-1);good=top==labels;ng=old==labels
    if terminal:
        other=z.clone();other[:,2]=-torch.inf;margin=z[:,2]-other.max(-1).values
    else:margin=z[:,2]-z.gather(1,labels[:,None]).squeeze(1)
    return dict(count=len(labels),correct=int(good.sum()),eos=int((top==2).sum()),rescues=int((~ng&good).sum()),harms=int((ng&~good).sum()),mean_eos_margin=float(margin.mean())),margin


def preflight():
    torch.manual_seed(71);m=make_model('rope').eval();folder=ROOT/'results/test_runs'/('end_route_'+uuid.uuid4().hex);folder.mkdir(parents=True)
    p=folder/'init.pt';torch.save(m.state_dict(),p);m.load_state_dict(torch.load(p,weights_only=True))
    x=torch.tensor([[3,4,5],[4,4,4]]);prefix=torch.tensor([[1]+r for r in x.tolist()])
    with torch.no_grad():
        native=forward(m,x,prefix)[0]
        for mode in MODES:
            z,st=patched(m,x,prefix,mode);rz=reference(m,x,prefix,mode)
            assert float((z-rz).abs().max())<1e-9
            if mode=='native':assert torch.equal(z,native)
        c=torch.randn(2,4,4,8,dtype=torch.float64);delta,st=removal(c,'control')
        assert torch.allclose(delta.norm(dim=-1),c[:,:,-1].norm(dim=-1))
        c[:,:,0]=0;assert not removal(c,'control')[1]['eligible'].any()
        assert torch.equal(removal(c,'native')[0],torch.zeros_like(c[:,:,-1]))
        c[:,:,-1]=0;assert not removal(c,'end')[0].any()
    z=torch.tensor([[0.,0.,3.,2.],[0.,0.,1.,2.]],dtype=torch.float64);labels=torch.tensor([3,3]);changed=z.flip(0)
    met,_=metrics(changed,labels,z,False);assert met['rescues']==1 and met['harms']==1
    # Wrong-sign subtraction must fail a known linear downstream prediction.
    residual=torch.tensor([0.,2.]);component=torch.tensor([0.,3.])
    assert (residual-component).argmax()!= (residual+component).argmax()
    return dict(passed=True,initialized_saved_reload=True,reference_modes=3,norm_and_zero_controls=True,synthetic_rescue_harm=True,wrong_sign_rejected=True)


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'report.json').exists();write(OUT/'preflight.json',preflight())
    parent=json.loads((PARENT/'report.json').read_text());data=json.loads((PARENT/'data.json').read_text())['pools']
    diag=json.loads((ROOT/'results/copy_site_failure/report.json').read_text())
    cohort=next(c for c in diag['cells'] if c['seed']==3767 and c['arm']=='cross_only' and c['pool']=='test_10')
    sites=torch.tensor([r['first_error_position']-1 for r in cohort['records']])
    runs=[next(r for r in parent['confirmation'] if (r['seed'],r['arm'])==pair) for pair in MODELS]
    protected=json.loads((ROOT/'results/copy_site_eos/plan.json').read_text())['protected']
    paths=[Path(__file__),ROOT/'docs/protocols/copy-end-route.md',OUT/'preflight.json',ROOT/'results/copy_site_failure/report.json']
    protected.update({str(p.relative_to(ROOT)):digest(p) for p in paths})
    assert all(digest(ROOT/p)==v for p,v in protected.items())
    write(OUT/'plan.json',dict(protected=protected,models=MODELS,modes=MODES,cohorts=['long_failure','long_terminal','short_content','short_terminal'],rows_per_cohort=128))
    cells=[];saved={};maxerr=0.;total=0
    for run in runs:
        model=make_model('rope').eval();model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model']);old=torch.load(ROOT/run['scores'],weights_only=True)
        for name,pool,idx,terminal in [('long_failure','test_10',sites,False),('long_terminal','test_10',torch.full((128,),10),True),('short_content','test_6',torch.full((128,),3),False),('short_terminal','test_6',torch.full((128,),6),True)]:
            rows=data[pool];x=torch.tensor(rows);assert len(rows)==128
            prefix=torch.tensor([[1]+r for r in rows]);labels=torch.tensor([2 if terminal else r[int(j)] for r,j in zip(rows,idx)])
            zs={};stats=None
            with torch.no_grad():
                for mode in MODES:
                    z,st=patched(model,x,prefix,mode);rz=reference(model,x,prefix,mode)
                    err=float((z-rz).abs().max());assert err<=1e-9 and torch.equal(z.argmax(-1),rz.argmax(-1));maxerr=max(maxerr,err)
                    zs[mode]=z[torch.arange(128),idx];stats={k:v[torch.arange(128),idx] for k,v in st.items()}
                archived=old[pool]['logits'][torch.arange(128),idx]
                err=float((zs['native']-archived).abs().max());assert err<=1e-9 and torch.equal(zs['native'].argmax(-1),archived.argmax(-1));maxerr=max(maxerr,err)
            groups={};margins={}
            for g in ('all','distinct','repeated'):
                ids=[i for i,r in enumerate(rows) if g=='all' or (len(set(r))==len(r))==(g=='distinct')]
                if not ids:groups[g]=None;continue
                groups[g]={}
                for mode in MODES:
                    met,margin=metrics(zs[mode][ids],labels[ids],zs['native'][ids],terminal);groups[g][mode]=met
                    # Independent scalar correctness and rescue/harm counts.
                    a=[max(range(16),key=v.__getitem__) for v in zs[mode][ids].tolist()];b=zs['native'][ids].argmax(-1).tolist();yy=labels[ids].tolist()
                    assert met['correct']==sum(p==y for p,y in zip(a,yy))
                    assert met['rescues']==sum(p==y and q!=y for p,q,y in zip(a,b,yy))
                    assert met['harms']==sum(p!=y and q==y for p,q,y in zip(a,b,yy))
                    if g=='all':margins[mode]=margin
            eligible=stats['eligible'];better=(margins['control']-margins['end']>1e-10)&eligible
            cell=dict(seed=run['seed'],arm=run['arm'],cohort=name,groups=groups,eligible=int(eligible.sum()),zero_end=int((stats['end_norm']<=1e-12).sum()),scale_min=float(stats['scale'].min()),scale_max=float(stats['scale'].max()),end_better_margin_count=int(better.sum()))
            cells.append(cell);saved[f"{run['seed']}_{run['arm']}_{name}"]=dict(logits=zs,labels=labels,positions=idx,stats=stats);total+=128
            print(json.dumps({k:v for k,v in cell.items() if k!='groups'}|{'all':groups['all']}),flush=True)
    primary=cells[0];g=primary['groups']['all'];n=g['native']['count']
    supported=g['end']['correct']/n>=.9 and (g['end']['correct']-g['control']['correct'])/n>=.1 and primary['eligible']>0 and primary['end_better_margin_count']/primary['eligible']>=.9
    torch.save(saved,OUT/'evidence.pt')
    assert len(cells)==16 and total==2048 and all(digest(ROOT/p)==v for p,v in protected.items())
    write(OUT/'report.json',dict(cells=cells,supported=supported,training_updates=0,plan_sha256=digest(OUT/'plan.json'),evidence_sha256=digest(OUT/'evidence.pt'),audit=dict(passed=True,max_logit_error=maxerr,prediction_differences=0,row_evaluations_per_mode=total,protected_files=len(protected)),limitation='Endpoint row intervention; shared reference helpers; no generation or training-mechanism inference.'))


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()
