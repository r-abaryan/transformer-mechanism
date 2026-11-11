"""Decay followed by delta correction, audited independently of trained models."""
import itertools
import torch
from delta_memory import unit, recurrence as plain_delta
from cross_attention import ROOT, error
from position_distractors import digest
from recurrent_retrieval import write


def recurrent(q,k,v,a,b,state=None,wrong=None):
    s=q.new_zeros(*q.shape[:-2],q.shape[-1],v.shape[-1]) if state is None else state
    rows=[]
    for i in range(q.shape[-2]):
        key=k[...,i,:];prior=s;decayed=a[...,i,None,None]*s
        if wrong=='omit_decay':decayed=s
        old=(key[...,None]*(s if wrong=='undecayed_read' else decayed)).sum(-2)
        write=b[...,i,None,None]*key[...,None]*(v[...,i,:]-old)[...,None,:]
        s=decayed+(a[...,i,None,None]*write if wrong=='decay_write' else write)
        rows.append((q[...,i,:,None]*(prior if wrong=='read_before' else s)).sum(-2))
    return torch.stack(rows,-2),s


def expanded(q,k,v,a,b):
    n=q.shape[-2];eye=torch.eye(k.shape[-1],dtype=q.dtype)
    transitions=a[...,None,None]*(eye-b[...,None,None]*k[..., :,None]*k[...,None,:])
    writes=b[...,None,None]*k[..., :,None]*v[...,None,:];ys=[]
    for i in range(n):
        terms=[]
        for j in range(i+1):
            term=writes[...,j,:,:]
            for t in range(j+1,i+1):term=transitions[...,t,:,:]@term
            terms.append(term)
        state=torch.stack(terms).sum(0);ys.append((q[...,i,None,:]@state).squeeze(-2))
    return torch.stack(ys,-2),state


def chunked(q,k,v,a,b,detach=False):
    y,s=recurrent(q[...,:1,:],k[...,:1,:],v[...,:1,:],a[...,:1],b[...,:1])
    z,s=recurrent(q[...,1:,:],k[...,1:,:],v[...,1:,:],a[...,1:],b[...,1:],s.detach() if detach else s)
    return torch.cat((y,z),-2),s


def case(seed,n,m,dv):
    torch.manual_seed(seed)
    xs=[torch.randn(2,2,n,d,dtype=torch.float64,requires_grad=True) for d in (m,m,dv)]+[torch.randn(2,2,n,dtype=torch.float64,requires_grad=True) for _ in range(2)]
    q,raw,v,al,bl=xs;k=unit(raw);a=al.sigmoid();b=bl.sigmoid()
    ref,state=expanded(q,k,v,a,b);probe=torch.randn_like(ref)/ref.numel()
    grads=torch.autograd.grad((ref*probe).sum(),xs,retain_graph=True)
    forms=[f(q,k,v,a,b) for f in (recurrent,chunked)]
    oe=max(error(ref,y) for y,_ in forms);se=max(error(state,s) for _,s in forms)
    ge=max(error(u,w) for y,_ in forms for u,w in zip(grads,torch.autograd.grad((y*probe).sum(),xs,retain_graph=True)))
    assert bool((grads[3][...,0]==0).all())
    ds=[torch.randn_like(x) for x in xs];size=sum(d.square().sum() for d in ds).sqrt();ds=[d/size for d in ds]
    with torch.no_grad():
        losses=[]
        for sign in (1,-1):
            p=[x+sign*1e-6*d for x,d in zip(xs,ds)]
            losses.append(float((expanded(p[0],unit(p[1]),p[2],p[3].sigmoid(),p[4].sigmoid())[0]*probe).sum()))
        fd=abs((losses[0]-losses[1])/2e-6-sum(float((g*d).sum()) for g,d in zip(grads,ds)))
        p=[x.clone() for x in xs]
        for x in p:
            if x.ndim==4:x[...,-1,:]+=3
            else:x[...,-1]+=3
        fe=max(error(f(p[0],unit(p[1]),p[2],p[3].sigmoid(),p[4].sigmoid())[0][...,:-1,:],ref[...,:-1,:]) for f in (recurrent,expanded,chunked))
        de=error(recurrent(q,k,v,torch.ones_like(a),b)[0],plain_delta(q,k,v,b)[0])
        ze=float(recurrent(q,k,v,a,torch.zeros_like(b))[0].abs().max())
        re=error(recurrent(q,k,v,torch.zeros_like(a),b)[0],b[...,None]*(q*k).sum(-1,keepdim=True)*v)
        s=torch.randn(2,2,m,dv,dtype=torch.float64);qq=q[...,:1,:];kk=k[...,:1,:];vv=v[...,:1,:];aa=a[...,:1];bb=b[...,:1]
        new=recurrent(qq,kk,vv,aa,bb,s)[1]
        priorq=(qq@s).squeeze(-2);priork=(kk@s).squeeze(-2)
        expected=(aa-1)*priorq+bb*(qq*kk).sum(-1)*(vv.squeeze(-2)-aa*priork)
        ie=error((qq@(new-s)).squeeze(-2),expected)
    assert max(oe,se,ge,fe,de,ze,re,ie)<=1e-10 and fd<=1e-7
    return dict(seed=seed,n=n,m=m,dv=dv,output_error=oe,state_error=se,gradient_error=ge,directional_error=fd,future_error=fe,plain_delta_error=de,no_write_error=ze,reset_error=re,interference_error=ie)


def fixtures():
    d=torch.float64;q=torch.ones(1,1,2,1,dtype=d);v=torch.tensor([2.,8.],dtype=d).reshape_as(q).requires_grad_()
    a=torch.full((1,1,2),.5,dtype=d);b=torch.full_like(a,.5)
    good=recurrent(q,q,v,a,b)[0];bad=chunked(q,q,v,a,b,True)[0]
    ga=torch.autograd.grad(good[...,-1,:].sum(),v)[0];gb=torch.autograd.grad(bad[...,-1,:].sum(),v)[0]
    effects={name:error(good,recurrent(q,q,v,a,b,wrong=name)[0]) for name in ('undecayed_read','decay_write','omit_decay','read_before')};effects['detach']=error(ga,gb)
    assert all(e>1e-6 for e in effects.values()) and error(good,bad)==0
    rows=[];key=torch.tensor([[[[1.,0.]]]],dtype=d);val=torch.tensor([[[[8.]]]],dtype=d)
    for alpha,beta in ((1.,1.),(.5,1.),(0.,1.),(.5,.5),(0.,0.)):
        s=torch.tensor([[[[2.],[5.]]]],dtype=d,requires_grad=True)
        _,new=recurrent(key,key,val,torch.full((1,1,1),alpha,dtype=d),torch.full((1,1,1),beta,dtype=d),s)
        gA=torch.autograd.grad(new[...,0,:].sum(),s,retain_graph=True)[0].flatten().tolist()
        gB=torch.autograd.grad(new[...,1,:].sum(),s)[0].flatten().tolist()
        reads=new.detach().flatten().tolist();assert reads==[alpha*(1-beta)*2+beta*8,alpha*5]
        assert gA==[alpha*(1-beta),0.] and gB==[0.,alpha]
        rows.append(dict(alpha=alpha,beta=beta,reads=reads,A_old_gradients=gA,B_old_gradients=gB))
    s=torch.tensor([[[[2.],[4.75]]]],dtype=d)
    new=recurrent(key,key,val,torch.full((1,1,1),.5,dtype=d),torch.ones(1,1,1,dtype=d),s)[1]
    read=(torch.tensor([.6,.8],dtype=d)@new[0,0]).item();assert abs(read-6.7)<1e-12
    return dict(controls=effects,detach_output_error=error(good,bad),boundary_reads=rows,correlated_old_read=5.,correlated_new_read=read)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    p=ROOT/'results/gated_delta_memory_verified.json';assert not p.exists()
    history={str(f.relative_to(ROOT)):digest(f) for f in (ROOT/'results').glob('*.json')}
    paths=['experiments/gated_delta_memory.py','experiments/gated_delta_memory_verified.py','experiments/delta_memory.py','docs/protocols/gated-delta-memory.md','docs/protocols/gated-delta-memory-correction.md']
    r=dict(status='running',device='cpu',dtype='float64',torch=torch.__version__,training_updates=0,historical=history,provenance={f:digest(ROOT/f) for f in paths},cases=[]);write(p,r)
    try:
        for seed,n,(m,dv) in itertools.product((2693,2699,2707),(3,7,17),((3,2),(5,4))):r['cases'].append(case(seed,n,m,dv))
        r.update(fixtures=fixtures());assert all(digest(ROOT/f)==h for f,h in history.items());r.update(status='complete',passed=True)
    except Exception as exc:r.update(status='failed',failure=repr(exc));raise
    finally:write(p,r)
    print({k:max(c[k] for c in r['cases']) for k in r['cases'][0] if k.endswith('error')});print(r['fixtures'])


if __name__=='__main__':main()
