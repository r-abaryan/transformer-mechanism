"""Plain delta memory: independent transition expansion and local witnesses."""
import itertools
import torch
from cross_attention import ROOT, error
from position_distractors import digest
from recurrent_retrieval import write


def unit(k):
    norm=k.square().sum(-1,keepdim=True).sqrt()
    if not bool((norm>0).all()):raise ValueError('Zero key has no unit direction')
    return k/norm


def recurrence(q,k,v,b,state=None,before=False,mode='correct'):
    s=q.new_zeros(*q.shape[:-2],q.shape[-1],v.shape[-1]) if state is None else state
    rows=[]
    for i in range(q.shape[-2]):
        key=k[...,i,:];old=(key[...,None]*s).sum(-2)
        prior=s
        correction=v[...,i,:]-old
        if mode=='add':correction=v[...,i,:]
        if mode=='reverse':correction=-correction
        s=s+b[...,i,None,None]*key[...,None]*correction[...,None,:]
        rows.append((q[...,i,:,None]*(prior if before else s)).sum(-2))
    return torch.stack(rows,-2),s


def expansion(q,k,v,b,reverse=False):
    n=q.shape[-2];eye=torch.eye(k.shape[-1],dtype=q.dtype)
    transitions=eye-b[...,None,None]*k[..., :,None]*k[...,None,:]
    writes=b[...,None,None]*k[..., :,None]*v[...,None,:]
    outputs=[]
    for i in range(n):
        terms=[]
        for j in range(i+1):
            term=writes[...,j,:,:]
            order=list(range(j+1,i+1))
            for t in (order[::-1] if reverse else order):term=transitions[...,t,:,:]@term
            terms.append(term)
        state=torch.stack(terms).sum(0)
        outputs.append((q[...,i,None,:]@state).squeeze(-2))
    return torch.stack(outputs,-2),state


def chunked(q,k,v,b,detach=False):
    a,s=recurrence(q[...,:1,:],k[...,:1,:],v[...,:1,:],b[...,:1])
    c,s=recurrence(q[...,1:,:],k[...,1:,:],v[...,1:,:],b[...,1:],s.detach() if detach else s)
    return torch.cat((a,c),-2),s


def case(seed,n,m,dv):
    torch.manual_seed(seed)
    xs=[torch.randn(2,2,n,d,dtype=torch.float64,requires_grad=True) for d in (m,m,dv)]+[torch.randn(2,2,n,dtype=torch.float64,requires_grad=True)]
    q,raw,v,a=xs;k=unit(raw);b=a.sigmoid()
    ref,state=expansion(q,k,v,b);probe=torch.randn_like(ref)/ref.numel()
    gr=torch.autograd.grad((ref*probe).sum(),xs,retain_graph=True)
    forms=[f(q,k,v,b) for f in (recurrence,chunked)]
    oe=max(error(ref,y) for y,_ in forms);se=max(error(state,s) for _,s in forms)
    ge=max(error(u,w) for y,_ in forms for u,w in zip(gr,torch.autograd.grad((y*probe).sum(),xs,retain_graph=True)))
    ds=[torch.randn_like(x) for x in xs];norm=sum(d.square().sum() for d in ds).sqrt();ds=[d/norm for d in ds]
    with torch.no_grad():
        vals=[]
        for sign in (1,-1):
            p=[x+sign*1e-6*d for x,d in zip(xs,ds)]
            vals.append(float((expansion(p[0],unit(p[1]),p[2],p[3].sigmoid())[0]*probe).sum()))
        fd=abs((vals[0]-vals[1])/2e-6-sum(float((u*d).sum()) for u,d in zip(gr,ds)))
        changed=[x.clone() for x in xs]
        for x in changed:
            if x.ndim==4:x[...,-1,:]+=3
            else:x[...,-1]+=3
        p=changed
        fe=max(error(f(p[0],unit(p[1]),p[2],p[3].sigmoid())[0][...,:-1,:],ref[...,:-1,:]) for f in (recurrence,expansion,chunked))
        initial=torch.randn(2,2,m,dv,dtype=torch.float64)
        kk=k[...,0,:];qq=q[...,0,:];vv=v[...,0,:];bb=b[...,0]
        after=recurrence(qq[...,None,:],kk[...,None,:],vv[...,None,:],bb[...,None],initial)[1]
        change=(qq[...,None,:]@(after-initial)).squeeze(-2)
        expected=bb[...,None]*(qq*kk).sum(-1,keepdim=True)*(vv-(kk[...,None,:]@initial).squeeze(-2))
        ie=error(change,expected)
    assert max(oe,se,ge,fe,ie)<=1e-10 and fd<=1e-7
    return dict(seed=seed,n=n,m=m,dv=dv,output_error=oe,state_error=se,gradient_error=ge,directional_error=fd,future_error=fe,interference_error=ie)


def fixtures():
    dtype=torch.float64
    k=torch.tensor([[[[1.,0.],[.6,.8],[0.,1.]]]],dtype=dtype)
    q=torch.tensor([[[[1.,1.],[1.,1.],[1.,1.]]]],dtype=dtype)
    v=torch.tensor([2.,5.,8.],dtype=dtype).reshape(1,1,3,1).requires_grad_();b=torch.full((1,1,3),.7,dtype=dtype)
    good=recurrence(q,k,v,b)[0];bad=chunked(q,k,v,b,True)[0]
    ga=torch.autograd.grad(good[...,-1,:].sum(),v)[0];gb=torch.autograd.grad(bad[...,-1,:].sum(),v)[0]
    effects=dict(omit_subtraction=error(good,recurrence(q,k,v,b,mode='add')[0]),reverse_sign=error(good,recurrence(q,k,v,b,mode='reverse')[0]),reverse_transition_order=error(good,expansion(q,k,v,b,True)[0]),detach=error(ga,gb),read_before_write=error(good,recurrence(q,k,v,b,before=True)[0]))
    assert all(x>1e-6 for x in effects.values()) and error(good,bad)==0
    overwrite=[]
    for beta in (0.,.5,1.):
        initial=torch.tensor([[[[2.]]]],dtype=dtype,requires_grad=True)
        one=torch.ones(1,1,1,1,dtype=dtype)
        y,_=recurrence(one,one,one*8,torch.full((1,1,1),beta,dtype=dtype),initial)
        grad=torch.autograd.grad(y.sum(),initial)[0].item()
        assert y.item()==2+6*beta and grad==1-beta
        overwrite.append(dict(beta=beta,read=y.item(),old_value_gradient=grad))
    basis=torch.tensor([[[[1.,0.],[0.,1.],[1.,0.]]]],dtype=dtype)
    _,s=recurrence(basis,basis,v.detach(),torch.ones_like(b))
    assert error(s.flatten(),torch.tensor([8.,5.],dtype=dtype))==0
    initial=torch.tensor([[[[2.],[4.75]]]],dtype=dtype)
    _,changed=recurrence(basis[...,:1,:],basis[...,:1,:],v.detach()[...,2:,:],b[...,:1]*0+1,initial)
    other=torch.tensor([.6,.8],dtype=dtype)
    before=(other@initial[0,0]).item();after=(other@changed[0,0]).item()
    assert abs(before-5)<1e-12 and abs(after-8.6)<1e-12
    one=torch.ones(1,1,3,1,dtype=dtype)
    unnorm=recurrence(one*2,one*2,one*2,torch.ones_like(b))[0].flatten().tolist()
    normalized=recurrence(one,one,one*2,torch.ones_like(b))[0].flatten().tolist()
    assert unnorm==[8.,-16.,56.] and normalized==[2.,2.,2.]
    eigen=[]
    for length in (1.,2.):
        key=torch.tensor([length,0.],dtype=dtype);t=torch.eye(2,dtype=dtype)-key[:,None]*key[None,:]
        actual=torch.linalg.eigvalsh(t).tolist();assert actual==sorted([1-length*length,1.])
        eigen.append(dict(key_norm=length,beta=1.,eigenvalues=actual))
    try:unit(torch.zeros_like(one))
    except ValueError:rejected=True
    else:raise AssertionError('Zero key accepted')
    return dict(controls=effects,detach_output_error=error(good,bad),overwrite=overwrite,orthogonal_final_reads=s.flatten().tolist(),correlated_before=before,correlated_after=after,unnormalized_reads=unnorm,unit_reads=normalized,eigenvalues=eigen,zero_key_rejected=rejected)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    p=ROOT/'results/delta_memory.json';assert not p.exists()
    history={str(f.relative_to(ROOT)):digest(f) for f in (ROOT/'results').glob('*.json')}
    paths=['experiments/delta_memory.py','docs/protocols/delta-memory.md']
    r=dict(status='running',dtype='float64',device='cpu',torch=torch.__version__,training_updates=0,provenance={f:digest(ROOT/f) for f in paths},historical=history,cases=[]);write(p,r)
    try:
        for seed,n,(m,dv) in itertools.product((2663,2671,2683),(3,7,17),((3,2),(5,4))):r['cases'].append(case(seed,n,m,dv))
        r.update(fixtures=fixtures());assert all(digest(ROOT/f)==h for f,h in history.items());r.update(status='complete',passed=True)
    except Exception as exc:r.update(status='failed',failure=repr(exc));raise
    finally:write(p,r)
    print({k:max(c[k] for c in r['cases']) for k in r['cases'][0] if k.endswith('error')});print(r['fixtures'])


if __name__=='__main__':main()
