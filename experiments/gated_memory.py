"""Scalar forgetting in normalized positive-kernel memory; no training."""
import itertools
import torch
from linear_attention import phi, divide, pairwise
from cross_attention import ROOT, error
from position_distractors import digest
from recurrent_retrieval import write


def recurrent(q,k,v,g,state=None,decay_z=True):
    s,z=state if state is not None else (q.new_zeros(*q.shape[:-2],q.shape[-1],v.shape[-1]),q.new_zeros(*q.shape[:-2],q.shape[-1]))
    rows=[]
    for i in range(q.shape[-2]):
        key=phi(k[...,i,:]);gate=g[...,i,None]
        s=gate[...,None]*s+key[...,None]*v[...,i,None,:]
        z=(gate*z if decay_z else z)+key
        query=phi(q[...,i,:])
        rows.append(divide((query[...,None]*s).sum(-2),(query*z).sum(-1,keepdim=True)))
    return torch.stack(rows,-2),(s,z)


def expanded(q,k,v,g,own_gate=False,causal=True):
    fq=torch.where(q<0,q.exp(),q+1);fk=torch.where(k<0,k.exp(),k+1)
    rows=[];n=q.shape[-2]
    for i in range(n):
        weights=[]
        for j in range(i+1 if causal else n):
            decay=g[...,j if own_gate else j+1:i+1].prod(-1)
            weights.append((fq[...,i,:]*fk[...,j,:]).sum(-1)*decay)
        w=torch.stack(weights,-1)
        rows.append(divide((w[...,None]*v[...,:len(weights),:]).sum(-2),w.sum(-1,keepdim=True)))
    return torch.stack(rows,-2)


def chunked(q,k,v,g,detach=False):
    a,state=recurrent(q[...,:1,:],k[...,:1,:],v[...,:1,:],g[...,:1])
    if detach:state=tuple(t.detach() for t in state)
    b,state=recurrent(q[...,1:,:],k[...,1:,:],v[...,1:,:],g[...,1:],state)
    return torch.cat((a,b),-2),state


def case(seed,n,m,dv):
    torch.manual_seed(seed)
    xs=[torch.randn(2,2,n,d,dtype=torch.float64,requires_grad=True) for d in (m,m,dv)]
    xs.append(torch.randn(2,2,n,dtype=torch.float64,requires_grad=True))
    q,k,v,a=xs;g=a.sigmoid();ref=expanded(q,k,v,g)
    probe=torch.randn_like(ref)/ref.numel();gr=torch.autograd.grad((ref*probe).sum(),xs,retain_graph=True)
    forms=[f(q,k,v,g) for f in (recurrent,chunked)]
    oe=max(error(ref,y) for y,_ in forms)
    ge=max(error(u,w) for y,_ in forms for u,w in zip(gr,torch.autograd.grad((y*probe).sum(),xs,retain_graph=True)))
    se=max(error(u,w) for u,w in zip(forms[0][1],forms[1][1]))
    assert max(oe,ge,se)<=1e-10 and bool((gr[-1][...,0]==0).all())
    ds=[torch.randn_like(x) for x in xs];length=sum(d.square().sum() for d in ds).sqrt();ds=[d/length for d in ds]
    with torch.no_grad():
        losses=[]
        for sign in (1,-1):
            p=[x+sign*1e-6*d for x,d in zip(xs,ds)]
            losses.append(float((expanded(*p[:3],p[3].sigmoid())*probe).sum()))
        fd=abs((losses[0]-losses[1])/2e-6-sum(float((u*d).sum()) for u,d in zip(gr,ds)))
        changed=[x.clone() for x in xs]
        for x in changed:x[..., -1, :] .add_(3) if x.ndim==4 else x[..., -1].add_(3)
        fe=error(expanded(*changed[:3],changed[3].sigmoid())[...,:-1,:],ref[...,:-1,:])
        fe=max(fe,error(recurrent(*changed[:3],changed[3].sigmoid())[0][...,:-1,:],forms[0][0][...,:-1,:]))
        ungated=error(recurrent(q,k,v,torch.ones_like(g))[0],pairwise(q,k,v,torch.ones(2,1,n,dtype=torch.bool)))
        zero=error(recurrent(q,k,v,torch.zeros_like(g))[0],v)
        assert fd<=1e-7 and max(fe,ungated,zero)<=1e-10
    return dict(seed=seed,n=n,m=m,dv=dv,output_error=oe,gradient_error=ge,state_error=se,directional_error=fd,future_error=fe,ungated_error=ungated,reset_error=zero,first_gate_gradient=0.)


def fixtures():
    q=torch.zeros(1,1,2,1,dtype=torch.float64);v=torch.tensor([2.,8.],dtype=torch.float64).reshape_as(q).requires_grad_();g=torch.tensor([[[1.,.5]]],dtype=torch.float64)
    good=recurrent(q,q,v,g)[0]
    detached=chunked(q,q,v,g,True)[0]
    ga=torch.autograd.grad(good[...,-1,:].sum(),v)[0];gb=torch.autograd.grad(detached[...,-1,:].sum(),v)[0]
    effects=dict(omit_decay=error(good,recurrent(q,q,v,torch.ones_like(g))[0]),omit_z_decay=error(good,recurrent(q,q,v,g,decay_z=False)[0]),include_own_gate=error(good,expanded(q,q,v,g,own_gate=True)),detach=error(ga,gb),omit_causal=error(good,expanded(q,q,v,g,causal=False)))
    assert all(x>1e-6 for x in effects.values()) and error(good,detached)==0
    overwrite=[]
    for value in (0.,.5,1.):
        gate=torch.tensor([[[1.,value]]],dtype=torch.float64)
        y=recurrent(q,q,v,gate)[0][...,-1,:]
        grad=torch.autograd.grad(y.sum(),v)[0][...,0,:].item()
        assert abs(y.item()-(2*value+8)/(value+1))<=1e-12 and abs(grad-value/(1+value))<=1e-12
        overwrite.append(dict(gate=value,output=y.item(),old_value_gradient=grad))
    # Ideal supplied feature basis, not finite ELU+1 raw inputs.
    keys=torch.tensor([[1.,0.],[0.,1.],[1.,0.]],dtype=torch.float64)
    interference=[]
    for last in (.5,0.):
        s=torch.zeros(2,dtype=torch.float64);z=s.clone()
        for key,val,gate in zip(keys,(2.,5.,8.),(1.,.5,last)):s=gate*s+key*val;z=gate*z+key
        a=divide(s[:1],z[:1]).item();b=None
        try:b=divide(s[1:],z[1:]).item()
        except ValueError:assert last==0
        assert abs(a-(6.8 if last==.5 else 8))<1e-12 and b==(5. if last==.5 else None)
        interference.append(dict(final_gate=last,A=a,B=b,B_denominator=float(z[1])))
    k=torch.tensor([0.,1.],dtype=torch.float64).reshape_as(q);states=[]
    for vals in ([0.,1.],[4.,0.]):
        out,state=recurrent(q,k,torch.tensor(vals,dtype=torch.float64).reshape_as(q),g)
        states.append(dict(S=state[0].item(),z=state[1].item(),read=out[...,-1,:].item()))
    assert states[0]==states[1]==dict(S=2.,z=2.5,read=.8)
    return dict(controls=effects,detach_output_error=error(good,detached),overwrite=overwrite,interference=interference,collision=states)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    p=ROOT/'results/gated_memory.json';assert not p.exists()
    history={str(x.relative_to(ROOT)):digest(x) for x in (ROOT/'results').glob('*.json')}
    files=['experiments/gated_memory.py','experiments/linear_attention.py','docs/protocols/gated-memory.md']
    r=dict(status='running',dtype='float64',device='cpu',torch=torch.__version__,training_updates=0,provenance={x:digest(ROOT/x) for x in files},historical=history,cases=[]);write(p,r)
    try:
        for seed,n,(m,dv) in itertools.product((2633,2647,2657),(3,7,17),((3,2),(5,4))):r['cases'].append(case(seed,n,m,dv))
        r.update(fixtures=fixtures());assert all(digest(ROOT/x)==h for x,h in history.items());r.update(status='complete',passed=True)
    except Exception as exc:r.update(status='failed',failure=repr(exc));raise
    finally:write(p,r)
    print({k:max(c[k] for c in r['cases']) for k in r['cases'][0] if k.endswith('error')});print(r['fixtures'])


if __name__=='__main__':main()
