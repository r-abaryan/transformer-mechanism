"""Fixed local attention arithmetic and structural reach; no training."""
import itertools
import json
import math
import torch
from cross_attention import ROOT, error
from position_distractors import digest
from recurrent_retrieval import write


def visibility(n,r,causal):
    i=torch.arange(n)[:,None]
    j=torch.arange(n)[None,:]
    return ((i-j).abs()<=r) & ((j<=i) if causal else torch.ones(n,n,dtype=torch.bool))


def dense(q,k,v,r,causal):
    scores=q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])
    return scores.masked_fill(~visibility(q.shape[-2],r,causal),-torch.inf).softmax(-1)@v


def gathered(q,k,v,r,causal):
    rows=[]
    n=q.shape[-2]
    for i in range(n):
        lo=max(0,i-r)
        hi=i+1 if causal else min(n,i+r+1)
        scores=(q[:,:,i,None,:]*k[:,:,lo:hi,:]).sum(-1)/math.sqrt(q.shape[-1])
        e=(scores-scores.max(-1,keepdim=True).values).exp()
        rows.append(((e/e.sum(-1,keepdim=True))[:,:,:,None]*v[:,:,lo:hi,:]).sum(-2))
    return torch.stack(rows,-2)


def case(seed,n,r,causal):
    torch.manual_seed(seed)
    variables=[torch.randn(2,2,n,d,dtype=torch.float64,requires_grad=True) for d in (3,3,2)]
    a=dense(*variables,r,causal)
    b=gathered(*variables,r,causal)
    probe=torch.randn_like(a)/a.numel()
    ga=torch.autograd.grad((a*probe).sum(),variables)
    gb=torch.autograd.grad((b*probe).sum(),variables)
    oe=error(a,b); ge=max(error(x,y) for x,y in zip(ga,gb))
    assert max(oe,ge)<=1e-10
    ds=[torch.randn_like(x) for x in variables]
    norm=sum(d.square().sum() for d in ds).sqrt()
    ds=[d/norm for d in ds]
    with torch.no_grad():
        vals=[(dense(*[x+sign*1e-6*d for x,d in zip(variables,ds)],r,causal)*probe).sum().item() for sign in (1,-1)]
        fd=abs((vals[0]-vals[1])/2e-6-sum((g*d).sum().item() for g,d in zip(ga,ds)))
        full=error(a,dense(*variables,n-1,causal))
    assert fd<=1e-7
    if r==n-1:assert full<=1e-10
    if r==0:assert error(a,variables[2])==0
    return dict(seed=seed,length=n,radius=r,causal=causal,output_error=oe,gradient_error=ge,directional_error=fd,full_attention_difference=full)


def reach(n,r,depth,causal):
    a=visibility(n,r,causal)
    power=torch.linalg.matrix_power(a.double(),depth)>0
    sets=[{i} for i in range(n)]
    for _ in range(depth):
        sets=[set().union(*(sets[j] for j in range(max(0,i-r),i+1 if causal else min(n,i+r+1)))) for i in range(n)]
    expected=torch.tensor([[j in sets[i] for j in range(n)] for i in range(n)])
    formula=torch.tensor([[max(0,i-depth*r)<=j<=(i if causal else min(n-1,i+depth*r)) for j in range(n)] for i in range(n)])
    assert torch.equal(power,expected) and torch.equal(expected,formula)
    # A constructive attention instance: Q=K=0 makes all allowed weights uniform.
    def uniform_stack(x):
        for _ in range(depth):
            x=torch.stack([x[max(0,i-r):i+1 if causal else min(n,i+r+1)].mean() for i in range(n)])
        return x
    jac=torch.autograd.functional.jacobian(uniform_stack,torch.arange(n,dtype=torch.float64))
    assert torch.equal(jac>0,expected)
    outside=jac[~expected]
    assert not outside.numel() or float(outside.abs().max())==0
    return dict(length=n,radius=r,depth=depth,causal=causal,reachable_pairs=int(expected.sum()),
        last_token_sources=sorted(sets[-1]),minimum_reachable_gradient=float(jac[expected].min()),outside_gradient=0.)


def controls():
    q=k=torch.zeros(1,1,5,3,dtype=torch.float64)
    v=torch.zeros(1,1,5,2,dtype=torch.float64)
    v[:,:,4]=10
    omitted=error(dense(q,k,v,1,False)[:,:,0],dense(q,k,v,4,False)[:,:,0])
    future=error(dense(q,k,v,1,True)[:,:,3],dense(q,k,v,1,False)[:,:,3])
    v.zero_();v[:,:,1]=10
    boundary=error(dense(q,k,v,1,False)[:,:,0],dense(q,k,v,0,False)[:,:,0])
    assert min(omitted,future,boundary)>1e-6
    return dict(omitted_window=omitted,omitted_causality=future,off_by_one_radius=boundary,passed=True)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    target=ROOT/'results/sliding_window.json'
    assert not target.exists(),'Preserve prior attempts.'
    history={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'results').glob('*.json')}
    files=['experiments/sliding_window.py','docs/protocols/sliding-window.md','experiments/cross_attention.py']
    report=dict(status='running',device='cpu',dtype='float64',torch=torch.__version__,training_updates=0,
        historical=history,provenance={p:digest(ROOT/p) for p in files},cases=[])
    write(target,report)
    try:
        for seed,n,causal in itertools.product((2411,2417,2423),(5,9),(False,True)):
            for r in (0,1,2,n-1):
                report['cases'].append(case(seed,n,r,causal))
        report['controls']=controls()
        report['reach']=[reach(9,r,l,c) for r,l,c in itertools.product((0,1,2),(1,2,3),(False,True))]
        counts=[]
        for n,requested,c in itertools.product((5,9,128,1024,4096),(0,1,2,4),(False,True)):
            r=min(requested,n-1)
            enumerated=sum((i+1 if c else min(n,i+r+1))-max(0,i-r) for i in range(n))
            formula=n*(r+1)-r*(r+1)//2 if c else n*(2*r+1)-r*(r+1)
            assert enumerated==formula
            counts.append(dict(length=n,radius=r,causal=c,pairs=formula,full_pairs=n*(n+1)//2 if c else n*n))
        report['counts']=counts
        for p,h in history.items():assert digest(ROOT/p)==h,p
        report.update(status='complete',passed=True)
    except Exception as exc:
        report.update(status='failed',failure=repr(exc))
        raise
    finally:write(target,report)
    print(json.dumps(dict(cases=len(report['cases']),reach=len(report['reach']),counts=len(counts),
        maxima={k:max(c[k] for c in report['cases']) for k in ('output_error','gradient_error','directional_error')},controls=report['controls'])))


if __name__=='__main__':main()
