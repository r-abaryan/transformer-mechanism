"""Local/global visibility, causal safety, arithmetic and reachability."""
import itertools
import math
import torch
from sliding_window import dense as local_dense
from cross_attention import ROOT, error
from position_distractors import digest
from recurrent_retrieval import write


def patterns(n):
    return ((),(0,),(n//2,),(0,n-1))


def mask(n,r,g,causal):
    i=torch.arange(n)[:,None];j=torch.arange(n)[None,:]
    glob=torch.tensor([v in g for v in range(n)])
    visible=((i-j).abs()<=r)|glob[:,None]|glob[None,:]
    return visible & (j<=i) if causal else visible


def keys(n,r,g,c,i):
    return [j for j in range(n) if (not c or j<=i) and (i in g or j in g or i-r<=j<=i+r)]


def dense(q,k,v,visible):
    return (q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])).masked_fill(~visible,-torch.inf).softmax(-1)@v


def gathered(q,k,v,r,g,c):
    rows=[];n=q.shape[-2]
    for i in range(n):
        ix=keys(n,r,g,c,i)
        s=(q[:,:,i,None,:]*k[:,:,ix,:]).sum(-1)/math.sqrt(q.shape[-1])
        e=(s-s.max(-1,keepdim=True).values).exp()
        rows.append(((e/e.sum(-1,keepdim=True))[:,:,:,None]*v[:,:,ix,:]).sum(-2))
    return torch.stack(rows,-2)


def case(seed,n,r,g,c):
    torch.manual_seed(seed)
    xs=[torch.randn(2,2,n,d,dtype=torch.float64,requires_grad=True) for d in (3,3,2)]
    visible=mask(n,r,g,c);a=dense(*xs,visible);b=gathered(*xs,r,g,c)
    probe=torch.randn_like(a)/a.numel()
    ga=torch.autograd.grad((a*probe).sum(),xs);gb=torch.autograd.grad((b*probe).sum(),xs)
    oe=error(a,b);ge=max(error(x,y) for x,y in zip(ga,gb));assert max(oe,ge)<=1e-10
    ds=[torch.randn_like(x) for x in xs];scale=sum(d.square().sum() for d in ds).sqrt();ds=[d/scale for d in ds]
    with torch.no_grad():
        vals=[float((dense(*[x+sign*1e-6*d for x,d in zip(xs,ds)],visible)*probe).sum()) for sign in (1,-1)]
        fd=abs((vals[0]-vals[1])/2e-6-sum(float((x*y).sum()) for x,y in zip(ga,ds)))
        assert fd<=1e-7
        if not g:assert error(a,local_dense(*xs,r,c))==0
    return dict(seed=seed,n=n,r=r,globals=list(g),causal=c,output_error=oe,gradient_error=ge,directional_error=fd)


def reach(n,r,g,c,depth):
    a=mask(n,r,g,c);sets=[{i} for i in range(n)]
    for _ in range(depth):sets=[set().union(*(sets[j] for j in keys(n,r,g,c,i))) for i in range(n)]
    expected=torch.tensor([[j in s for j in range(n)] for s in sets])
    assert torch.equal(torch.linalg.matrix_power(a.double(),depth)>0,expected)
    def stack(x):
        for _ in range(depth):x=torch.stack([x[keys(n,r,g,c,i)].mean() for i in range(n)])
        return x
    jac=torch.autograd.functional.jacobian(stack,torch.arange(n,dtype=torch.float64))
    assert torch.equal(jac>0,expected) and bool((jac[~expected]==0).all())
    if c:assert not bool(expected.triu(1).any())
    if g and not c and depth>=2:assert bool(expected.all())
    return dict(n=n,r=r,globals=list(g),causal=c,depth=depth,reachable_pairs=int(expected.sum()),sources=[sorted(s) for s in sets],outside_gradient=0.)


def controls():
    n=9;i=torch.arange(n)[:,None];j=torch.arange(n)[None,:]
    correct=mask(n,1,(0,),False);local=(i-j).abs()<=1
    no_query=local|(j==0);no_key=local|(i==0)
    def avg(a):return a.double()/a.sum(-1,keepdim=True)
    impulse=torch.zeros(n,dtype=torch.float64);impulse[8]=9
    query_effect=float(abs((avg(correct)@impulse)[0]-(avg(no_query)@impulse)[0]))
    impulse.zero_();impulse[0]=9
    key_effect=float(abs((avg(correct)@impulse)[4]-(avg(no_key)@impulse)[4]))
    causal=mask(n,1,(0,),True);unsafe=causal|(i==0)
    impulse.zero_();impulse[8]=9
    good=avg(causal);bad=avg(unsafe)
    direct=float((bad@impulse)[4]);leak=float((bad@bad@impulse)[4]);safe=float((good@good@impulse)[4])
    derivative=torch.autograd.functional.jacobian(lambda x:good@good@x,torch.zeros(n,dtype=torch.float64))[4,8].item()
    relay=avg(mask(n,1,(4,),True));relay_weight=float((relay@relay)[8,0])
    assert min(query_effect,key_effect,leak,relay_weight)>1e-6
    assert direct==safe==derivative==0
    return dict(passed=True,omitted_global_query_effect=query_effect,omitted_global_key_effect=key_effect,
        unsafe_one_layer_effect=direct,unsafe_two_layer_effect=leak,safe_two_layer_effect=safe,safe_future_derivative=derivative,
        causal_middle_relay_derivative=relay_weight)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    target=ROOT/'results/global_attention.json';assert not target.exists()
    history={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'results').glob('*.json')}
    files=['experiments/global_attention.py','experiments/sliding_window.py','docs/protocols/global-attention.md']
    report=dict(status='running',device='cpu',dtype='float64',threads=1,torch=torch.__version__,training_updates=0,
        provenance={p:digest(ROOT/p) for p in files},historical=history,cases=[])
    write(target,report)
    try:
        for seed,n,r,c in itertools.product((2473,2477,2503),(5,9),(0,2),(False,True)):
            for g in patterns(n):report['cases'].append(case(seed,n,r,g,c))
        report['reach']=[reach(n,r,g,c,d) for n,r,c,d in itertools.product((5,9),(0,2),(False,True),(1,2,3)) for g in patterns(n)]
        counts=[]
        for n,r,c in itertools.product((5,9,128,1024),(0,2),(False,True)):
            for g in patterns(n):
                enumerated=sum(len(keys(n,r,g,c,i)) for i in range(n));partition=0
                for i in range(n):
                    hi=i if c else n-1
                    if i in g:partition+=hi+1
                    else:
                        lo=max(0,i-r);end=min(hi,i+r)
                        partition+=end-lo+1+sum(j<=hi and not lo<=j<=end for j in g)
                assert enumerated==partition==int(mask(n,r,g,c).sum())
                local=n*(r+1)-r*(r+1)//2 if c else n*(2*r+1)-r*(r+1)
                counts.append(dict(n=n,r=r,globals=list(g),causal=c,pairs=partition,local_pairs=local,full_pairs=n*(n+1)//2 if c else n*n))
        report.update(counts=counts,controls=controls())
        for p,h in history.items():assert digest(ROOT/p)==h
        report.update(status='complete',passed=True)
    except Exception as exc:
        report.update(status='failed',failure=repr(exc));raise
    finally:write(target,report)
    print({k:len(report[k]) for k in ('cases','reach','counts')})
    print({k:max(x[k] for x in report['cases']) for k in ('output_error','gradient_error','directional_error')})
    print(report['controls'])


if __name__=='__main__':main()
