"""Switch-style auxiliary-loss arithmetic and counterexamples, no training."""
import itertools
import json
import math
import platform

import torch

from moe_routing import ROOT, digest, dispatch, error


def balance(z, valid, alpha=1., mode="correct"):
    if not bool(valid.any()):
        raise ValueError("No valid tokens for balancing means")
    used = torch.ones_like(valid) if mode == "padding" else valid
    p = z[used].softmax(-1)
    e = z.shape[1]
    counts = torch.nn.functional.one_hot(p.argmax(-1), e).sum(0)
    f = counts.to(z.dtype) / len(p)
    mean = p.mean(0)
    if mode == "soft_counts":
        f = mean
    if mode == "post_capacity":
        f = counts.clamp(max=math.ceil(len(p)/e)).to(z.dtype) / len(p)
    if mode == "detach":
        mean = mean.detach() + z.sum() * 0
    loss = alpha * (1 if mode == "omit_e" else e) * (f * mean).sum()
    return loss, f, mean


def reference(z, valid, alpha):
    rows = [row for row, keep in zip(z, valid) if bool(keep)]
    e = z.shape[1]
    counts = [0] * e
    probabilities = []
    for row in rows:
        choice = max(range(e), key=lambda j: float(row[j].detach()))
        counts[choice] += 1
        weights = [(v-row.max()).exp() for v in row]
        probabilities.append([v/sum(weights) for v in weights])
    return alpha * e * sum(counts[j]/len(rows) * sum(p[j] for p in probabilities)/len(rows) for j in range(e))


def case(seed, n, e, alpha):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g, dtype=torch.float64, requires_grad=True)
    w = torch.randn(4, e, generator=g, dtype=torch.float64, requires_grad=True)
    valid = torch.arange(n) < n-1
    z = x @ w
    loss, f, _ = balance(z, valid, alpha)
    grads = torch.autograd.grad(loss, [z, x, w], retain_graph=True)
    ref = reference(z, valid, alpha)
    rg = torch.autograd.grad(ref, [z, x, w])
    with torch.no_grad():
        p = z.softmax(-1)
        gz = alpha*e/(n-1)*p*(f-(p*f).sum(-1, keepdim=True))
        gz[~valid] = 0
        analytical = [gz, gz @ w.T, x.T @ gz]
        dx = torch.randn(x.shape, generator=g, dtype=x.dtype)
        dw = torch.randn(w.shape, generator=g, dtype=w.dtype)
        norm = (dx.square().sum()+dw.square().sum()).sqrt()
        dx, dw = dx/norm, dw/norm
        vals = []
        stable = True
        for sign in [-1, 1]:
            zp = (x+sign*1e-6*dx) @ (w+sign*1e-6*dw)
            stable &= torch.equal(zp[valid].argmax(-1), z[valid].argmax(-1))
            vals.append(float(balance(zp, valid, alpha)[0]))
        fd = abs((vals[1]-vals[0])/2e-6-float((grads[1]*dx).sum()+(grads[2]*dw).sum()))
        changed=z.clone(); changed[~valid]+=torch.arange(e)*100
        padding_error=error(loss,balance(changed,valid,alpha)[0])
        scale_error=error(loss*2,balance(z,valid,alpha*2)[0])
    le=error(loss,ref)
    ge=max(error(a,b) for a,b in zip(grads,rg))
    ae=max(error(a,b) for a,b in zip(grads,analytical))
    return dict(seed=seed,tokens=n,experts=e,alpha=alpha,loss=float(loss.detach()),loss_error=le,
                gradient_error=ge,analytical_error=ae,finite_difference_error=fd,stable=stable,
                padding_error=padding_error,scale_error=scale_error,
                passed=max(le,ge,ae)<=1e-10 and fd<=1e-7 and stable and padding_error==0 and scale_error==0)


def fixtures():
    definitions={"balanced":[[.8,.2],[.2,.8]]*2,"concentrated":[[.9,.1]]*4,
                 "uniform_tie":[[.5,.5]]*4,"uniform_mean":[[.6,.4]]*3+[[.2,.8]],
                 "below_one":[[.51,.49]]*3+[[.01,.99]]}
    rows=[]
    for name, values in definitions.items():
        z=torch.tensor(values,dtype=torch.float64).log().requires_grad_()
        valid=torch.ones(4,dtype=torch.bool)
        loss,f,p=balance(z,valid)
        grad=torch.autograd.grad(loss,z)[0]
        counts=[sum(max(range(2),key=lambda j: row[j])==i for row in values) for i in range(2)]
        expected_loss=2*sum(counts[i]/4*sum(row[i] for row in values)/4 for i in range(2))
        # Existing dispatch takes xW logits: identity x and router z preserve this fixture.
        xs=[torch.eye(4,dtype=z.dtype),z.detach(),torch.ones(2,4,1,dtype=z.dtype),torch.ones(2,1,4,dtype=z.dtype)]
        _,_,accepted=dispatch(xs,valid,1.)
        overflow=sum(max(c-2,0) for c in counts)
        passed=abs(float(loss.detach())-expected_loss)<1e-12 and int(accepted.sum())==4-overflow
        if name=="balanced":passed &= float(grad.abs().max())<1e-12
        rows.append(dict(name=name,f=f.tolist(),P=p.detach().tolist(),loss=float(loss.detach()),
                         gradient_max=float(grad.abs().max()),overflow=overflow,passed=passed))
    z=torch.tensor([[.8,.2],[.7,.3],[.6,.4],[.1,.9],[.01,.99]],dtype=torch.float64).log().requires_grad_()
    valid=torch.tensor([True]*4+[False])
    loss=balance(z,valid)[0]; grad=torch.autograd.grad(loss,z)[0]
    controls={}
    for mode in ["omit_e","soft_counts","post_capacity","padding","detach"]:
        bad=balance(z,valid,mode=mode)[0]; bg=torch.autograd.grad(bad,z)[0]
        controls[mode]=dict(loss_change=error(loss,bad),gradient_change=error(grad,bg))
    rejected=False
    try:balance(z,torch.zeros(5,dtype=torch.bool))
    except ValueError:rejected=True
    return dict(rows=rows,controls=controls,empty_rejected=rejected,
                passed=all(r['passed'] for r in rows) and rejected and
                all(max(c.values())>1e-6 for c in controls.values()))


def main():
    torch.set_num_threads(1)
    path=ROOT/'results/moe_balancing.json'
    if path.exists():raise FileExistsError('Preserve executed report')
    old={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'results').glob('*.json')}
    cases=[case(*v) for v in itertools.product([2777,2789,2791],[5,9],[2,3],[.01,1.])]
    witness=fixtures()
    report=dict(python=platform.python_version(),torch=torch.__version__,device='cpu',dtype='float64',
                source_sha256=digest(ROOT/'experiments/moe_balancing.py'),
                dependency_sha256=digest(ROOT/'experiments/moe_routing.py'),
                protocol_sha256=digest(ROOT/'docs/protocols/moe-balancing.md'),cases=cases,fixtures=witness,
                preserved_json_sha256=old,
                passed=all(c['passed'] for c in cases) and witness['passed'] and all(digest(ROOT/p)==h for p,h in old.items()))
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],cases=len(cases),fixtures=witness,
                         maxima={k:max(c[k] for c in cases) for k in ['loss_error','gradient_error','analytical_error','finite_difference_error']},
                         preserved=len(old)),indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
