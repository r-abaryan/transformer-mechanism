"""Frozen positive-kernel attention arithmetic audit, without training."""
import itertools
import torch
from cross_attention import ROOT, error
from position_distractors import digest
from recurrent_retrieval import write


def phi(x):
    return torch.where(x >= 0, x + 1, torch.exp(torch.minimum(x, torch.zeros_like(x))))


def divide(num, den):
    if not bool(torch.isfinite(den).all() and (den > 0).all()):
        raise ValueError('Nonpositive or nonfinite kernel denominator')
    return num / den


def pairwise(q, k, v, valid):
    # Separate feature expression and explicit pair weights serve as reference.
    fq = torch.where(q < 0, q.exp(), 1 + q)
    fk = torch.where(k < 0, k.exp(), 1 + k)
    n = q.shape[-2]
    visible = torch.ones(n, n, dtype=torch.bool).tril() & valid[..., None, :]
    weights = (fq @ fk.transpose(-1, -2)).masked_fill(~visible, 0)
    return divide(weights @ v, weights.sum(-1, keepdim=True))


def parallel(q, k, v, valid):
    fk = phi(k) * valid[..., None]
    states = (fk[..., :, None] * v[..., None, :]).cumsum(-3)
    totals = fk.cumsum(-2)
    fq = phi(q)
    return divide((fq[..., :, None] * states).sum(-2),
                  (fq * totals).sum(-1, keepdim=True)), (states[..., -1, :, :], totals[..., -1, :])


def recurrent(q, k, v, valid, state=None):
    if state is None:
        state = (q.new_zeros(*q.shape[:-2], q.shape[-1], v.shape[-1]),
                 q.new_zeros(*q.shape[:-2], q.shape[-1]))
    s, z = state
    rows = []
    for i in range(q.shape[-2]):
        key = phi(k[..., i, :]) * valid[..., i, None]
        s = s + key[..., :, None] * v[..., i, None, :]
        z = z + key
        query = phi(q[..., i, :])
        rows.append(divide((query[..., :, None] * s).sum(-2),
                           (query * z).sum(-1, keepdim=True)))
    return torch.stack(rows, -2), (s, z)


def chunked(q, k, v, valid, detach=False):
    first, state = recurrent(q[..., :1, :], k[..., :1, :], v[..., :1, :], valid[..., :1])
    if detach:
        state = tuple(t.detach() for t in state)
    rest, state = recurrent(q[..., 1:, :], k[..., 1:, :], v[..., 1:, :], valid[..., 1:], state)
    return torch.cat((first, rest), -2), state


def case(seed, n, m, dv, masked):
    torch.manual_seed(seed)
    xs = [torch.randn(2, 2, n, d, dtype=torch.float64, requires_grad=True) for d in (m, m, dv)]
    valid = torch.ones(2, 1, n, dtype=torch.bool)
    if masked:
        valid[..., 1] = False
    ref = pairwise(*xs, valid)
    probe = torch.randn_like(ref) / ref.numel()
    grads = torch.autograd.grad((ref * probe).sum(), xs)
    forms = [f(*xs, valid) for f in (parallel, recurrent, chunked)]
    oe = max(error(ref, y) for y, _ in forms)
    ge = max(error(a, b) for y, _ in forms for a, b in zip(grads, torch.autograd.grad((y * probe).sum(), xs)))
    se = max(error(a, b) for _, state in forms[1:] for a, b in zip(forms[0][1], state))
    assert max(oe, ge, se) <= 1e-10
    directions = [torch.randn_like(x) for x in xs]
    norm = sum(d.square().sum() for d in directions).sqrt()
    directions = [d / norm for d in directions]
    with torch.no_grad():
        values = [float((pairwise(*[x + sign * 1e-6 * d for x, d in zip(xs, directions)], valid) * probe).sum()) for sign in (1, -1)]
        fd = abs((values[0] - values[1]) / 2e-6 - sum(float((g * d).sum()) for g, d in zip(grads, directions)))
        assert fd <= 1e-7
        changed = [x.clone() for x in xs]
        for x in changed:
            x[..., -1, :] += 7
        future = max(error(f(*changed, valid)[0][..., :-1, :], y[..., :-1, :]) for f, (y, _) in zip((parallel, recurrent, chunked), forms))
        future = max(future, error(pairwise(*changed, valid)[..., :-1, :], ref[..., :-1, :]))
        assert future <= 1e-10
    invalid = max(float(g[..., 1, :].abs().max()) for g in grads[1:]) if masked else 0.
    assert invalid == 0
    return dict(seed=seed,n=n,m=m,dv=dv,masked=masked,output_error=oe,gradient_error=ge,state_error=se,directional_error=fd,future_error=future,invalid_gradient=invalid)


def controls():
    q = torch.zeros(1, 1, 2, 1, dtype=torch.float64)
    v = torch.tensor([2., 8.], dtype=torch.float64).reshape_as(q).requires_grad_()
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    good, state = recurrent(q, q, v, valid)
    reset = recurrent(q[..., 1:, :], q[..., 1:, :], v[..., 1:, :], valid[..., 1:])[0]
    detached = chunked(q, q, v, valid, detach=True)[0]
    ga = torch.autograd.grad(good[..., -1, :].sum(), v)[0]
    gb = torch.autograd.grad(detached[..., -1, :].sum(), v)[0]
    q3 = torch.zeros(1, 1, 3, 1, dtype=torch.float64)
    v3 = torch.tensor([2., 100., 8.], dtype=torch.float64).reshape_as(q3)
    mask = torch.tensor([[[True, False, True]]])
    effects = dict(omit_denominator=abs(float(state[0].item())-5),
        reset_state=abs(float(reset.item())-5),detach_old_value_gradient=error(ga,gb),
        omit_mask=error(recurrent(q3,q3,v3,mask)[0],recurrent(q3,q3,v3,torch.ones_like(mask))[0]),
        exclude_current_value=abs(float(good[...,0,:].item())-5))
    assert all(x > 1e-6 for x in effects.values())
    assert error(good,detached)==0
    rejected = {}
    for name, x, mask2 in [('empty_prefix',q,torch.zeros_like(valid)),('underflow',q-1000,valid)]:
        try:
            recurrent(x,x,v,mask2)
        except ValueError:
            rejected[name]=True
        else:
            raise AssertionError(name)
    return dict(effects=effects,detach_output_error=error(good,detached),rejected=rejected)


def witnesses():
    k = torch.tensor([0.,1.],dtype=torch.float64)
    a = torch.tensor([0.,1.],dtype=torch.float64)
    b = torch.tensor([2.,0.],dtype=torch.float64)
    s = [float((phi(k)*v).sum()) for v in (a,b)]
    z = float(phi(k).sum())
    reads = [float((phi(torch.tensor(q))*s[0])/(phi(torch.tensor(q))*z)) for q in (-2.,0.,1.,2.)]
    soft = [float(k.softmax(0)@v) for v in (a,b)]
    assert s == [2.,2.] and z == 3. and abs(soft[0]-soft[1])>1e-6
    return dict(kernel_at_zero=s[0]/z,softmax_at_zero=float(torch.zeros_like(k).softmax(0)@a),S=s,z=z,query_reads=reads,softmax_at_one=soft)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    target=ROOT/'results/linear_attention.json'
    assert not target.exists()
    history={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'results').glob('*.json')}
    files=['experiments/linear_attention.py','docs/protocols/linear-attention.md']
    report=dict(status='running',device='cpu',dtype='float64',threads=1,torch=torch.__version__,training_updates=0,provenance={p:digest(ROOT/p) for p in files},historical=history,cases=[])
    write(target,report)
    try:
        for args in itertools.product((2551,2557,2579),(3,7,17),(3,5),(2,4),(False,True)):
            report['cases'].append(case(*args))
        report.update(controls=controls(),witnesses=witnesses())
        report['counts']=[dict(n=n,m=m,dv=dv,state_scalars=m*dv+m,kv_scalars=n*(m+dv),state_bytes=8*(m*dv+m),kv_bytes=8*n*(m+dv)) for n,(m,dv) in itertools.product((16,128,1024),((4,3),(64,64)))]
        assert all(digest(ROOT/p)==h for p,h in history.items())
        report.update(status='complete',passed=True)
    except Exception as exc:
        report.update(status='failed',failure=repr(exc))
        raise
    finally:
        write(target,report)
    print({k:max(c[k] for c in report['cases']) for k in ('output_error','gradient_error','state_error','directional_error','future_error')})
    print(report['controls']);print(report['witnesses'])


if __name__=='__main__':
    main()
