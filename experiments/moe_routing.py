"""Bounded top-one MoE arithmetic audit; no training."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform

import torch

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dispatch(xs, valid, factor, mode="correct"):
    x, router, first, second = xs
    experts = router.shape[1]
    cap = math.ceil(factor * int(valid.sum()) / experts)
    probs = (x @ router).softmax(-1)
    chosen = probs.argmin(-1) if mode == "wrong_expert" else probs.argmax(-1)
    mask = torch.nn.functional.one_hot(chosen, experts).bool()
    eligible = mask if mode == "padding_slot" else mask & valid[:, None]
    slots = eligible.long().cumsum(0)
    accepted = mask & valid[:, None] & (slots <= cap)
    if mode == "no_capacity":
        accepted = mask & valid[:, None]
    y = x * 0 + sum(t.sum() * 0 for t in xs[1:])
    for e in range(experts):
        ids = accepted[:, e].nonzero().flatten()
        values = (x[ids] @ first[e]).tanh() @ second[e]
        gate = probs[ids, e, None]
        if mode == "no_gate":
            gate = torch.ones_like(gate)
        if mode == "detach_gate":
            gate = gate.detach()
        y = y.index_add(0, ids, gate * values)
    return y, chosen, accepted


def token_reference(xs, valid, factor):
    x, router, first, second = xs
    count = [0] * router.shape[1]
    cap = math.ceil(factor * int(valid.sum()) / len(count))
    rows = []
    for t in range(len(x)):
        zero = x[t] * 0 + sum(w.sum() * 0 for w in xs[1:])
        if not bool(valid[t]):
            rows.append(zero)
            continue
        logits = torch.stack([(x[t] * router[:, e]).sum() for e in range(len(count))])
        weights = (logits - logits.max()).exp()
        probs = weights / weights.sum()
        e = max(range(len(count)), key=lambda i: float(logits[i].detach()))
        count[e] += 1
        if count[e] > cap:
            rows.append(zero)
            continue
        hidden = torch.stack([(x[t] * first[e, :, j]).sum() for j in range(first.shape[-1])]).tanh()
        out = torch.stack([(hidden * second[e, :, j]).sum() for j in range(x.shape[1])])
        rows.append(probs[e] * out)
    return torch.stack(rows)


def error(a, b):
    return float((a - b).detach().abs().max())


def case(seed, tokens, experts, factor):
    gen = torch.Generator().manual_seed(seed)
    xs = [(torch.randn(shape, generator=gen, dtype=torch.float64) * .3).requires_grad_()
          for shape in [(tokens, 4), (4, experts), (experts, 4, 6), (experts, 6, 4)]]
    valid = torch.arange(tokens) < tokens - 1
    y, chosen, accepted = dispatch(xs, valid, factor)
    ref = token_reference(xs, valid, factor)
    probe = torch.randn(y.shape, generator=gen, dtype=y.dtype) / y.numel()
    grad = torch.autograd.grad((y * probe).sum(), xs, retain_graph=True)
    rg = torch.autograd.grad((ref * probe).sum(), xs)
    dirs = [torch.randn(t.shape, generator=gen, dtype=t.dtype) for t in xs]
    norm = sum(d.square().sum() for d in dirs).sqrt()
    dirs = [d / norm for d in dirs]
    values = []
    stable = True
    with torch.no_grad():
        for sign in [-1, 1]:
            py, pc, pa = dispatch([t + sign * 1e-6 * d for t, d in zip(xs, dirs)], valid, factor)
            stable &= torch.equal(chosen, pc) and torch.equal(accepted, pa)
            values.append(float((py * probe).sum()))
        changed = [t.detach().clone() for t in xs]
        changed[0][-1] += 100
        pad_error = error(dispatch(changed, valid, factor)[0], y)
    oe = error(y, ref)
    ge = max(error(a, b) for a, b in zip(grad, rg))
    fd = abs((values[1] - values[0]) / 2e-6 - sum(float((g*d).sum()) for g, d in zip(grad, dirs)))
    return dict(seed=seed, tokens=tokens, experts=experts, factor=factor,
                capacity=math.ceil(factor*(tokens-1)/experts), accepted=int(accepted.sum()),
                output_error=oe, gradient_error=ge, finite_difference_error=fd,
                stable_selection=stable, padding_error=pad_error,
                passed=oe <= 1e-10 and ge <= 1e-10 and fd <= 1e-7 and stable and pad_error == 0)


def fixtures():
    xs = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in
          [[[1.], [1.], [1.]], [[2., 0.]], [[[1.]], [[2.]]], [[[3.]], [[5.]]]]]
    valid = torch.tensor([False, True, True])
    y, chosen, accepted = dispatch(xs, valid, 1.)
    effects = {mode: error(y, dispatch(xs, valid, 1., mode)[0])
               for mode in ["wrong_expert", "no_gate", "no_capacity", "padding_slot"]}
    native = torch.autograd.grad(y.sum(), xs, retain_graph=True)
    bad = torch.autograd.grad(dispatch(xs, valid, 1., "detach_gate")[0].sum(), xs)
    effects["detach_gate"] = error(native[1], bad[1])
    tied = [t.detach().clone() for t in xs]
    tied[1].zero_()
    ty, tc, ta = dispatch(tied, valid, 1.)
    empty = dispatch(xs, torch.zeros(3, dtype=torch.bool), 1.)[0]
    residual_error = error((xs[0] + y)[2], xs[0][2])
    unselected = max(float(native[2][1].abs().max()), float(native[3][1].abs().max()))
    return dict(output=y.detach().flatten().tolist(), accepted=accepted.tolist(),
                negative_control_effects=effects, tie_chooses_zero=bool((tc == 0).all()),
                tie_first_valid_only=ta[:, 0].tolist() == [False, True, False],
                empty_output_error=float(empty.abs().max().detach()),
                overflow_residual_error=residual_error, unselected_expert_gradient=unselected,
                router_gradient_max=float(native[1].abs().max()),
                passed=all(v > 1e-6 for v in effects.values()) and bool((tc == 0).all())
                and ta[:, 0].tolist() == [False, True, False] and float(empty.abs().max().detach()) == 0
                and residual_error == 0 and unselected == 0 and float(native[1].abs().max()) > 1e-6)


def main():
    torch.set_num_threads(1)
    destination = ROOT / "results/moe_routing.json"
    if destination.exists():
        raise FileExistsError("Preserve the executed report; choose an explicit diagnostic instead")
    prior = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "results").glob("*.json")}
    cases = [case(*v) for v in itertools.product([2741, 2749, 2753], [5, 9], [2, 3], [.5, 1.5])]
    witness = fixtures()
    counts = [dict(experts=e, router=4*e, all_experts=e*2*4*6,
                   total=4*e+e*2*4*6, per_accepted_token=4*e+2*4*6) for e in [2, 3]]
    unchanged = all(digest(ROOT / p) == h for p, h in prior.items())
    report = dict(python=platform.python_version(), torch=torch.__version__, device="cpu", dtype="float64",
                  source_sha256=digest(Path(__file__)), protocol_sha256=digest(ROOT / "docs/protocols/moe-routing.md"),
                  cases=cases, fixtures=witness, parameter_counts=counts,
                  preserved_json_count=len(prior), preserved_json_sha256=prior,
                  passed=all(c["passed"] for c in cases) and witness["passed"] and unchanged)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "cases": len(cases), "fixtures": witness,
                      "max_errors": {key: max(c[key] for c in cases) for key in
                                     ["output_error", "gradient_error", "finite_difference_error"]},
                      "parameter_counts": counts, "preserved": len(prior)}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
