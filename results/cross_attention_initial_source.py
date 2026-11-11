"""Finite CPU cross-attention equations, gradients and visibility contracts."""
import copy
import itertools
import json
import math
import platform
from pathlib import Path

import torch
from torch import nn
from position_distractors import digest
from recurrent_retrieval import write

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2309, 2311, 2333)


def native(model, target, memory, padding, mask=None):
    if bool(padding.all(-1).any()):
        raise ValueError('Each example needs at least one visible source key')
    return model(target, memory, memory, key_padding_mask=padding,
                 attn_mask=mask, need_weights=True, average_attn_weights=False)


def equations(model, target, memory, padding):
    """Separate projections and explicit per-example/per-head arithmetic."""
    q = target@model.q_proj_weight.T
    k = memory@model.k_proj_weight.T
    v = memory@model.v_proj_weight.T
    if model.in_proj_bias is not None:
        bq, bk, bv = model.in_proj_bias.chunk(3)
        q, k, v = q+bq, k+bk, v+bv
    width = model.embed_dim//model.num_heads
    batches, maps = [], []
    for b in range(len(target)):
        heads, weights = [], []
        for h in range(model.num_heads):
            features = slice(h*width, (h+1)*width)
            scores = q[b, :, features]@k[b, :, features].T/math.sqrt(width)
            scores = torch.where(padding[b][None, :], -torch.inf, scores)
            e = (scores-scores.max(-1, keepdim=True).values).exp()
            a = e/e.sum(-1, keepdim=True)
            heads.append(a@v[b, :, features])
            weights.append(a)
        batches.append(torch.cat(heads, -1))
        maps.append(torch.stack(weights))
    output = torch.stack(batches)@model.out_proj.weight.T
    if model.out_proj.bias is not None:
        output = output+model.out_proj.bias
    return output, torch.stack(maps)


def error(a, b):
    return (a-b).abs().max().item()


def decoder(self_attention, cross, target, memory, padding, causal=True):
    n = target.shape[1]
    mask = torch.ones(n, n, dtype=torch.bool).triu(1) if causal else None
    h = target+self_attention(target, target, target, attn_mask=mask)[0]
    return h+native(cross, h, memory, padding)[0]


def case(seed, target_length, source_length, bias, padded):
    torch.manual_seed(seed)
    model = nn.MultiheadAttention(8, 2, kdim=6, vdim=6, bias=bias,
                                 batch_first=True, dropout=0, dtype=torch.float64)
    target = torch.randn(2, target_length, 8, dtype=torch.float64, requires_grad=True)
    memory = torch.randn(2, source_length, 6, dtype=torch.float64, requires_grad=True)
    padding = torch.zeros(2, source_length, dtype=torch.bool)
    if padded:
        padding[1, -1] = True
    probe = torch.randn_like(target)/target.numel()
    output, weights = native(model, target, memory, padding)
    reference, expected_weights = equations(model, target, memory, padding)
    variables = (target, memory, *model.parameters())
    ga = torch.autograd.grad((output*probe).sum(), variables)
    gb = torch.autograd.grad((reference*probe).sum(), variables)
    out_error, weight_error = error(output, reference), error(weights, expected_weights)
    grad_error = max(error(a, b) for a, b in zip(ga, gb))
    assert max(out_error, weight_error, grad_error) <= 1e-10

    directions = [torch.randn_like(t) for t in variables]
    length = sum(d.square().sum() for d in directions).sqrt()
    directions = [d/length for d in directions]
    analytic = sum((g*d).sum() for g, d in zip(ga, directions)).item()
    originals = [t.detach().clone() for t in variables]
    scalars = []
    with torch.no_grad():
        for sign in (1, -1):
            for var, original, direction in zip(variables, originals, directions):
                var.copy_(original+sign*1e-6*direction)
            scalars.append((native(model, target, memory, padding)[0]*probe).sum().item())
        for var, original in zip(variables, originals):
            var.copy_(original)
    finite_difference = (scalars[0]-scalars[1])/2e-6
    derivative_error = abs(finite_difference-analytic)
    assert derivative_error <= 1e-7

    with torch.no_grad():
        permutation = torch.arange(source_length-1, -1, -1)
        permutation_error = error(output, native(model, target, memory[:, permutation], padding[:, permutation])[0])
        padding_error = padding_gradient = wrong_padding = None
        if padded:
            changed = memory.clone()
            changed[padding] += 100*torch.randn_like(changed[padding])
            padding_error = error(output, native(model, target, changed, padding)[0])
            padding_gradient = ga[1][padding].abs().max().item()
            wrong_padding = error(output, native(model, target, changed, torch.zeros_like(padding))[0])
            assert max(padding_error, padding_gradient) <= 1e-10 and wrong_padding > 1e-6
        wrong_cross = torch.ones(target_length, source_length, dtype=torch.bool).triu(1)
        wrong_cross_error = error(output, native(model, target, memory, padding, wrong_cross)[0])
        assert permutation_error <= 1e-10 and wrong_cross_error > 1e-6
        all_masked = padding.clone()
        all_masked[1] = True
        rejected = False
        try:
            native(model, target, memory, all_masked)
        except ValueError as exc:
            rejected = 'visible source key' in str(exc)
        assert rejected

        self_attention = nn.MultiheadAttention(8, 2, bias=bias, batch_first=True, dropout=0, dtype=torch.float64)
        cut = target_length//2
        changed_target = target.clone()
        changed_target[:, cut:] += 10*torch.randn_like(changed_target[:, cut:])
        before = decoder(self_attention, model, target, memory, padding)
        after = decoder(self_attention, model, changed_target, memory, padding)
        prefix_error = error(before[:, :cut], after[:, :cut])
        leaky_before = decoder(self_attention, model, target, memory, padding, causal=False)
        leaky_after = decoder(self_attention, model, changed_target, memory, padding, causal=False)
        leakage = error(leaky_before[:, :cut], leaky_after[:, :cut])
        assert prefix_error <= 1e-10 and leakage > 1e-6

    labels = torch.arange(1, 2*target_length+1).reshape(2, target_length)
    shifted = torch.cat((torch.zeros(2, 1, dtype=torch.int64), labels[:, :-1]), 1)
    def correct_shift(rows):
        return all(int(rows[b, t]) == (0 if t == 0 else int(labels[b, t-1]))
                   for b in range(2) for t in range(target_length))
    assert correct_shift(shifted) and not correct_shift(labels)
    return dict(seed=seed, target_length=target_length, source_length=source_length, bias=bias, padded=padded,
                output_error=out_error, weight_error=weight_error, gradient_error=grad_error,
                directional_derivative_error=derivative_error, permutation_error=permutation_error,
                padding_output_error=padding_error, padding_gradient=padding_gradient,
                ignored_padding_error=wrong_padding, wrong_cross_mask_error=wrong_cross_error,
                causal_prefix_error=prefix_error, missing_causal_mask_leakage=leakage,
                all_masked_rejected=rejected, unshifted_target_error_detected=True,
                differentiable_arrays=len(variables), differentiable_scalars=sum(t.numel() for t in variables))


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    path = ROOT/'results/cross_attention.json'
    assert not path.exists(), 'Preserve the completed study.'
    files = [Path(__file__), ROOT/'docs/protocols/cross-attention.md',
             ROOT/'experiments/position_distractors.py', ROOT/'experiments/recurrent_retrieval.py']
    report = dict(phase='running', torch=torch.__version__, python=platform.python_version(),
                  device='cpu', dtype='float64', provenance={str(p.relative_to(ROOT)):digest(p) for p in files}, cases=[])
    write(path, report)
    for seed, lengths, bias, padded in itertools.product(SEEDS, ((2,5),(5,3),(4,4)), (False,True), (False,True)):
        report['cases'].append(case(seed, *lengths, bias, padded))
        write(path, report)
    assert len(report['cases']) == 36
    names = ('output_error','weight_error','gradient_error','directional_derivative_error',
             'permutation_error','padding_output_error','padding_gradient','causal_prefix_error')
    report.update(phase='complete', passed=True,
        maxima={n:max(c[n] for c in report['cases'] if c[n] is not None) for n in names},
        negative_minima={n:min(c[n] for c in report['cases'] if c[n] is not None)
            for n in ('ignored_padding_error','wrong_cross_mask_error','missing_causal_mask_leakage')})
    write(path, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','provenance')}), flush=True)


if __name__ == '__main__':
    main()
