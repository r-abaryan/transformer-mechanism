"""Full finite encoder-decoder arithmetic and input/loss contract; no training."""
import itertools
import json
import math
import platform
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from position_distractors import pe, digest
from audit_kv_cache_equivalence import norm
from recurrent_retrieval import write

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2341, 2347, 2351)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.source = nn.Embedding(16, 8)
        self.target = nn.Embedding(16, 8)
        self.encoder = nn.TransformerEncoderLayer(8, 2, 16, dropout=0, batch_first=True)
        self.decoder = nn.TransformerDecoderLayer(8, 2, 16, dropout=0, batch_first=True)
        self.output = nn.Linear(8, 16, bias=False)
        self.double()

    def features(self, source, target):
        return (self.source(source)*math.sqrt(8)+pe(torch.arange(source.shape[1]), 8),
                self.target(target)*math.sqrt(8)+pe(torch.arange(target.shape[1]), 8))

    def forward(self, source, target, sp, tp, encoder_mask=True, memory_mask=True, causal=True):
        if bool(sp.all(-1).any()):
            raise ValueError('Empty source')
        memory = self.encoder(source, src_key_padding_mask=sp if encoder_mask else None)
        mask = torch.ones(target.shape[1], target.shape[1], dtype=torch.bool).triu(1) if causal else None
        hidden = self.decoder(target, memory, tgt_mask=mask, tgt_key_padding_mask=tp,
                              memory_key_padding_mask=sp if memory_mask else None)
        return self.output(hidden)


def attention(module, query, key, value, padding, causal=False):
    wq, wk, wv = module.in_proj_weight.chunk(3)
    bq, bk, bv = module.in_proj_bias.chunk(3)
    batch, n, width = query.shape
    source = key.shape[1]
    q = (query@wq.T+bq).reshape(batch, n, 2, 4).transpose(1, 2)
    k = (key@wk.T+bk).reshape(batch, source, 2, 4).transpose(1, 2)
    v = (value@wv.T+bv).reshape(batch, source, 2, 4).transpose(1, 2)
    scores = q@k.transpose(-1, -2)/2
    blocked = padding[:, None, None, :].expand(batch, 2, n, source)
    if causal:
        blocked = blocked | (torch.arange(source)[None, :] > torch.arange(n)[:, None])
    scores = scores.masked_fill(blocked, -torch.inf)
    exponents = (scores-scores.max(-1, keepdim=True).values).exp()
    a = exponents/exponents.sum(-1, keepdim=True)
    result = (a@v).transpose(1, 2).reshape(batch, n, width)
    return result@module.out_proj.weight.T+module.out_proj.bias


def ffn(layer, x):
    return (x@layer.linear1.weight.T+layer.linear1.bias).clamp_min(0)@layer.linear2.weight.T+layer.linear2.bias


def independent(model, source, target, sp, tp):
    enc, dec = model.encoder, model.decoder
    memory = norm(source+attention(enc.self_attn, source, source, source, sp), enc.norm1)
    memory = norm(memory+ffn(enc, memory), enc.norm2)
    x = norm(target+attention(dec.self_attn, target, target, target, tp, causal=True), dec.norm1)
    x = norm(x+attention(dec.multihead_attn, x, memory, memory, sp), dec.norm2)
    x = norm(x+ffn(dec, x), dec.norm3)
    return x@model.output.weight.T


def loss(logits, labels):
    if not bool((labels != 0).any()):
        raise ValueError('Empty loss')
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=0)


def reference_loss(logits, labels):
    values = []
    for b, row in enumerate(labels.tolist()):
        for t, label in enumerate(row):
            if label:
                z = logits[b, t]
                maximum = z.max()
                values.append(maximum+(z-maximum).exp().sum().log()-z[label])
    return torch.stack(values).sum()/len(values)


def error(a, b):
    return (a-b).abs().max().item()


def case(seed, target_length, source_length, padded):
    torch.manual_seed(seed)
    model = Model()
    ids = torch.randint(3, 16, (2, source_length))
    labels = torch.randint(3, 16, (2, target_length))
    sp = torch.zeros_like(ids, dtype=torch.bool)
    sp[1, -1] = True
    ids[sp] = 0
    lengths = (target_length, target_length-int(padded))
    tp = torch.arange(target_length)[None, :] >= torch.tensor(lengths)[:, None]
    for b, length in enumerate(lengths):
        labels[b, length-1] = 2
    labels[tp] = 0
    shifted = torch.cat((torch.ones(2, 1, dtype=torch.long), labels[:, :-1]), 1)
    shifted[tp] = 0
    def valid_shift(candidate):
        return all(int(candidate[b, t]) == (0 if t >= lengths[b] else 1 if t == 0 else int(labels[b, t-1]))
                   for b in range(2) for t in range(target_length))
    assert valid_shift(shifted) and not valid_shift(labels)
    source, target = model.features(ids, shifted)
    output = model(source, target, sp, tp)
    other = independent(model, source, target, sp, tp)
    native_loss, other_loss = loss(output, labels), reference_loss(other, labels)
    params = tuple(model.parameters())
    variables = (*params, source, target)
    ga = torch.autograd.grad(native_loss, variables, retain_graph=True)
    gb = torch.autograd.grad(other_loss, variables)
    errors = dict(logit_error=error(output, other), loss_error=abs(native_loss.item()-other_loss.item()),
                  gradient_error=max(error(a, b) for a, b in zip(ga, gb)),
                  source_padding_gradient=ga[-2][sp].abs().max().item())
    if padded:
        errors['target_padding_gradient'] = ga[-1][tp].abs().max().item()
    assert max(errors.values()) <= 1e-10

    direction = [torch.randn_like(p) for p in params]
    magnitude = sum(v.square().sum() for v in direction).sqrt()
    direction = [v/magnitude for v in direction]
    derivative = sum((g*v).sum() for g, v in zip(ga, direction)).item()
    original = [p.detach().clone() for p in params]
    probes = []
    with torch.no_grad():
        for sign in (1, -1):
            for p, o, d in zip(params, original, direction):
                p.copy_(o+sign*1e-6*d)
            s, t = model.features(ids, shifted)
            probes.append(loss(model(s, t, sp, tp), labels).item())
        for p, o in zip(params, original):
            p.copy_(o)
    fd_error = abs((probes[0]-probes[1])/2e-6-derivative)
    assert fd_error <= 1e-7

    negative = {}
    with torch.no_grad():
        source, target = model.features(ids, shifted)
        original_logits = model(source, target, sp, tp)
        prefix_error = 0.
        for b, length in enumerate(lengths):
            for t in range(length):
                prefix = model(source[b:b+1], target[b:b+1, :t+1], sp[b:b+1], tp[b:b+1, :t+1])
                prefix_error = max(prefix_error, error(prefix[0, -1], original_logits[b, t]))
        errors['prefix_decoding_error'] = prefix_error
        perturbation = torch.zeros_like(source)
        perturbation[sp] = 3*torch.randn_like(source[sp])
        padding_error = padding_loss = 0.
        no_encoder_mask = no_memory_mask = 0.
        for sign in (1, -1):
            changed = source+sign*perturbation
            logits = model(changed, target, sp, tp)
            padding_error = max(padding_error, error(logits[~tp], original_logits[~tp]))
            padding_loss = max(padding_loss, abs(loss(logits, labels).item()-native_loss.item()))
            no_encoder_mask = max(no_encoder_mask, error(model(changed, target, sp, tp, encoder_mask=False)[~tp], original_logits[~tp]))
            no_memory_mask = max(no_memory_mask, error(model(changed, target, sp, tp, memory_mask=False)[~tp], original_logits[~tp]))
        errors.update(raw_source_padding_error=padding_error, source_padding_loss_error=padding_loss)
        negative.update(missing_encoder_mask_effect=no_encoder_mask, missing_memory_mask_effect=no_memory_mask)
        cut = target_length//2
        change = torch.zeros_like(target)
        change[:, cut:] = 3*torch.randn_like(target[:, cut:])
        change[tp] = 0
        future_error = leakage = 0.
        leaky_original = model(source, target, sp, tp, causal=False)
        for sign in (1, -1):
            changed = target+sign*change
            future_error = max(future_error, error(model(source, changed, sp, tp)[:, :cut], original_logits[:, :cut]))
            leakage = max(leakage, error(model(source, changed, sp, tp, causal=False)[:, :cut], leaky_original[:, :cut]))
        errors['future_target_error'] = future_error
        negative['missing_causal_mask_leakage'] = leakage
        if padded:
            changed = original_logits.clone()
            changed[tp] += 10*torch.randn_like(changed[tp])
            errors['padded_logits_loss_error'] = abs(loss(changed, labels).item()-native_loss.item())
        rejected = 0
        bad_source = sp.clone()
        bad_source[1] = True
        try:
            model(source, target, bad_source, tp)
        except ValueError as exc:
            rejected += int(str(exc) == 'Empty source')
        try:
            loss(original_logits, torch.zeros_like(labels))
        except ValueError as exc:
            rejected += int(str(exc) == 'Empty loss')
        assert rejected == 2
    if padded:
        detached = original_logits.detach().requires_grad_()
        masked_gradient = torch.autograd.grad(loss(detached, labels), detached)[0]
        wrong_gradient = torch.autograd.grad(F.cross_entropy(detached.flatten(0, 1), labels.flatten()), detached)[0]
        errors['padded_logits_gradient'] = masked_gradient[tp].abs().max().item()
        negative['unmasked_loss_gradient'] = wrong_gradient[tp].abs().max().item()
    assert max(errors.values()) <= 1e-10, errors
    assert min(negative.values()) > 1e-6, negative
    return dict(seed=seed, target_length=target_length, source_length=source_length, target_padded=padded,
                valid_targets=int((~tp).sum()), parameters=sum(p.numel() for p in params),
                errors=errors, directional_derivative_error=fd_error, intentional_errors=negative,
                unshifted_targets_rejected=True, empty_source_rejected=True, empty_loss_rejected=True)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    path = ROOT/'results/encoder_decoder_contract.json'
    assert not path.exists(), 'Preserve attempts.'
    files = [Path(__file__), ROOT/'docs/protocols/encoder-decoder-contract.md',
             ROOT/'experiments/position_distractors.py', ROOT/'experiments/audit_kv_cache_equivalence.py',
             ROOT/'experiments/recurrent_retrieval.py']
    report = dict(phase='running', torch=torch.__version__, python=platform.python_version(),
                  device='cpu', dtype='float64', provenance={str(p.relative_to(ROOT)):digest(p) for p in files}, cases=[])
    write(path, report)
    for seed, lengths, padded in itertools.product(SEEDS, ((3,5),(5,3),(4,4)), (False,True)):
        report['cases'].append(case(seed, *lengths, padded))
        write(path, report)
    keys = {key for c in report['cases'] for key in c['errors']}
    negative = {key for c in report['cases'] for key in c['intentional_errors']}
    report.update(phase='complete', passed=True,
        maxima={key:max(c['errors'][key] for c in report['cases'] if key in c['errors']) for key in sorted(keys)},
        directional_derivative_max=max(c['directional_derivative_error'] for c in report['cases']),
        negative_minima={key:min(c['intentional_errors'][key] for c in report['cases'] if key in c['intentional_errors']) for key in sorted(negative)})
    write(path, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('provenance','cases')}))


if __name__ == '__main__':
    main()
