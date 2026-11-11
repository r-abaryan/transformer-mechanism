"""Compare explicit decoder math with PyTorch's library block and finite differences."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from transformer_walkthrough import fixture, forward


class TinyDecoder(nn.Module):
    def __init__(self, vocabulary=7, width=8, heads=2, hidden=12):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width)
        self.block = nn.TransformerEncoderLayer(width, heads, hidden, dropout=0,
                                               batch_first=True, norm_first=True)
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocabulary, bias=False)

    def forward(self, ids, padding=None):
        x = self.embedding(ids)
        n, d = x.shape[1:]
        position = torch.arange(n, dtype=x.dtype, device=x.device)[:, None]
        frequency = 10000 ** (-torch.arange(0, d, 2, dtype=x.dtype, device=x.device) / d)
        pe = torch.stack(((position * frequency).sin(), (position * frequency).cos()), -1).flatten(1)
        x = x + pe
        mask = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1)
        return self.output(self.final_norm(self.block(x, src_mask=mask, src_key_padding_mask=padding)))


def explicit_block(x, block, padding):
    def ln(a, layer):
        mean = a.mean(-1, keepdim=True)
        variance = ((a - mean) ** 2).mean(-1, keepdim=True)
        return (a - mean) / torch.sqrt(variance + layer.eps) * layer.weight + layer.bias
    z = ln(x, block.norm1)
    q, k, v = F.linear(z, block.self_attn.in_proj_weight, block.self_attn.in_proj_bias).chunk(3, -1)
    b, n, d = x.shape
    h = block.self_attn.num_heads
    q, k, v = [a.reshape(b, n, h, d // h).transpose(1, 2) for a in (q, k, v)]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d // h)
    forbidden = torch.ones(n, n, dtype=torch.bool).triu(1)[None, None] | padding[:, None, None, :]
    a = scores.masked_fill(forbidden, -torch.inf).softmax(-1)
    y = (a @ v).transpose(1, 2).reshape(b, n, d)
    u = x + block.self_attn.out_proj(y)
    return u + block.linear2(F.relu(block.linear1(ln(u, block.norm2))))


def maximum(a, b):
    return float((a - b).abs().max().detach())


def check_walkthrough():
    w = fixture()
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    original = forward([0, 1, 2], w)
    x = t(w['embeddings']) + t(original['positions'])
    z = F.layer_norm(x, (4,))
    ys = []
    for p in w['projections']:
        q = (z @ t(p))[None, None]
        ys.append(F.scaled_dot_product_attention(q, q, q, is_causal=True)[0, 0])
    u = x + torch.cat(ys, -1) @ t(w['output'])
    block = u + F.relu(F.layer_norm(u, (4,)) @ t(w['w1'])) @ t(w['w2'])
    logits = F.layer_norm(block, (4,)) @ t(w['unembed'])
    difference = maximum(logits, t(original['logits']))
    assert difference < 1e-10, difference
    return difference


def main():
    torch.set_num_threads(1)
    records = []
    for seed in (11, 29, 47):
        torch.manual_seed(seed)
        layer = nn.TransformerEncoderLayer(4, 2, 7, dropout=0, batch_first=True, norm_first=True).double()
        reference = copy.deepcopy(layer)
        x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        padding = torch.tensor([[False, False, False], [False, False, True]])
        mask = torch.ones(3, 3, dtype=torch.bool).triu(1)
        actual = explicit_block(x, layer, padding)
        expected = reference(xr, src_mask=mask, src_key_padding_mask=padding)
        output_error = maximum(actual, expected)
        probe = torch.randn_like(actual)
        (actual * probe).sum().backward()
        (expected * probe).sum().backward()
        gradient_error = max([maximum(x.grad, xr.grad)] +
                             [maximum(p.grad, r.grad) for p, r in zip(layer.parameters(), reference.parameters(), strict=True)])
        assert output_error < 1e-10 and gradient_error < 1e-9
        # Numerical check of library block input derivatives, independent of autograd.
        gradcheck = torch.autograd.gradcheck(
            lambda a: reference(a, src_mask=mask, src_key_padding_mask=padding),
            (xr.detach().requires_grad_(),), eps=1e-6, atol=1e-5, rtol=1e-3)
        # Check every parameter coordinate by central finite differences.
        numerical_error = 0.0
        with torch.no_grad():
            for p in reference.parameters():
                for index in range(p.numel()):
                    flat = p.view(-1)
                    old = flat[index].item()
                    flat[index] = old + 1e-6
                    plus = (reference(xr, src_mask=mask, src_key_padding_mask=padding) * probe).sum().item()
                    flat[index] = old - 1e-6
                    minus = (reference(xr, src_mask=mask, src_key_padding_mask=padding) * probe).sum().item()
                    flat[index] = old
                    numerical_error = max(numerical_error, abs((plus-minus)/2e-6 - p.grad.view(-1)[index].item()))
        assert numerical_error < 1e-5, numerical_error
        records.append(dict(seed=seed, output_error=output_error, gradient_error=gradient_error,
                            input_gradcheck=gradcheck, parameter_finite_difference_error=numerical_error))
    torch.manual_seed(73)
    model = TinyDecoder().double()
    ids = torch.tensor([[0, 1, 2, 3], [2, 3, 4, 5]])
    changed = ids.clone()
    changed[:, -1] = 6
    causal_error = maximum(model(ids)[:, :-1], model(changed)[:, :-1])
    padding = torch.tensor([[False, False, True, False], [False, False, True, False]])
    pad_changed = ids.clone()
    pad_changed[:, 2] = 6
    padding_error = maximum(model(ids, padding)[:, [0, 1, 3]], model(pad_changed, padding)[:, [0, 1, 3]])
    control = maximum(model(ids)[:, 3], model(pad_changed)[:, 3])
    assert causal_error < 1e-10 and padding_error < 1e-10 and control > 1e-4
    inputs, targets = ids[:, :-1], ids[:, 1:]
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss = F.cross_entropy(model(inputs).reshape(-1, 7), targets.reshape(-1))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    after = F.cross_entropy(model(inputs).reshape(-1, 7), targets.reshape(-1)).item()
    assert after < loss.item()
    report = dict(torch=torch.__version__, python=sys.version, device='cpu', dtype='float64',
                  walkthrough_error=check_walkthrough(), comparisons=records,
                  causal_error=causal_error, padding_error=padding_error, padding_control_change=control,
                  one_step_loss_before=loss.item(), one_step_loss_after=after,
                  source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ('verify_decoder.py', 'transformer_walkthrough.py', 'attention_spikes.py')})
    path = Path(__file__).resolve().parents[1] / 'results/decoder_verification.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
