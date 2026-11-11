"""RoPE copy variant and separate equations; the historical model is unchanged."""
import math

import torch
from torch.nn import functional as F

from encoder_decoder_contract import Model as SinusoidalModel, independent as sinusoidal_reference
from audit_kv_cache_equivalence import norm


def rotate(x, positions):
    width = x.shape[-1]
    if width % 2:
        raise ValueError('RoPE needs an even head width')
    frequency = 10000. ** (-torch.arange(0, width, 2, dtype=x.dtype, device=x.device) / width)
    angle = positions.to(dtype=x.dtype, device=x.device)[:, None] * frequency
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a*angle.cos()-b*angle.sin(), a*angle.sin()+b*angle.cos()), -1).flatten(-2)


def matrix_rotation(x, positions):
    """Independent explicit 2x2 matrices, without calling rotate."""
    pairs = []
    for i in range(x.shape[-1] // 2):
        angle = positions.to(x.dtype) * (10000. ** (-2*i/x.shape[-1]))
        c, s = angle.cos(), angle.sin()
        matrix = torch.stack((c, -s, s, c), -1).reshape(len(positions), 2, 2)
        pairs.append(torch.einsum('nij,bhnj->bhni', matrix, x[..., 2*i:2*i+2]))
    return torch.cat(pairs, -1)


def attention(module, query, key, value, padding, qp, kp, causal=False, reference=False):
    b, n, width = query.shape
    heads = module.num_heads
    d = width // heads
    weights = module.in_proj_weight.chunk(3)
    biases = module.in_proj_bias.chunk(3)
    q, k, v = [(x@w.T+bias).reshape(b, x.shape[1], heads, d).transpose(1, 2)
               for x, w, bias in zip((query, key, value), weights, biases)]
    rotation = matrix_rotation if reference else rotate
    q, k = rotation(q, qp), rotation(k, kp)
    blocked = padding[:, None, None, :].expand(b, heads, n, key.shape[1])
    if causal:
        blocked = blocked | (torch.arange(key.shape[1])[None, :] > torch.arange(n)[:, None])
    if reference:
        scores = (q@k.transpose(-1, -2))/math.sqrt(d)
        scores = scores.masked_fill(blocked, -torch.inf)
        weights = (scores-scores.max(-1, keepdim=True).values).exp()
        result = (weights/weights.sum(-1, keepdim=True))@v
    else:
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=~blocked, dropout_p=0.)
    result = result.transpose(1, 2).reshape(b, n, width)
    return result@module.out_proj.weight.T+module.out_proj.bias


class RopeModel(SinusoidalModel):
    def features(self, source, target):
        return self.source(source)*math.sqrt(8), self.target(target)*math.sqrt(8)

    def forward(self, source, target, sp, tp, encoder_mask=True, memory_mask=True, causal=True,
                position_offset=0):
        if bool(sp.all(-1).any()):
            raise ValueError('Empty source')
        ep = torch.arange(source.shape[1])+position_offset
        dp = torch.arange(target.shape[1])+position_offset
        enc, dec = self.encoder, self.decoder
        enc_pad = sp if encoder_mask else torch.zeros_like(sp)
        mem_pad = sp if memory_mask else torch.zeros_like(sp)
        memory = enc.norm1(source+attention(enc.self_attn, source, source, source, enc_pad, ep, ep))
        memory = enc.norm2(memory+enc.linear2(F.relu(enc.linear1(memory))))
        x = dec.norm1(target+attention(dec.self_attn, target, target, target, tp, dp, dp, causal))
        x = dec.norm2(x+attention(dec.multihead_attn, x, memory, memory, mem_pad, dp, ep))
        return self.output(dec.norm3(x+dec.linear2(F.relu(dec.linear1(x)))))


def make_model(arm):
    if arm == 'sinusoidal':
        return SinusoidalModel()
    if arm == 'rope':
        return RopeModel()
    raise ValueError(arm)


def reference(model, source_ids, target_ids, arm):
    sp, tp = torch.zeros_like(source_ids, dtype=torch.bool), torch.zeros_like(target_ids, dtype=torch.bool)
    if arm == 'sinusoidal':
        s, t = model.features(source_ids, target_ids)
        return sinusoidal_reference(model, s, t, sp, tp)
    # Independent features, normalization, FFN and matrix rotations.
    s, t = model.source(source_ids)*math.sqrt(8), model.target(target_ids)*math.sqrt(8)
    return rope_reference_features(model, s, t, sp, tp)


def rope_reference_features(model, source, target, sp, tp, position_offset=0):
    ep, dp = torch.arange(source.shape[1])+position_offset, torch.arange(target.shape[1])+position_offset
    enc, dec = model.encoder, model.decoder
    def ff(layer, x):
        return (x@layer.linear1.weight.T+layer.linear1.bias).clamp_min(0)@layer.linear2.weight.T+layer.linear2.bias
    def att(module, q, k, v, pad, qp, kp, causal=False):
        return attention(module, q, k, v, pad, qp, kp, causal, reference=True)
    memory = norm(source+att(enc.self_attn, source, source, source, sp, ep, ep), enc.norm1)
    memory = norm(memory+ff(enc, memory), enc.norm2)
    x = norm(target+att(dec.self_attn, target, target, target, tp, dp, dp, True), dec.norm1)
    x = norm(x+att(dec.multihead_attn, x, memory, memory, sp, dp, ep), dec.norm2)
    x = norm(x+ff(dec, x), dec.norm3)
    return x@model.output.weight.T
