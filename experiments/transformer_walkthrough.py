"""Fixed-weight decoder example; reuses the verified spike arithmetic."""

import hashlib
import json
import math
from pathlib import Path
import sys

from attention_spikes import attention, error, matmul, require, softmax


def add(a, b):
    return [[x + y for x, y in zip(ra, rb, strict=True)]
            for ra, rb in zip(a, b, strict=True)]


def norm(rows):
    result = []
    for row in rows:
        mean = sum(row) / len(row)
        variance = sum((x - mean) ** 2 for x in row) / len(row)
        result.append([(x - mean) / math.sqrt(variance + 1e-5) for x in row])
    return result


def fixture():
    # Synthetic teaching weights, not learned parameters.
    return dict(
        embeddings=[[1, 0, -1, 0], [0, 1, 0, -1], [1, 1, 0, 0]],
        projections=[[[1, 0], [0, 1], [0, 0], [0, 0]],
                     [[0, 0], [0, 0], [1, 0], [0, 1]]],
        output=[[1, 0, 0, 0], [0, 1, 0, 0],
                [0, 0, 1, 0], [0, 0, 0, 1]],
        w1=[[1, 0, -1], [0, 1, 0], [1, 0, 1], [0, -1, 0]],
        w2=[[0.2, 0, -0.2, 0], [0, 0.2, 0, -0.2], [0.1, -0.1, 0.1, -0.1]],
        unembed=[[1, 0, -1], [0, 1, -1], [-1, 0, 1], [0, -1, 1]])


def forward(ids, weights, causal=True):
    embeddings = [weights['embeddings'][i] for i in ids]
    positions = [[math.sin(i), math.cos(i), math.sin(i / 100), math.cos(i / 100)]
                 for i in range(len(ids))]
    x = add(embeddings, positions)
    normalized = norm(x)
    heads = []
    for projection in weights['projections']:
        q = matmul(normalized, projection)
        # Q, K and V intentionally share this projection in this fixture.
        y, a = attention(q, q, q, causal=causal)
        scores = [[sum(u * v for u, v in zip(qi, qj, strict=True)) / math.sqrt(2)
                   for qj in q] for qi in q]
        heads.append(dict(q=q, k=q, v=q, scores_before_mask=scores, attention=a, output=y))
    concat = [heads[0]['output'][i] + heads[1]['output'][i] for i in range(len(ids))]
    projected = matmul(concat, weights['output'])
    residual = add(x, projected)
    normalized_ffn = norm(residual)
    preactivation = matmul(normalized_ffn, weights['w1'])
    hidden = [[max(0, x) for x in row] for row in preactivation]
    ffn = matmul(hidden, weights['w2'])
    block = add(residual, ffn)
    final = norm(block)
    logits = matmul(final, weights['unembed'])
    return dict(ids=ids, embeddings=embeddings, positions=positions, x=x,
                normalized_attention_input=normalized, heads=heads, concatenated=concat,
                projected_attention=projected, attention_residual=residual,
                normalized_ffn_input=normalized_ffn, ffn_preactivation=preactivation,
                ffn_hidden=hidden, ffn_output=ffn, block_output=block,
                final_normalized=final, logits=logits,
                probabilities=[softmax(row) for row in logits])


def cross_entropy(logits, targets):
    losses = []
    for row, target in zip(logits, targets, strict=True):
        m = max(row)
        losses.append(m + math.log(sum(math.exp(x - m) for x in row)) - row[target])
    return losses


def main():
    weights = fixture()
    sequence = [0, 1, 2, 0]
    inputs, targets = sequence[:-1], sequence[1:]
    trace = forward(inputs, weights)
    losses = cross_entropy(trace['logits'], targets)
    trace.update(targets=targets, token_losses=losses, mean_loss=sum(losses) / len(losses))
    changed = forward([0, 1, 0], weights)
    causal_error = error(trace['logits'][:2], changed['logits'][:2])
    unmasked_change = error(forward(inputs, weights, False)['logits'][:2],
                            forward([0, 1, 0], weights, False)['logits'][:2])
    prefix_error = max(error(forward(inputs[:i], weights)['logits'], trace['logits'][:i])
                       for i in (1, 2, 3))
    uniform_loss = cross_entropy([[0, 0, 0]], [2])[0]
    require(causal_error < 1e-12, 'Whole-block future leakage')
    require(prefix_error < 1e-12, 'Prefix evaluation disagrees')
    require(unmasked_change > 1e-3, 'Unmasked control did not detect future influence')
    require(abs(uniform_loss - math.log(3)) < 1e-12, 'Uniform loss oracle failed')
    require(error(norm([[1, 1, 1, 1]]), [[0, 0, 0, 0]]) == 0, 'Constant normalization failed')
    require(all(abs(sum(row) - 1) < 1e-12 for row in trace['probabilities']), 'Invalid probabilities')
    sources = [Path(__file__), Path(__file__).with_name('attention_spikes.py')]
    report = dict(python=sys.version, weights=weights, trace=trace,
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  checks=dict(causal_error=causal_error, prefix_error=prefix_error,
                              unmasked_change=unmasked_change, uniform_loss=uniform_loss,
                              constant_normalization='pass', probability_sums='pass'))
    destination = Path(__file__).resolve().parents[1] / 'results/transformer_walkthrough.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(loss=trace['mean_loss'], probabilities=trace['probabilities'],
                         checks=report['checks']), indent=2))


if __name__ == '__main__':
    main()
