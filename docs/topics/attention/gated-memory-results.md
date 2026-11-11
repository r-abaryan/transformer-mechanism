# Gated-memory audit results

Executed2026-10-03 and closed. No training. [Guide and equations](gated-memory-guide.md) | [Frozen protocol](../../protocols/gated-memory.md).

**Scalar forgetting changes retention correctly, but does not guarantee selective replacement or unique history storage.** These are explicit arithmetic examples, not an explanation of the earlier trained retrieval failures.

## Numerical checks

Three seeds, lengths3/7/17, feature/value widths3/2 and5/4:18 cases. Batch2, heads2, CPUfloat64, one thread. Supplied random Q/K/V and gate logits; no learned projections, masking or positional encoding.

Compare serial recurrence, independently expanded decayed pair weights and two chunks carrying state. Check all Q/K/V/gate-logit gradients, plus a joint directional finite difference.

| Check | Largest error |
|---|---:|
| Outputs | 1.12e-15 |
| All input gradients | 2.61e-17 |
| Final chunk state | 0 |
| Finite difference | 3.71e-11 |
| Future perturbation | 0 |
| Gate1 versus ungated attention | 6.67e-16 |
| Gate0 versus current value | 8.89e-16 |

All pass frozen1e-10 arithmetic and1e-7 finite-difference tolerances. First gate-logit gradients are exactly zero because the initial state is zero. This is expected, not failed gradient flow.

## Intentional error controls

| Incorrect operation | Detected difference |
|---|---:|
| Omit decay | 1 |
| Decay S but not z | 1.5 |
| Decay each write by its own gate | 1 |
| Detach state at chunk boundary | Old-value gradient changes by1/3; outputs unchanged |
| Allow future writes | 3 |

## What the examples establish

| Example | Observed result | Limit |
|---|---|---|
| Values2 then8, same scalar features | Gate1/.5/0 gives5/6/8 | Forgetting improves this overwrite, but partial retention still mixes values |
| Ideal orthogonal A/B features, A:2 B:5 A:8 | Gates1/.5/.5 give A6.8,B5; final reset gives A8 and undefined B read | A scalar reset erases unrelated information too; zero denominator is rejected |
| Scalar keys0/1, gates1/.5, values(0,1) versus(4,0) | Both end at S=2,z=2.5 and read0.8 | Gating does not eliminate state collisions |

Equal S/z means every subsequent read with the same query agrees. The collision is a witness for this construction, not an impossibility theorem for every gated model. The orthogonal fixture is outside finite ELU+1 feature outputs and is labeled separately.

## Source comparison and closure

[Yang et al. (2024)](https://arxiv.org/html/2312.06635v6), sections2.1/4.1, provides the gated-memory context. Its GLA gate is feature-wise; our scalar gate and explicit z normalization are a restricted local construction. We did not reproduce its kernels, learned gates, language results or speed claims. This is a focused source review, not a latest-literature survey.

The gate adds no history-sized forward state to S/z. That says nothing about total autograd memory or measured runtime. Float32/low precision, long-product stability and trained behavior remain untested here.

Stop here. No further training, gate tuning or architecture adoption is selected.

Evidence: [source](../../../experiments/gated_memory.py), [raw cases and fixtures](../../../results/gated_memory.json).
