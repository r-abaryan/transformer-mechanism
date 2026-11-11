# Residual effects through FFN, normalization and logits

Executed on 2026-09-29. **Seed 47's rescue does not require changing attention or the FFN output.** Correcting the skip-path vector alone fixed all 210 errors on the 1,024-example pathway set. This is a controlled intervention result, not a complete explanation of training or a new architecture.

## Method: split the residual correction

The FFN transforms features; its surrounding skip connection also carries features directly forward. We separated these routes with attention held fixed.

Let u be the original post-attention residual, delta=PE(6)-PE(10), and f(u)=FFN(LN2(u)). The vector entering final normalization is b:

| Condition | Vector b | Meaning |
|---|---|---|
| Original | u+f(u) | Unchanged computation |
| Full residual patch | u+delta+f(u+delta) | Earlier residual-only rescue, including its FFN response |
| Skip only | u+delta+f(u) | Correct direct path; retain original FFN output |
| FFN only | u+f(u+delta) | Retain direct path; substitute donor FFN output |

All five saved checkpoints were tested without training. The data are reused from the preceding fresh-data experiment, so this decomposition is exploratory. The [protocol](../../downstream-protocol.md) was fixed before execution. Patched combinations may be outside the training distribution.

## Accuracy across seeds

These percentages measure final predictions after ordinary final normalization and vocabulary projection, not intermediate diagnostic classifications.

| Seed | Original | Full residual patch | Skip only | FFN only |
|---|---:|---:|---:|---:|
| 11 | 100.00 | 100.00 | 100.00 | 100.00 |
| 29 | 97.27 | 100.00 | 100.00 | 100.00 |
| 47 | 79.49 | 100.00 | 100.00 | 91.89 |
| 73 | 99.71 | 100.00 | 100.00 | 100.00 |
| 101 | 96.78 | 98.63 | 99.22 | 93.65 |

The FFN's response to the positional change helps seed 47, but worsens aggregate accuracy for seed 101. A single explanation should not be assumed to cover every seed.

## Seed 47: margins and paired changes

The margin is the correct-token logit minus the largest competing logit. Positive margins indicate correct predictions. The table averages margins over the 210 originally incorrect examples only.

| Condition | Mean margin on original errors | Corrected | Broken | Overall loss, nats |
|---|---:|---:|---:|---:|
| Original | -1.2023 | 0 | 0 | 0.353092 |
| Full residual patch | 6.7021 | 210 | 0 | 0.002031 |
| Skip only | 5.7275 | 210 | 0 | 0.002759 |
| FFN only | 0.0822 | 127 | 0 | 0.169330 |

The skip-only result establishes that changing the FFN output is unnecessary for this particular rescue. It does not establish that the FFN itself is unnecessary: its original output is retained.

## Final normalization: statistics versus vector content

LayerNorm changes centering and scale before the vocabulary projection:

\[
z=W\left[\gamma\odot\frac{b-\mu(b)}{s(b)}+\beta\right],\qquad
s(b)=\sqrt{\operatorname{Var}(b)+\epsilon}.
\]

For the full residual patch, we separately reused the original mean and standard deviation while retaining learned gamma and beta.

| Seed 47 normalization condition | Accuracy | Loss, nats |
|---|---:|---:|
| Recompute both statistics | 100% | 0.002031 |
| Keep original mean | 100% | 0.001811 |
| Keep original standard deviation | 100% | 0.005537 |
| Keep both original statistics | 100% | 0.004732 |

Updating normalization statistics is not necessary for the full rescue on this set. Normalization still affects confidence and remains in the computation; these interventions do not justify removing it.

## Exact logit-gap decomposition

To make contributions additive, fix competitor c to each example's strongest original competitor. With w=W_target-W_c, the gap decomposes exactly:

\[
z_{target}-z_c=
\frac{w^T(\gamma\odot b)}s
-\frac{\mu\,w^T\gamma}s+w^T\beta.
\]

| Seed 47, original-error group | Feature term | Centering term | Learned bias | Sum |
|---|---:|---:|---:|---:|
| Original | -1.4721 | 0.1691 | 0.1007 | -1.2023 |
| Full residual patch | 7.4552 | -0.4982 | 0.1007 | 7.0577 |

The full-patch sum differs from the earlier margin because its strongest competitor can change. This is an algebraic accounting, not an independent causal attribution: the feature and centering terms share the scale s.

## Verification, scope and next boundary

Reconstructed baseline/full-patch logits and additive decompositions matched their references within 5.330e-15. Checkpoint and dataset hashes matched earlier artifacts. CPU float64; five seeds; no retraining or architecture changes. [Code](../../../experiments/trace_downstream.py), [raw results](../../../results/downstream_trace.json), and [protocol](../../downstream-protocol.md) record all variants, including intermediate projections that are diagnostic rather than calibrated predictions.

The bounded finding is now: a positional change carried through the direct skip path is sufficient to rescue seed 47 on these examples, even with its original attention and FFN outputs. We have not shown that it is the unique explanation, isolated which feature coordinates matter, or confirmed this new decomposition on untouched data. Connections to papers remain those in the [evidence summary](../../assessments/evidence-summary.md); no published benchmark is reproduced.

Before pursuing more mechanisms, confirm the skip-only prediction on a newly fixed test set and additional lengths. That provides a stopping check against overexplaining a single reused dataset.
