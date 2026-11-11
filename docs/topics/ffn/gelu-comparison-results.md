# FFN activation results: ReLU versus GELU

**Retain ReLU under the fixed decision rule.** Mean six-pair accuracy changes from **95.85% to 96.08%** (+0.225 percentage points). GELU improves four seeds but worsens one. The required >=2-point gain and no-worse-seed criteria fail. Ten training runs and all 30 independent checkpoint evaluations completed; comparison closed.

## What was tested

The [learning guide](../../ffn-activation-guide.md) explains the equations, derivatives and FFN diagram. ReLU uses `max(0,u)`; exact GELU uses `u*Phi(u)`. This changes both the features sent to the next projection and the gradients used to learn them. Smoothness alone does not predict which model will perform better.

Both arms use RoPE. Change only the activation in both FFNs. New models share initial learned tensors, data, minibatches and optimizer settings. Two pre-norm blocks, width16, two width8 heads, FFN32, dropout0,5,088 parameters. Adam.001, batch64,6,000 updates, alternating three/four pairs. Final checkpoints only; tests opened after all ten attempts. No failed or replacement seed. Eight runs completed before interruption; execution resumed from saved state under unchanged hashes.

The task asks for the newer value of the only duplicated key. Six-pair inputs are test-only. As before, a solver could identify that duplicated key without using the query token; success is not proof of unrestricted query-conditioned recall.

## Fresh data and setup correction

The original request for1,100 fresh three-pair families was impossible. There are only `12*11*C(6,2)*4/6 = 1,320` under our order/cyclic-value equivalence; the prior study used1,100. Generation was stopped before any model training or input artifact. [Original setup and hashes](../../../results/gelu_setup_failure/manifest.json) are preserved.

The [corrected protocol](../../gelu-comparison-protocol.md) was frozen before valid training. It allocates the 220 remaining three-pair families as160 train/20 validation/40 test. Four-pair counts remain800/100/200; six-pair test200. Each family provides36 order/value variants. Audit excludes all 2,400 families from the previous study and checks zero overlap across new splits. Smaller fragments and other renamings may be familiar.

Both arms share this reduced three-pair coverage. Only40 families support short-length test conclusions; six-pair tests contain200 families and7,200 answers per model. Seeds and data differ from earlier studies, so historical percentages are not a matched comparison. No unused three-pair families remain under this exact exclusion definition.

## Paired results

Accuracy is the fraction of correct individual answers. The final column is GELU minus ReLU in percentage points.

| Seed | ReLU 3 / 4 pairs, % | GELU 3 / 4 pairs, % | ReLU 6 pairs | GELU 6 pairs | Gain, pp |
|---|---:|---:|---:|---:|---:|
| 1103 | 100.00 / 99.71 | 100.00 / 99.96 | 95.75% | 96.58% | +0.833 |
| 1123 | 99.93 / 99.74 | 100.00 / 100.00 | 96.54% | 96.75% | +0.208 |
| 1151 | 100.00 / 100.00 | 100.00 / 99.93 | 97.67% | 98.64% | +0.972 |
| 1171 | 100.00 / 99.67 | 100.00 / 99.75 | 92.03% | 91.11% | -0.917 |
| 1193 | 99.93 / 99.76 | 100.00 / 99.99 | 97.28% | 97.31% | +0.028 |

Both activations exceed90% in every six-pair run here. This does not show that GELU repaired the previous study: the fresh ReLU baseline also clears that threshold.

| Six-pair mean metric | ReLU | GELU | Meaning |
|---|---:|---:|---|
| All-variant family accuracy | 62.40% | 63.80% | All36 examples correct. |
| Reversal accuracy | 91.91% | 92.48% | Both a sequence and its reversal correct. |
| Obsolete-value rate | 0.26% | 0.62% | Predicting the older matching value; lower is better. |
| Cross-entropy | 0.21319 | 0.20519 | Penalty for true-answer probability; lower is better. |

A small average accuracy gain does not improve every metric: GELU selects obsolete values more often on average. This is observed behavior, not a causal explanation.

Detailed diagnostics:

| Seed / activation | CE: 3 / 4 / 6 pairs | Six-pair family / reversal / obsolete, % | Accuracy by latest pair position2..6, % |
|---|---|---|---|
| 1103 / relu | 0.00038 / 0.00716 / 0.24545 | 57.50 / 91.56 / 0.22 | 60.33 / 90.20 / 97.73 / 99.76 / 99.87 |
| 1103 / gelu | 0.00033 / 0.00096 / 0.19347 | 66.00 / 93.17 / 0.08 | 61.74 / 92.83 / 99.15 / 100.00 / 100.00 |
| 1123 / relu | 0.00100 / 0.00873 / 0.14327 | 58.50 / 93.22 / 0.67 | 89.91 / 93.94 / 94.82 / 96.69 / 99.83 |
| 1123 / gelu | 0.00014 / 0.00019 / 0.14356 | 64.00 / 93.81 / 0.99 | 77.46 / 95.86 / 94.33 / 99.18 / 100.00 |
| 1151 / relu | 0.00011 / 0.00018 / 0.13970 | 77.50 / 95.33 / 0.00 | 77.23 / 93.64 / 99.43 / 100.00 / 100.00 |
| 1151 / gelu | 0.00069 / 0.00476 / 0.05849 | 81.50 / 97.28 / 0.00 | 87.79 / 96.26 / 99.36 / 100.00 / 100.00 |
| 1171 / relu | 0.00026 / 0.01462 / 0.40957 | 47.50 / 84.83 / 0.33 | 70.89 / 87.58 / 86.24 / 94.91 / 98.82 |
| 1171 / gelu | 0.00023 / 0.00912 / 0.48506 | 39.50 / 83.53 / 2.01 | 75.12 / 89.90 / 90.28 / 91.98 / 94.33 |
| 1193 / relu | 0.00124 / 0.01201 / 0.12795 | 71.00 / 94.58 / 0.06 | 82.86 / 92.22 / 97.87 / 99.28 / 99.96 |
| 1193 / gelu | 0.00017 / 0.00171 / 0.14535 | 68.00 / 94.61 / 0.01 | 75.59 / 92.73 / 99.22 / 99.66 / 100.00 |

## Verification and decision

| Check | Result |
|---|---|
| Activation math | GELU formula/derivative errors <=2.22e-16; ReLU derivative check passes. |
| Baseline equivalence | ReLU logits and gradients match the prior RoPE implementation exactly on the synthetic fixture. |
| Change actually applied | Switching activation changes fixture logits by up to 0.1377. |
| Model gradient | One GELU parameter's finite-difference error is 7.36e-10; this is a local check, not an exhaustive gradient proof. |
| Data and training | Labels, reversal controls, exclusions, paired initial weights, regenerated batch hashes and6,000 optimizer steps pass. |
| Independent forward | All 30 saved-logit evaluations agree within 7.37e-14, using explicit rotation matrices, SDPA and the erf formula independently of the training GELU call. |
| Frozen criteria | All-seed six-pair90% and short-length preservation pass. Mean gain >=2pp fails; no-worse-seed fails at1171 (-0.917pp). |

CPU float64, one thread, PyTorch 2.14.0+cpu. Each run uses 3,456,000 tokens and 31,488,000 batch-weighted squared sequence lengths before head/layer factors. GELU changes arithmetic cost. Loop totals: ReLU 226.08s; GELU 237.90s. These include periodic validation and exclude checkpoint writes/final testing. Interruption and sequential execution make them unsuitable for a speed claim.

## What this means

The fixed activation replacement does not earn adoption here. Retain ReLU with RoPE for this recipe. This is an engineering decision, not statistical evidence that the functions are equivalent or that ReLU is universally better.

Five seeds share one split. We did not tune the best optimizer for each activation, test other tasks or isolate whether forward features or gradients caused differences. No further seed, budget extension or activation search follows this result.

[Hendrycks and Gimpel, GELU v5, sections2-3](https://arxiv.org/html/1606.08415v5) defines the function and reports benefits on other tasks/settings. Our math checks agree with its definition. Our fixed-budget recall test is a separate experiment; failing our adoption rule does not refute those reported benchmarks.

## Evidence and reproduction

[Protocol](../../gelu-comparison-protocol.md), [training and curves](../../../results/gelu_comparison.json), [verification](../../../results/gelu_comparison_verification.json), [audit](../../../results/gelu_comparison_audit.json), [inputs](../../../results/gelu_comparison_data.json). Final weights and logits remain in `results/gelu_comparison_checkpoints/` and `results/gelu_comparison_scores.pt`.

```powershell
.venv/Scripts/python.exe experiments/verify_gelu_comparison.py
.venv/Scripts/python.exe experiments/gelu_comparison.py
.venv/Scripts/python.exe experiments/audit_gelu_comparison.py
```

The runner resumes compatible saved state and leaves a completed report unchanged. Preserve both the corrected source hashes and the archived setup failure.
