# Broader copy training helps within range, not beyond it

Executed 2026-10-03. All six fresh models fit their full training pools exactly. Training on lengths 3 through 6 instead of 3/4 raises six-symbol exact copying from **0% to 92.71%**. Neither arm copies any eight- or ten-symbol test sequence exactly. Broader exposure helps, but reliable extrapolation is not demonstrated.

[Educational guide and diagram](copy-length-exposure-guide.md) | [Frozen protocol](../../protocols/copy-length-exposure.md)

## Methods and math

Three matched seed pairs, 3061/3067/3079. Same initial parameter values within each pair; unchanged 1,888-parameter encoder-decoder: width8, two heads, one encoder/decoder layer, post-norm, ReLU FFN16, sinusoidal positions, dropout0. Adam .001, no schedule/clipping, 6,000 updates each. CPU float64, one thread, native Windows; GPU unused.

Correct target is the source followed by EOS. The decoder receives BOS and the correct earlier symbols during training. Each batch has B(n+1)=840 supervised targets, including EOS:

\[
\mathcal L=-\frac{1}{840}\sum_{b=1}^{B}\sum_{j=0}^{n}
\log p_\theta(y_{bj}\mid x_b,y_{b,<j}).
\]

| Per run | Short: cycle 3/4 | Mixed: cycle 3/4/5/6 |
|---|---:|---:|
| Supervised targets | 5,040,000 | 5,040,000 |
| Sampled sequences | 1,134,000 | 957,000 |
| Unique training sequences | 320 | 832 |
| Batch by length | 210 / 168 | 210 / 168 / 140 / 120 |
| Visible attention pairs, excluding head multiplier | 45,234,000 | 57,657,000 |

Both arms share uniform sampling ranks; indices match at same-length updates. Token budgets match, but compute, sequence count, unique input support and EOS frequency differ. The result concerns this exposure package, not positional diversity alone. Complete-step profiling includes allocated Adam states. Six training loops total 498.81 seconds, including scheduled evaluations/saves; this is not a speed benchmark.

New exact-input-disjoint pools exclude the previously inspected copy sequences. Vocabulary/subsequences are shared. Distinct/repeated groups are balanced through length8. Length10 has only repeated symbols because the vocabulary has eight content symbols. Length3 held-out groups contain only eight examples each; other test groups have 64 examples, or 128 for length10.

All six final checkpoints and a pretest snapshot precede model test scoring. Ordinary greedy generation uses learned attention and stops at first EOS or the source-length+3 cap. No attention intervention or EOS blocking. Full training/validation evaluations occur at 1,000/3,000/6,000 updates; no checkpoint selection or tuning.

## Mean test results

Equal-weight means across three seeds, percentages. **Content exact** requires all source symbols; **full exact** also requires immediate EOS. Teacher accuracy uses correct prefixes.

| Symbols | Short full | Mixed full | Short content exact | Mixed content exact | Short teacher content | Mixed teacher content |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 87.50 | 100 | 91.67 | 100 | 96.53 | 100 |
| 4 | 86.20 | 99.48 | 88.28 | 99.74 | 96.88 | 99.93 |
| 5 | 1.04 | 98.18 | 1.04 | 98.96 | 72.40 | 99.79 |
| 6 | 0 | **92.71** | 0 | **95.57** | 57.99 | 99.26 |
| 8 | 0 | 0 | 0 | 0.26 | 32.23 | 66.89 |
| 10 | 0 | 0 | 0 | 0 | 26.35 | 44.87 |

Length6 is inside mixed's training range. Its gain establishes improved copying of new inputs at a seen length, not mixed-arm extrapolation.

## Seed and group reliability

Full exact accuracy (%). Every short test6 group is 0%; every eight/ten-symbol full test group in both arms is 0%.

| Seed | Mixed test6 distinct | Mixed test6 repeated | Mixed test5 repeated | Worst short seen-length test group | Worst mixed seen-length test group | Worst mixed validation group |
|---|---:|---:|---:|---:|---:|---:|
| 3061 | 96.88 | 87.50 | 100 | 87.50 | 87.50 | 93.75 |
| 3067 | 93.75 | 93.75 | 100 | 62.50 | 93.75 | 87.50 |
| 3079 | 100 | 84.38 | 90.63 | 78.13 | 84.38 | 68.75 |

Every training group is 100%. Both >=99% training-fit gates pass. Paired test6 gains are 92.19/93.75/92.19 points, and no short test group regresses. The frozen broader-support benefit criterion passes. Both >=95% every-seed/group seen-length reliability gates fail; improvement is not uniform reliability. Both extrapolation criteria fail, with raw long-length full accuracy also zero.

Mixed early-EOS rates rise from 2.60% at six symbols to 98.70% at eight and 99.22% at ten. Teacher content accuracy also drops. This documents stopping and content difficulties; it does not identify their cause. The earlier [EOS/routing interventions](copy-aligned-routing-results.md) used different trained checkpoints and privileged decoding, so their scores are not paired controls here.

## Verification, interpretation and stopping

Independent equations and a separate decoding controller match all 78 seed/arm/pool evaluations: 56,064 teacher targets, 8,640 generated sequences and 47,777 generated tokens. Zero teacher argmax or generated-sequence differences; maximum logit error 4.47e-13 and metric error 3.00e-15. Initial gradients, labels/shifting, historical exclusion, optimizer counters, sampling streams and exposure counts pass. Deliberate reversal-label, after-EOS controller and criterion errors are detected. Relevant archived hashes remain unchanged. These counts are three independent seed pairs, not thousands of independent training experiments.

[Newman et al. (2020), sections 1 and 3](https://arxiv.org/html/2010.07174v1) motivate separating content from stopping. They compare EOS/no-EOS training with oracle-length evaluation. We retain EOS in both arms and use ordinary generation; our results neither reproduce their tasks nor identify their hidden-state mechanisms.

The unchanged architecture can learn substantially better copying within broader training support under this recipe. Matching tokens does not isolate whether longer positions, more unique inputs or changed stopping feedback drive the gain. Reliable unseen-length copying remains unresolved. Stop after the declared six runs and audit; no automatic tuning, added training or architecture change.

Evidence: [producer](../../../experiments/copy_length_exposure.py), [independent audit source](../../../experiments/audit_copy_length_exposure.py), [report](../../../results/copy_length_exposure/report.json), [audit](../../../results/copy_length_exposure/audit.json), [all per-seed/group metrics](../../../results/copy_length_exposure/metrics.csv), [data](../../../results/copy_length_exposure/data.json), [profile](../../../results/copy_length_exposure/profile.json). Original copy evidence remains unchanged.
