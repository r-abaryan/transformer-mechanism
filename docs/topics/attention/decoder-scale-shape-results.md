# Decoder scale carries the held-out loss penalty

The fixed scale-dominance rule **passes all 12 primary contrasts**: three seeds, two routing anchors, validation/test at load 32. Both conditional scale effects are positive and larger than either non-scalar effect's absolute magnitude. Scale-removed L-BFGS decoding retains every original class decision and complete-map score while substantially lowering held-out CE.

This identifies a decoder contribution to the loss gap under this fixed decomposition. It does not repair the wrong answers or explain their historical learning cause.

## Method, not a temperature search

Center the effective decoder across classes, including bias. Project the final L-BFGS map onto the final negative-gradient map using weights only. The resulting scalar s defines four cases: base, scaled base, scale-removed L map, and full L map. **L** means L-BFGS; **N** means negative gradient. The [guide](decoder-scale-shape-guide.md) derives the equations; the [protocol](../../protocols/decoder-scale-shape.md) was frozen before scoring.

| Seed | Fixed scalar s | Non-scalar residual relative to L map |
|---|---:|---:|
| 3473 | 4.3432 | 4.69% |
| 3479 | 3.0683 | 1.76% |
| 3481 | 2.4895 | 3.33% |

This differs from the earlier scalar-versus-warm-start statistic. No labels or held-out scores select s. The normalized map is not an exact equal-norm or pure-rotation intervention. Coordinates and the chosen Frobenius metric matter.

## Every primary result

CE below is per answer. Routing stays fixed within each row. These are test-32 results:

| Seed | Routing | Base CE | Scale CE | Shape CE | Full CE |
|---|---|---:|---:|---:|---:|
| 3473 | L | 0.9978 | 4.2092 | 0.9999 | 4.2212 |
| 3473 | N | 1.0055 | 4.2542 | 1.0060 | 4.2614 |
| 3479 | L | 0.8849 | 2.6281 | 0.8839 | 2.6227 |
| 3479 | N | 0.8838 | 2.6244 | 0.8827 | 2.6214 |
| 3481 | L | 0.9163 | 2.2194 | 0.9208 | 2.2299 |
| 3481 | N | 0.9787 | 2.3727 | 0.9850 | 2.3897 |

Validation-32 satisfies the same declared rule:

| Seed | Routing | Base CE | Scale CE | Shape CE | Full CE |
|---|---|---:|---:|---:|---:|
| 3473 | L | 0.9739 | 4.0764 | 0.9735 | 4.0680 |
| 3473 | N | 1.0518 | 4.4489 | 1.0568 | 4.4651 |
| 3479 | L | 0.9847 | 2.9339 | 0.9851 | 2.9364 |
| 3479 | N | 1.0015 | 2.9887 | 1.0029 | 2.9918 |
| 3481 | L | 0.9929 | 2.4006 | 0.9893 | 2.3911 |
| 3481 | N | 0.9693 | 2.3513 | 0.9685 | 2.3486 |

Across both paths and all 12 primary contrasts, the conditional scale effect is **98.79% to 100.69%** of the total full-minus-base penalty. These are order-dependent ratios, not a unique causal share. Values above 100% occur when the remaining change offsets some scale damage. Full conditional effects and interactions remain in the [raw report](../../../results/decoder_scale_shape/report.json).

For L routing, removing scale changes test CE from 4.2212/2.6227/2.2299 to 0.9999/0.8839/0.9208. Accuracy remains 87.4023%/88.6637%/87.5895%, with 0/12/6 complete maps out of 384. Better loss leaves the retrieval failures visible.

## Training tradeoff

The scale-removed map preserves native L-BFGS accuracy and maps, while increasing training CE. Here routing is L; errors include all four training loads, and maps are load 32 out of 768.

| Seed | Shape all-load CE | Full all-load CE | Shape / full errors | Shape / full maps |
|---|---:|---:|---:|---:|
| 3473 | 0.0100502 | 0.0004363 | 0 / 0 | 768 / 768 |
| 3479 | 0.0110856 | 0.0020771 | 0 / 0 | 768 / 768 |
| 3481 | 0.0115666 | 0.0035381 | 6 / 6 | 762 / 762 |

At the same L routing, scaling the N decoder preserves its 149/14/218 errors, despite lowering training CE. Non-scalar decoding differences therefore matter for the observed class-decision improvement; positive uniform scale cannot create that improvement. This is an endpoint effect, not a theorem that the observed parameter change is universally required.

All loads, splits and unique/repeated-key groups, including empty groups, appear in the [864-row CSV](../../../results/decoder_scale_shape/evaluation.csv). Complete-map fields measure whole histories and repeat across group rows; absent groups have blank CE/accuracy. These scores were not used to select settings.

## What was actually run

We scored **288 exact decoder-intervention cells / 1,870,848 repeated answers** from cached audited logits. Producer time was 0.92 seconds: scalar transformations and score reduction, not new training or model forwards. CPU float64, one thread; zero optimizer, gradient and solver calls. The audit separately recomputed all 144 source cells through saved models using an independent matrix recurrence, then checked projection arithmetic, transformed scores, scalar metrics and primary decisions.

Two runtime preflight methods passed in 4.510 seconds. Controls covered an initialized saved source/evaluator and transformed scorer/audit, actual scaled-classifier forwards, exact ties/empty groups, negative projection rejection and deliberate projection/logit/label/count/decision/missing-group faults. The [audit](../../../results/decoder_scale_shape/audit.json) passes all 288 cells and protects 52 files. Maximum discrepancies: source logits 4.55e-13, transformed logits 2.28e-13. Weight projections and the primary conjunction receive separate scalar checks; the full contrast reducer is shared. Prior artifacts stay unchanged.

Three initialization seeds share one previously opened dataset; shifted histories and repeated scoring are not independent experiments. No calibration metric, fresh-distribution test or deployment improvement was measured.

## What closes, what remains

Close the decoder scale/shape branch without more scales, donor variants or training extensions. We now distinguish the held-out CE penalty, dominated by scale here, from inaccurate retrieval and endpoint routing/decoder compatibility.

Earlier [scale-pressure](readout-scale-pressure-results.md) and [Adam transport](readout-adam-transport-results.md) studies already identify expansion incentives and stored-moment movement in other windows. Do not repeat those partitions or treat them as proof of these L-BFGS updates. The remaining learning question is how the successful procedure coordinates routing and non-scalar decoding. Next, inspect the archived optimizer gradients/secant directions for cross-block transport before declaring a targeted learning intervention. No new training is queued; RL remains parked.

Sources inspected: [Guo et al. (2017), section 4.2/equation 9](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) for positive logit scaling and invariant prediction, and [Yang et al. (2024), section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2) for delta memory context. Guo et al. fit validation temperature; we use a weight-derived scalar and do not establish calibration. The decomposition, criterion and measured effects are local, not paper forecasts.
