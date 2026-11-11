# Beta swaps: a contribution to retention, not a complete repair

2026-10-04. Actual saved-model evaluation and independent audit completed, with zero training updates. Lowering beta improves fixed-query final answers; raising beta harms resampled-query final answers in both seeds. Both directional criteria pass. Complete-map reliability fails.

## Method

Four delta-query-coverage checkpoints, two seeds. Native versus opposite-arm beta at the same seed; only `write_logit` changes in disposable clones. Keys, values, classifier and 16x8 state size remain unchanged. Eight opened development train/validation pools, loads 2/8/16/32, all present queries. No fresh confirmation or test access. CPU float64, one thread; evaluation took 24.59 seconds, excluding preflight/audit.

Immediate reads follow each key's first write. Final reads share the state after event 32. Complete maps count only the latter. [Protocol](../../protocols/beta-swap.md) | [Learning guide](beta-swap-guide.md).

## Validation at 32 keys

Each row has 6,144 answers from 192 histories. Rescues/harms compare final answers with that checkpoint's native beta. `donor_beta` means lower beta for fixed and higher beta for resampled.

| Seed | Model | Variant | Beta | Immediate % | Final % | Final CE | Complete maps | Rescues / harms |
|---|---|---|---|---|---|---|---|---|
| 3461 | fixed | native | 0.94757 | 100.00 | 60.56 | 1.94225 | 0/192 | 0 / 0 |
| 3461 | fixed | donor_beta | 0.32236 | 99.40 | 79.02 | 0.62904 | 0/192 | 1248 / 114 |
| 3461 | resampled | native | 0.32236 | 99.64 | 86.34 | 0.39657 | 0/192 | 0 / 0 |
| 3461 | resampled | donor_beta | 0.94757 | 100.00 | 68.51 | 2.19231 | 0/192 | 100 / 1196 |
| 3463 | fixed | native | 0.96204 | 100.00 | 62.22 | 1.99770 | 0/192 | 0 / 0 |
| 3463 | fixed | donor_beta | 0.31870 | 99.17 | 79.98 | 0.60855 | 2/192 | 1198 / 107 |
| 3463 | resampled | native | 0.31870 | 99.89 | 85.47 | 0.46407 | 0/192 | 0 / 0 |
| 3463 | resampled | donor_beta | 0.96204 | 100.00 | 66.39 | 2.57397 | 0/192 | 86 / 1258 |

Fixed models gain **18.46 / 17.76 points** after lowering beta. Resampled models lose **17.84 / 19.08 points** after raising it. Every effect exceeds the frozen one-point margin. Harms remain: gentler writes are not beneficial to every answer.

Fixed mean final accuracy rises **61.39% to 79.50%**. Resampled mean falls **85.90% to 67.45%**. At matched lower beta, resampled models retain a **6.40-point** mean advantage. This is a conditional comparison, not an additive decomposition of the historical training gain.

## Every training and validation load

Final accuracy percentages list **2 / 8 / 16 / 32**. Full stage-wise CE and paired counts are in the [128-row CSV](../../../results/beta_swap/metrics.csv).

| Seed | Model | Variant | Training % | Validation % |
|---|---|---|---|---|
| 3461 | fixed | native | 100 / 99.90 / 89.82 / 64.06 | 100 / 99.61 / 90.82 / 60.56 |
| 3461 | fixed | donor_beta | 100 / 100 / 95.59 / 79.65 | 100 / 99.87 / 96.06 / 79.02 |
| 3461 | resampled | native | 100 / 100 / 99.72 / 89.75 | 100 / 100 / 97.59 / 86.34 |
| 3461 | resampled | donor_beta | 100 / 99.97 / 94.44 / 70.43 | 100 / 100 / 92.48 / 68.51 |
| 3463 | fixed | native | 100 / 99.76 / 89.53 / 63.48 | 100 / 99.61 / 91.28 / 62.22 |
| 3463 | fixed | donor_beta | 100 / 99.63 / 95.83 / 80.75 | 100 / 100 / 96.13 / 79.98 |
| 3463 | resampled | native | 100 / 100 / 99.62 / 90.03 | 100 / 100 / 98.83 / 85.47 |
| 3463 | resampled | donor_beta | 100 / 100 / 94.34 / 69.40 | 100 / 100 / 93.62 / 66.39 |

Training-32 complete maps: fixed native 0/768 in both seeds; lower beta gives 3/768 and 6/768. Resampled native gives 27/768 and 24/768; higher beta gives zero. The best swapped validation count is 2/192, far below the frozen 95% reliability criterion.

## What this establishes

Write strength contributes to final-answer survival in these checkpoints. Raising beta makes resampled immediate validation answers perfect while worsening final accuracy. Correct early classifications can therefore fail after later writes. This does not prove literal information erasure or identify every damaging event.

The swap isolates beta's inference effect within a fixed model. Training jointly changes beta, geometry and decoding. We have not proved why the optimizer chose these strengths, isolated the remaining gap, or found a learned retention solution. A donor beta can also change decoder calibration.

## Verification and sources

Five reusable preflight methods pass, including beta-only/self-copy checks, analytical/numerical derivatives, actual checkpoint evaluator/audit and intentional errors. All 64 intervention cells, 128 stage-metric cells and **890,880 predictions** pass independent matrix and scalar reductions: maximum logit error 1.87e-14, zero argmax disagreements. Native archived scores reproduce; **286 earlier files remain unchanged**. Data, weights and PyTorch are shared; policy replay shares tested functions. [Audit](../../../results/beta_swap/independent_audit.json) | [Raw report](../../../results/beta_swap/report.json).

[Yang et al., DeltaNet 2024, section 2.2](https://arxiv.org/html/2406.06484v2) was inspected for correction/read equations. Its input-dependent beta differs from our global scalar; no paper benchmark or predicted local score is claimed. The guide's cross-read and two-write formulas are local derivations checked in reusable tests.

Close this diagnostic and [consolidate the retention branch](../../assessments/compact-retention-assessment.md). No beta grid, retraining or automatic extension.
