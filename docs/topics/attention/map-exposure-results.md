# Matched map-exposure continuation

Completed and independently audited, 2026-10-05. Replacing half the high-load maps lowers unseen cross-entropy in every seed, but accuracy changes are mixed. All replacement arms miss both active fitting and original-map retention gates. None passes the frozen benefit or repair rule.

Read the [guide](map-exposure-guide.md) for the loss and retention checks, and the [protocol](../../protocols/map-exposure.md) for frozen budgets and criteria.

## Method and results

Same 615-parameter delta reader, saved weights and Adam moments after 12,000 updates. Seeds 3473/3479/3481 receive 6,096/6,144/6,096 further updates per arm, batch 64, CPU float64. Reuse keeps all training maps. Half-new replaces 64 of 128 load-32 bases while preserving key order, queries, active pool size and sampling streams. Six cyclic value shifts per base give 768 histories. Lower-load data stays unchanged. Cumulative high-load exposure increases from 128 to 192 distinct map families; new maps lacked the original warm-up exposure. Every history still has 32 events: this measures new-map transfer at fixed length, not longer-input transfer.

Primary fresh validation/test have 192/384 histories at each load. High-load semantic maps exclude original and added maps. At load 2, 18 validation and 24 test histories reuse original training associations despite having different histories; this overlap is disclosed, not treated as unseen associations. Six shifts are related examples; these are three reused initialization seeds on one shared new draw.

At load 32, cells show **accuracy % / cross-entropy**, all queries. Complete maps require every answer correct.

| Seed | Arm | Active training | Original training | Fresh validation | Fresh test | Complete test maps /384 |
|---|---|---|---|---|---|---|
| 3473 | Reuse | 99.93 / .0146 | 99.93 / .0146 | 87.11 / 1.8849 | 87.17 / 1.6497 | 18 |
| 3473 | Half-new | 98.50 / .0579 | 94.55 / .3170 | 88.41 / .8150 | 87.92 / .7919 | 18 |
| 3479 | Reuse | 99.53 / .0236 | 99.53 / .0236 | 85.61 / 1.7865 | 87.60 / 1.4410 | 12 |
| 3479 | Half-new | 98.75 / .0414 | 93.58 / .4823 | 86.60 / 1.0508 | 86.13 / 1.2073 | 12 |
| 3481 | Reuse | 99.52 / .0279 | 99.52 / .0279 | 86.41 / 1.5245 | 85.57 / 1.7078 | 0 |
| 3481 | Half-new | 98.74 / .0474 | 93.21 / .4938 | 87.52 / .9946 | 86.06 / 1.1232 | 0 |

Fresh test accuracy changes are +0.757/−1.465/+0.488 percentage points. The >=0.5-point criterion is met only by seed 3473, whose fitting gate fails. Test complete-map counts are unchanged in every pair. Lower CE does not establish better class ranking, calibrated confidence or a learned retrieval repair; its scale contribution was not measured here.

All lower-load training groups pass 99%; the weakest accuracy is 99.9919% (half-new seed 3473, load 16). Fresh lower-load validation/test minima are 96.875%/97.2656%, both at load 16. All seeds, loads, groups, stages and original reference splits are retained in the [raw report](../../../results/map_exposure/report.json) and [1,080-row group table](../../../results/map_exposure_documentation/group_metrics.csv).

In the group table, accuracy is a fraction, CE uses natural logarithms, and step counts continuation updates after the 12,000-update warm-up. Empty groups remain explicit.

## Where original-context accuracy falls

A [descriptive partition](../../../results/map_exposure_documentation/descriptive_cohorts.json) uses the frozen replacement membership, with no new model evaluation or outcome selection. Half-new accuracy on original **removed** maps is 90.27/88.06/87.27%; on original **retained** maps it is 98.84/99.10/99.15%. Reuse scores 99.41% to 100% across those cohorts. Most deterioration is therefore concentrated on contexts no longer receiving feedback.

This is loss of performance on previously solved contexts. It does not prove that the weights memorized literal maps, nor isolate data diversity from withdrawal of old feedback. Mixed unseen accuracy and incomplete active fitting leave the finite-reuse explanation unresolved. This Adam contrast also does not isolate the earlier L-BFGS trajectory causes. No budget extension, replacement seeds or further exposure variants were run.

## Execution and verification

Six actual continuations: 36,672 updates/gradients, 2,347,008 sampled histories, 75,104,256 events and 34,031,616 supervised answers. Producer runtime 1,063.875 seconds, excluding preflight/profile. Evaluation covers 360 cells and 2,672,640 repeated answers. Two preflight methods passed; discarded profiles include optimizer state. CUDA peaks are inapplicable; CPU peak memory was not measured.

The [audit](../../../results/map_exposure/audit.json) passes every update and score cell. All three reuse endpoints, optimizer states and traces reproduce the archive exactly; 39 protected files are unchanged. Maximum gradient/Adam/logit discrepancies are 8.47e-15/1.78e-15/1.71e-13. Labels, matrix forwards, scalar Adam and prediction criteria receive separate checks; the training helper and decision reducer are shared. [Documentation verification](../../../results/map_exposure_documentation/verification.json) separately checks tables, counts, cohorts and links without rerunning training.

Evidence: [report](../../../results/map_exposure/report.json), [frozen lock](../../../results/map_exposure/lock.json), [producer](../../../experiments/map_exposure.py), [auditor](../../../experiments/audit_map_exposure.py) and [reusable checks](../../../tests/test_map_exposure.py). Binary data, checkpoints and full gradient/update traces remain locally preserved and Git-ignored.

Sources inspected: [Yang et al. (2024), section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2), delta writes/reads; [Muennighoff et al., JMLR (2025), abstract](https://www.jmlr.org/papers/v26/24-1000.html), repeated versus unique language data at fixed compute. Their language study finds little loss cost for several repetitions in its setting. Our mixed accuracy result neither reproduces nor refutes that result. The local intervention, cohort partition and measured scores are separate evidence.
