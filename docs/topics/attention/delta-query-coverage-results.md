# Fixed versus resampled queries: better answers, incomplete retention

2026-10-04. Four actual development runs completed and audited. Resampling raises mean all-query validation accuracy at 32 keys from **61.39% to 85.90%**. Both seeds miss the frozen 99% all-query training gate. Fresh confirmation is skipped; no test scores or reliable retention solution are claimed.

## Method

Original 615-parameter delta reader, 16x8 state, no overlap penalty. Same histories, initialization and minibatch streams per pair; only the supervised query changes. Both arms receive one target per example. Adam .003, 3,000 updates, batch 64, CPU float64, one thread. Total: 12,000 updates, 768,000 targets, 24,576,000 events. Training plus final evaluation took 137.55 seconds across four runs, excluding preflight/profile/audit time.

Per load: 768 training histories and 192 validation histories, grouped into families with six value renamings. These are held-out contexts at fixed 32-event length, not length extrapolation. [Protocol](../../protocols/delta-query-coverage.md) | [Guide and equations](delta-query-coverage-guide.md).

## All-query accuracy

Percentages list loads **2 / 8 / 16 / 32**. Every present key reads the same unchanged final state.

| Seed | Arm | Training % | Validation % |
|---|---|---|---|
| 3461 | Fixed | 100 / 99.90 / 89.82 / 64.06 | 100 / 99.61 / 90.82 / 60.56 |
| 3461 | Resampled | 100 / 100 / 99.72 / 89.75 | 100 / 100 / 97.59 / 86.34 |
| 3463 | Fixed | 100 / 99.76 / 89.53 / 63.48 | 100 / 99.61 / 91.28 / 62.22 |
| 3463 | Resampled | 100 / 100 / 99.62 / 90.03 | 100 / 100 / 98.83 / 85.47 |

Both pairs improve at 32 keys: validation gains **25.78 / 23.24 points**. This is paired development evidence, not fresh confirmation.

## Assigned queries versus complete maps at 32 keys

Assigned-query fit tests the fixed arm's actual objective. A complete map requires all 32 answers correct. CE below uses all queries, giving a common measurement despite different training objectives.

| Seed | Arm | Assigned accuracy, train / validation % | All-query CE, train / validation | Complete maps, train / validation |
|---|---|---|---|---|
| 3461 | Fixed | 100 / 49.48 | 1.95581 / 1.94225 | 0/768 / 0/192 |
| 3461 | Resampled | 92.19 / 79.17 | 0.33718 / 0.39657 | 27/768 / 0/192 |
| 3463 | Fixed | 100 / 58.33 | 2.03351 / 1.99770 | 0/768 / 0/192 |
| 3463 | Resampled | 91.80 / 79.69 | 0.30974 / 0.46407 | 24/768 / 0/192 |

Fixed fits assigned queries at 100% in every training load. Resampled fits all-query training above 99% at loads 2/8/16, but not 32. Its mean training all-query accuracy there is **89.89%**, versus fixed **63.77%**. Zero complete validation maps means the higher answer average still falls short of reliable whole-map storage.

## What training actually covered

Each arm receives 48,000 targets at load 32. There are 24,576 possible training history/key combinations.

| Seed | Arm | Combinations supervised at least once | Learned beta | Squared key overlap R |
|---|---|---|---|---|
| 3461 | Fixed | 768/24,576 (3.12%) | 0.94757 | 0.04265 |
| 3461 | Resampled | 21,169/24,576 (86.14%) | 0.32236 | 0.03315 |
| 3463 | Fixed | 768/24,576 (3.12%) | 0.96204 | 0.04161 |
| 3463 | Resampled | 21,074/24,576 (85.75%) | 0.31870 | 0.03303 |

Random draws leave roughly 14% of those combinations unsupervised. Query coverage, beta and geometry change together during learning. These observations do not isolate beta or overlap as the cause, or show that unsampled combinations explain the remaining errors.

## Verification and boundaries

Five formal preflight methods pass. A wrong-label fixture initially shared repeated event objects; it was corrected before freezing or training. [Summary](../../../results/delta_query_coverage/summary.json) records that correction.

The audit checks four checkpoints, 64 metric cells and 238,080 predictions using independent matrix equations and Python CE/map reductions. It also reconstructs every minibatch, query/label stream and exposure count. Maximum logit difference is 1.42e-14 in the [audit](../../../results/delta_query_coverage/independent_audit.json); zero argmax disagreements. All 249 protected earlier files remain unchanged. Audit shares data, weights and PyTorch; gate replay shares tested policy functions. Prediction counts are not independent seeds.

Disposable complete GPU optimizer steps peak at 71,903,232 allocated / 73,400,320 reserved bytes in both arms. These are resource profiles, not GPU training or speed measurements. [Full metrics CSV](../../../results/delta_query_coverage/metrics.csv) contains every seed/load/split accuracy and CE; [raw report](../../../results/delta_query_coverage/report.json) includes exposure, streams and checkpoints.

The fit gate passes for fixed assigned queries and fails for resampled all queries. Confirmation is therefore skipped as declared. Broader query exposure helps under this budget, but does not solve simultaneous retention. Optimization, finite exposure and representation remain possible limits; this does not prove a hard capacity ceiling. No automatic extra training or architecture change.

## Paper comparison

[Yang et al., DeltaNet 2024, section 2.2](https://arxiv.org/html/2406.06484v2) supplies the correction/read equations inspected here. Our scalar beta and standalone compact reader differ from its input-dependent beta and complete model. The finite-average query objective is a local derivation. Neither the paper nor earlier [Transformer query exposure](../training/query-exposure-results.md) predicts these scores or provides a paired control. This is not a paper benchmark reproduction.
