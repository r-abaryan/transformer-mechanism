# More associations expose a delta generalization gap

Executed 2026-10-04. **In confirmation, both readers fit every training load at 100%. Softmax gets every validation/test answer correct. Delta test accuracy falls from 100% at two associations to 58.42% at 32.** Sequence length stays 32. This is a held-out-context gap under the tested recipe, not incomplete training fit or a measured hard capacity bound.

[Learning guide](association-load-guide.md) | [Frozen design](../../protocols/association-load.md) | [Source](../../../experiments/association_load.py) | [Report](../../../results/association_load/report.json) | [Every seed/pool accuracy and CE](../../../results/association_load/metrics.csv) | [Independent audit](../../../results/association_load/independent_audit.json).

## Method

Thirty-two known key symbols, six values. Contexts contain M=2/8/16/32 distinct keys; every key occurs exactly 32/M times with a consistent value. The final query names a present key. Both readers learn 16-dimensional unit keys, 8-dimensional values and a classifier, with 615 parameters. The same model per run trains on all four loads. Softmax recency is zero because assignments never conflict. Delta uses its unchanged correction rule. All shapes, widths and parameter counts stay fixed across loads.

Development: two seeds, two declared Adam recipes, both arms. Both select constant .003 using training fit/loss only. Six fresh confirmation runs use seeds 3323/3329/3331 and disjoint context families. Every run has 3,000 updates, batch 64, 192,000 final-query targets and 6,144,000 input records; each load receives 48,000 targets. CPU float64, one thread. All 14 runs complete: 42,000 updates, 2,688,000 targets and 86,016,000 records. Summed per-run training/evaluation time is 416.3 seconds, excluding preflight/final audit; this is execution provenance, not a speed comparison.

Training/validation/test pools contain 768/192/384 examples per load. Each test pool has 64 context families with six cyclic value renamings, not 384 independent contexts. Every query key and label is balanced; families remain within one pool and exclude development/confirmation overlap. Symbols/templates remain shared. Full pools are evaluated; tests open after all confirmation training finishes.

## Confirmation results

Three-seed means. Both readers' training accuracy is 100% at every load. Softmax validation/test accuracy is also 100% in every seed/load; its mean CE is approximately 0.000279 in all pools.

| Associations | Writes/key | Softmax test % | Delta validation % | Delta test % | Delta CE: train / validation / test |
|---|---:|---:|---:|---:|---|
| 2 | 16 | 100.00 | 100.00 | 100.00 | 0.000016 / 0.000016 / 0.000016 |
| 8 | 4 | 100.00 | 98.96 | 99.48 | 0.000141 / 0.037676 / 0.013528 |
| 16 | 2 | 100.00 | 90.97 | 90.45 | 0.000815 / 0.382591 / 0.356397 |
| 32 | 1 | 100.00 | 74.48 | 58.42 | 0.001518 / 1.996401 / 2.321675 |

| Seed | Delta test 2 % | Test 8 % | Test 16 % | Test 32 % |
|---|---:|---:|---:|---:|
| 3323 | 100.00 | 100.00 | 92.19 | 61.20 |
| 3329 | 100.00 | 98.44 | 90.36 | 51.04 |
| 3331 | 100.00 | 100.00 | 88.80 | 63.02 |

The >=99% all-seed training-fit rule passes at every load for both arms. Delta's >=95% test reliability passes at 2/8 and fails at 16/32. Its declared >=10-point load-drop criterion passes in every seed; softmax has zero drop. Do not hide the weakest 32-key seed behind the mean.

Constant-class accuracy is 16.67%. Final-event-value baselines score 53.125%/23.4375%/14.0625%/15.625% at 2/8/16/32. Supplied first-match lookup gets 100% because values are consistent. It verifies task solvability, not learned behavior.

## Can a redundant write disturb another association?

Append one existing nonqueried key with its unchanged value. The requested answer stays fixed. This is a separately declared 33-event probe; it is not a larger-load training condition. Counts pool the same 384 test examples over three checkpoints, 1,152 evaluations per load.

| Associations | Delta correct answers harmed | Wrong answers rescued | Mean delta read-shift norm | Mean absolute query/write overlap |
|---|---:|---:|---:|---:|
| 2 | 0 | 0 | 0.000030 | 0.1662 |
| 8 | 0 | 0 | 0.132432 | 0.1660 |
| 16 | 2 | 0 | 0.445699 | 0.1555 |
| 32 | 41 | 29 | 0.901939 | 0.1658 |

Softmax has zero class changes in every probe. Delta has 43 harms and 29 rescues overall. At 32 associations, harms/rescues by seed are 11/3, 15/14 and 15/12. These are input interventions on fixed weights, not additional training.

The exact delta read change is `beta*(q dot k)*(v-S^T k)`. Independent equations verify every before/after score, and the native state/read changes match that identity. A redundant write can still make a correction because the matrix's current read differs from the supplied value. Similar average key overlap across loads does not imply similar read changes: the correction residual also matters. Read movement and classification failure are different measurements, as the eight-key row demonstrates.

The probe establishes harmful cross-effects for these saved models. Its mixed harms/rescues do not establish that the same mechanism caused every original test error. Softmax's added event also changes multiplicities.

## What this adds, and its limits

Longer histories over a small map and more distinct associations are different demands. Here, the same delta model fits the finite training contexts at high load yet transfers poorly to new contexts. That is stronger evidence of a practical load sensitivity than an unfitted arm, while its cause and best remedy remain open.

Increasing load also reduces repetitions and changes intervening keys. All keys are known, outputs come from six classes, and contexts use structured roles. Sixteen key dimensions do not imply a 16-key classification ceiling. This study does not isolate rank, optimizer behavior, memorization, general language quality or arbitrary-valued storage capacity. Earlier eight-key success remains valid; historical scores are not paired controls for this new task.

[DeltaNet 2024, section 2.2](https://arxiv.org/html/2406.06484v2) defines key-directed correction and its dependence on the prior read. Our measured class harms instantiate local interference; the paper does not predict these exact scores or establish their training cause. Full DeltaNet layers and optimized kernels remain outside this study.

## Verification and stopping

Four preflight methods pass: initialized outputs/all-parameter gradients at all four loads, actual saved reload/evaluator/audit entry, absent groups and deliberate equation, recency, label, leakage, scalar and fit-rule errors. Four disposable CPU/GPU optimizer steps precede the frozen formal design. GPU float64 profiles fit comfortably, with allocated peaks of 69,810,688/71,916,032 bytes for softmax/delta. Formal training uses CPU; those profiles are not formal training-memory measurements.

The initial added-recency control was insensitive because its two keys shared a value. Its failed sources/report are [preserved](../../../results/association_load/preflight_failure_001/failure.json). Explicitly different values make the corrected control detect the error. No formal run preceded this correction; thresholds and equations remain unchanged.

Independent matrix equations and Python scalar reductions pass 136 final pool cells (62,976 predictions) and 24 write cells (9,216 before/after pairs). Maximum logit difference is 1.42e-14; zero argmax disagreements. All 14 optimizer counters, initialization/batch streams, decision rules and 148 targeted earlier files verify. References share parameters/data/PyTorch, not an independent software stack. [All per-seed write metrics](../../../results/association_load/write_probes.csv).

Close this bounded study. Preserve the reliable softmax control, delta's trained-but-poorly-transferring high-load behavior and the mixed write effects. No extra epochs, wider state, replacement seeds or novel architecture proposal is selected.
