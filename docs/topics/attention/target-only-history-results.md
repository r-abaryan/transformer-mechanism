# Removing competing writes recovers every tested answer

Executed 2026-10-04. **All 48 filtered cells reach 100% accuracy. All 13,973 originally wrong answers are rescued; zero correct answers are lost.** These counts include repeated endpoints and reused histories, not independent experiments.

[Short learning guide](target-only-history-guide.md) | [Frozen protocol](../../protocols/target-only-history.md) | [Executed source](../../../experiments/target_only_history.py) | [Raw report](../../../results/target_only_history/report.json) | [All groups and losses](../../../results/target_only_history/metrics.csv) | [Independent audit](../../../results/target_only_history/audit.json).

## Method

Exhaustive-feedback delta reader, seeds 3461/3463, steps 3000/12000. Use full existing train/validation/test pools: 768/192/384 histories per load. Answer every present key in sorted order. For each answer, remove all nonmatching-key writes and run the actual saved model on the retained events. Keep keys, values, beta and classifier unchanged; no padding events.

CPU float64, one thread, 311,808 evaluated answer outputs from batched model forwards and 688,128 retained write events. Zero optimizer calls or training updates. Original scores are reused from audited endpoints. Inputs have 32 writes with a stable value per key; retained lengths are 16/4/2/1 for original loads 2/8/16/32. This is a query-conditioned input intervention.

## Accuracy

Every filtered entry is **100%**, including every nonempty unique/repeat group. This table retains all original comparisons; absent groups are null in the CSV. Load 32 is unique-key; other loads are repeated-key.

| Split | Original load | 3461: 3000 % | 3461: 12000 % | 3463: 3000 % | 3463: 12000 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 2 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| Train | 8 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| Train | 16 | 99.9674 | 100.0000 | 99.9919 | 100.0000 |
| Train | 32 | 94.4499 | 98.9990 | 94.9259 | 98.5636 |
| Validation | 2 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| Validation | 8 | 99.2188 | 100.0000 | 99.6094 | 100.0000 |
| Validation | 16 | 98.2422 | 99.2188 | 98.3398 | 98.4375 |
| Validation | 32 | 86.9466 | 87.8906 | 87.0768 | 86.7350 |
| Test | 2 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| Test | 8 | 100.0000 | 99.6094 | 100.0000 | 100.0000 |
| Test | 16 | 97.4447 | 98.7956 | 97.0052 | 97.1680 |
| Test | 32 | 85.8805 | 86.4421 | 85.3027 | 86.9792 |

| Seed | Step | Test-32 CE: original → filtered | Rescued / 12,288 | Original complete maps / 384 |
| --- | ---: | ---: | ---: | ---: |
| 3461 | 3000 | 0.442601 → 0.001708325 | 1,735 | 6 |
| 3461 | 12000 | 0.976972 → 2.041959e-12 | 1,666 | 18 |
| 3463 | 3000 | 0.476152 → 0.001108613 | 1,806 | 0 |
| 3463 | 12000 | 0.879149 → 1.765654e-11 | 1,600 | 6 |

Filtered all-query correctness is 384/384 per test cell, but uses a **different memory for each query**. It does not repair the original complete-map failures. Some filtered CE values round to zero in float64; do not interpret them as exactly certain probabilities or convergence. Exact original training-fit failures remain unchanged.

## Verification and interpretation

Three preflight methods exercise the actual evaluator and audit on an initialized saved checkpoint, missing groups, tied scores, clamped nonunit keys and deliberate defects. The independent audit reconstructs labels/query order, all metrics and transitions. A locally derived clean-read formula checks every answer; a separate matrix recurrence checks all 9,216 distinct retained-sequence cases across cells. Maximum native/closed-form and matrix/closed-form logit differences are both 5.684342e-14, below the frozen 1e-9 tolerance. All 169 locked files remain unchanged.

Primary source inspected: [Yang et al. (2024), section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2), delta correction/read equations. The guide's closed form and all scores are local work. This reader is not a reproduction of their complete DeltaNet.

| Evidence | Supported conclusion | Still open |
| --- | --- | --- |
| Clean answers all correct at both endpoints | The saved reader/classifier can decode all tested clean associations | Robustness to mixed memory states |
| Every original error disappears under filtering | Competing-write removal is sufficient for recovery in these pools | Which interactions cause each error, and whether a practical repair can preserve all associations |
| Older models also recovered under supplied order/geometry | Recovery persists after exhaustive feedback and continuation | Why joint training still selects parameters that fail on mixed histories |

Stop this sufficiency diagnostic. A distinct next question is whether gradients from correct training answers oppose gradients repairing the remaining load-32 errors **on keys, values and beta**. That would examine a local learning incentive, not repeat filtering or establish historical causality. It needs a separate bounded protocol; no new training or RL ran here.
