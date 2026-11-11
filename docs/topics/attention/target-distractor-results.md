# Target and other value-position keys

**Restoring the target-value K helps most on its own. Restoring all three value-position K vectors is needed to meet our every-seed rescue criterion among the tested subsets.** This is conditional on compact final Q in both heads.

## Short explanation

Suppose the input says `A->2, B->3, C->1`, and asks for B. The target-value position contains 3. The other value positions contain 2 and 1. We change their learned attention K vectors, not their symbols or attention V vectors.

Attention compares all source scores together. Increasing the target's score can help. Reducing other scores can also help. Changing both can work better than either alone. Other pairs sometimes contain the same answer symbol, so an “other value” is not always harmful.

The final query stays compact throughout this experiment. Thus the forward baseline is **63.57%**, the previous Q-only condition, rather than the wholly shifted 60.60% baseline.

## Method and math

Follow the [predeclared protocol](../../target-distractor-protocol.md). Five frozen models, 5,120 inspected six-derived crops per model, no training. Each crop contains three pairs. Forward: shifted recipient, compact K donors. Reverse: compact recipient, shifted K donors. Recipient V, residual stream and downstream computation remain in their respective state. Replace both heads at the target-value slot, the other two value slots, or all three. Final Q is compact in every case.

For final-query head h, score s_j=q_7 dot k_j/sqrt(8), and probability a_j=exp(s_j)/sum_l exp(s_l). We measure target probability, the sum over other value positions, and margin m=s_target-max(s_other values). A larger margin means the target outscores those two sources more strongly. It does not summarize competition from all other tokens or guarantee a correct final prediction.

## Accuracy and reverse controls

Means across five seeds. Reverse accuracy drops are relative to compact baseline; forward rescue is relative to the Q-only baseline.

| Direction and K replacement | Accuracy | Cross-entropy, nats | Predeclared criterion |
|---|---:|---:|---|
| Forward: none | 63.57% | 2.0033 | Below 90% |
| Forward: target only | 89.96% | 0.4503 | All-seed 90% failed |
| Forward: others only | 70.70% | 1.9173 | All-seed 90% failed |
| Forward: all three | 98.66% | 0.0502 | All-seed 90% passed |
| Reverse: none | 99.996% | 0.0005 | Reference |
| Reverse: target only | 70.91% | 2.0440 | Every-seed >=10-point drop passed |
| Reverse: others only | 89.98% | 0.4834 | Every-seed >=10-point drop failed |
| Reverse: all three | 59.50% | 2.5061 | Every-seed >=10-point drop passed |

| Seed | Forward target | Forward others | Forward all | Reverse target | Reverse others |
|---|---:|---:|---:|---:|---:|
| 137 | 90.74% | 73.89% | 99.92% | 73.93% | 90.72% |
| 163 | 84.63% | 71.17% | 97.97% | 74.04% | 84.41% |
| 191 | 93.01% | 70.80% | 98.93% | 68.09% | 93.48% |
| 223 | 88.91% | 67.32% | 99.73% | 69.12% | 88.83% |
| 257 | 92.52% | 70.31% | 96.74% | 69.38% | 92.44% |

## What happens to routing?

Forward measurements, averaged over crops and seeds. Each cell lists head 0 / head 1. Head indices are not guaranteed to have aligned semantic roles across independent models. Per-seed measurements remain in the report.

| Compact K supplied | Target probability | Other-value probability sum | Target-minus-largest-other score |
|---|---:|---:|---:|
| None | 0.376 / 0.306 | 0.346 / 0.304 | 0.462 / 0.333 |
| Target | 0.720 / 0.649 | 0.197 / 0.204 | 3.548 / 3.004 |
| Others | 0.475 / 0.377 | 0.276 / 0.258 | 1.313 / 0.976 |
| All three | 0.832 / 0.718 | 0.058 / 0.096 | 4.399 / 3.648 |

The target-only replacement changes the target score while leaving other source scores fixed. The others-only replacement changes the two other-value scores while leaving the target score fixed. Both improve the mean margin under compact Q. Their score-margin changes combine algebraically here, but the final accuracy effects are not additive: softmax and downstream computation are nonlinear.

## Conclusions and limits

The controlled replacements support contributions from both the target and other value-position K vectors. Target restoration alone has a larger accuracy effect in this setting. Reverse target corruption also causes substantial harm. This is stronger evidence than inspecting attention probabilities alone, but it is not a necessity proof or a complete binding circuit.

All findings depend on compact final Q. We have not isolated the behavior with shifted Q here. Target selection uses the known query, mixed activations are artificial, data are reused, and crops are dependent. We have not shown how block 1 constructs these K vectors. The [attention definitions and source comparison](qkv-results.md#learn-the-operation) remain unchanged; no new architecture or paper benchmark is claimed.

## Verification and next step

[Script](../../../experiments/target_distractor_keys.py) and [raw report](../../../results/target_distractor_keys.json) preserve all eight conditions per seed, routing statistics and paired rescue/harm counts. Forward none/target/all reproduced previous metrics within 1e-10. Reverse no-replacement reconstruction had maximum logit error 8.89e-15. Label checks, softmax row sums, finite outputs and unchanged checkpoints passed. Source hashes and paired-count arithmetic were checked separately.

Follow-up completed: [block-1 contributions to block-2 K](block1-key-results.md). Compact block-1 attention suffices for strong rescue under the controlled projection procedure. Its source-token mechanism remains untested.
