# Why can equal values make the score look better?

**Making nonqueried values equal to the answer improves accuracy on matched inputs, without increasing average target-slot attention.** This supports an answer-redundancy explanation. It does not prove a complete internal mechanism.

## Short explanation

“All equal” means `A->2, B->2, C->2`. Whatever key is queried, the answer is 2. Reading the wrong pair can still give the correct answer. Therefore high accuracy does not necessarily mean correct retrieval.

The earlier 96.68% came from an observed subgroup. This new test changes the nonqueried values while keeping the rest fixed. Its all-equal accuracy is 95.63%, on a different balanced sample. These numbers are not contradictory; they describe different input sets.

## Matched construction

The [protocol](../../value-counterfactual-protocol.md) fixed 16 ordered key triples, all three query ranks, all four target labels and all six ordered distinct distractor choices. This gives 1,152 matched families. Query ranks and target labels are exactly balanced. A family keeps keys, order, query, target value and positions unchanged.

Example with B queried:

| Condition | Values for A, B, C | Correct answer |
|---|---|---|
| Distinct | 1, 2, 3 | 2 |
| Repeat at first nonqueried slot | 2, 2, 3 | 2 |
| Repeat at second nonqueried slot | 1, 2, 2 | 2 |
| All equal | 2, 2, 2 | 2 |
| Repeat wrong value | 1, 2, 1 | 2 |

Evaluate shifts 0/+8, original and the fixed later-two paired-key intervention, five frozen models. No training. Families and conditions are dependent; some edited inputs repeat, especially all-equal inputs. The sample is not claimed globally unseen. Input edits can change all Q/K/V representations, so this is not a V-only intervention.

## Predictions

All zero-shift conditions reached 100% accuracy for every seed. At +8, means across seeds were:

| Condition | Original accuracy | Forced accuracy | Forced cross-entropy, nats |
|---|---:|---:|---:|
| Distinct | 57.40% | 81.98% | 0.9232 |
| Target repeated at first nonqueried slot | 74.06% | 88.06% | 0.5896 |
| Target repeated at second nonqueried slot | 81.91% | 91.32% | 0.3634 |
| All equal | 92.60% | 95.63% | 0.1331 |
| Wrong value repeated | 53.06% | 81.01% | 0.9648 |

Repeating an arbitrary wrong value does not reproduce the benefit. The target-answer repetitions matter under these matched edits. Which nonqueried slot is changed also matters; these two conditions are not interchangeable.

| Seed, +8 with forcing | Distinct | All equal | Gain, percentage points |
|---|---:|---:|---:|
| 137 | 91.75% | 98.96% | +7.20 |
| 163 | 95.66% | 98.44% | +2.78 |
| 191 | 90.45% | 96.35% | +5.90 |
| 223 | 63.45% | 97.92% | +34.46 |
| 257 | 68.58% | 86.46% | +17.88 |

The predeclared >=5-point gain in every seed **failed** because seed 163 gained 2.78 points. All five improved, but a positive gain is not the same as meeting the criterion. Seed 257 still performs poorly even when all answers agree.

## Routing versus answer accuracy

For block-2 final-query attention, let A_t be weight on the queried pair's value position. Also measure sum of A_j over value positions whose literal symbol equals the answer. The second metric can rise merely because more positions carry that answer; it is not a selection-quality metric.

Values below are means over families and seeds at +8 with forcing. Entries are head 0 / head 1. Head roles are not aligned across seeds; all per-seed observations are preserved.

| Condition | Target-slot weight | Weight on answer-valued slots | Target highest among value slots, ties included |
|---|---:|---:|---:|
| Distinct | 0.6847 / 0.5674 | 0.6847 / 0.5674 | 86.67% / 83.02% |
| All equal | 0.6782 / 0.5581 | 0.8369 / 0.7083 | 89.37% / 85.31% |
| Wrong value repeated | 0.6842 / 0.5663 | 0.6842 / 0.5663 | 86.67% / 83.47% |

All-equal accuracy rises by 13.65 points while average target-slot weight slightly falls in both heads. The target-top-frequency proxy rises modestly. Thus routing is not literally unchanged, and not every proxy worsens. The evidence supports increased tolerance for reading another answer-bearing position; it does not quantify how much of the accuracy gain is mediated by that effect. Literal value symbols are also not the contextual V vectors themselves.

## Evidence and next question

[Script](../../../experiments/value_counterfactuals.py), [exact matched inputs](../../../results/value_counterfactual_inputs.json) and [raw results](../../../results/value_counterfactuals.json) preserve all 20 conditions per seed, predictions, routing metrics and hashes. Family label/key checks and exact target/rank balance passed. Explicit block-2 attention matched library output within 3.71e-15. Finite outputs, checkpoint integrity, source hashes and paired-count arithmetic passed.

This separates an input-level redundancy effect from an unsupported claim of improved binding. It does not resolve remaining shift failures or justify architecture changes. The [attention equations and source comparison](../attention/qkv-results.md#learn-the-operation) remain applicable.

Next consolidate the mechanism evidence and its limits before adding more patches. For a stronger retrieval test, predeclare balanced unique-value cases and evaluate fresh key/mapping configurations, separating answer accuracy from source-routing measures. Do not treat all-equal success as evidence of correct association.
