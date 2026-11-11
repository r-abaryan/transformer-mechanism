# Separating query and key effects

K-only replacement produces the larger rescue in every varied-position model. Mean +8 accuracy rises from 77.65% to 97.49%, compared with 79.86% for Q-only. QK together reaches 99.44%. The reverse direction also shows greater sensitivity to K.

## What Q and K mean here

The final query describes what to retrieve. Keys are the vectors it scores at available token positions. Scores are q k_j^T / sqrt(8), followed by causal softmax. Both vectors matter: a score change can come from either side.

Q-only uses Q from +4 with K/V from +8. K-only uses K from +4 with Q/V from +8. QK uses both from +4. The residual, FFN, normalization and output remain at the recipient offset. All patches use the same example within the same model. No values are transplanted.

Both heads and all rows are replaced. In this final attention block, only the final query row affects the final prediction; the subsequent FFN and normalization act separately at each position. K replacement still changes all available token keys, not just the target value's key.

## Method and results

Five frozen varied-position checkpoints; 7,200 opened test sequences from 400 mappings. No training or architecture change. [Protocol](../../query-key-protocol.md) fixed before execution. These are exploratory interventions, not fresh confirmation.

| Replacement | +4 donor into +8: accuracy | Loss, nats | +8 donor into +4: accuracy | Loss, nats |
|---|---:|---:|---:|---:|
| None | 77.65% | 1.1138 | 99.98% | .0018 |
| Q only | 79.86% | 1.0203 | 98.45% | .0531 |
| K only | 97.49% | .0916 | 80.83% | 1.0415 |
| QK | 99.44% | .0205 | 78.36% | 1.1491 |

Five-seed means. Reverse replacements test whether the worse-offset representation can damage otherwise strong performance. They support the same ordering of sensitivity; they are not exact inverses because the recipient's remaining computation differs.

| Seed | +8 Q only | +8 K only | +8 QK | +4 reverse Q | +4 reverse K | +4 reverse QK |
|---|---:|---:|---:|---:|---:|---:|
| 461 | 89.63% | 98.58% | 99.82% | 98.97% | 90.29% | 89.44% |
| 487 | 88.82% | 98.29% | 99.69% | 99.57% | 88.31% | 84.81% |
| 509 | 82.03% | 99.57% | 99.82% | 99.85% | 82.28% | 81.60% |
| 541 | 65.68% | 95.42% | 99.40% | 96.65% | 68.38% | 64.14% |
| 571 | 73.14% | 95.57% | 98.47% | 97.19% | 74.90% | 71.81% |

Across 36,000 seed/example observations, forward K replacement rescues 7,393 answers and harms 252. Q rescues 1,702 and harms 907. QK rescues 7,870 and harms 25. These observations are dependent across seeds and mapping variants.

Mean correct-minus-best-other-logit margin rises from 3.35 to 3.67 with Q, 6.28 with K, and 6.66 with QK. Joint accuracy gain minus the two individual gains is -.253 percentage point forward and -.944 reverse. This descriptive interaction does not assign a fraction of the mechanism to either component; accuracy is nonlinear and effects depend on the recipient state.

## What this leaves us with

Observed: importing +4 keys largely restores accuracy while keeping +8 queries and values. Q still matters: adding Q to K improves mean accuracy by 1.96 points and reduces harm substantially. Neither replacement is perfect.

Supported interpretation: block-2 keys are a stronger intervention site than queries under these conditions. Unknown: which token keys matter most and whether their offset sensitivity originates in the residual position signal, block-1 attention, its FFN, or normalization. A useful replacement does not establish the unique original cause.

Stopping decision: retain this diagnostic and stop further token-key breakdown. Joint QK performs best in every seed. The [consolidated assessment](../../assessments/joint-routing-assessment.md) defines the evidence boundary and links a fresh-mapping confirmation protocol. That evaluation will distinguish donor-assisted rescue from ordinary performance.

## Evidence and limits

[Code](../../../experiments/query_key.py), [summary results](../../../results/query_key.json), [full logits](../../../results/query_key_scores.json), [audit](../../../results/query_key_audit.json). Original and joint QK metrics reproduce the preceding experiment. Attention reconstruction, sham outputs and unchanged checkpoint hashes passed; maximum discrepancy 3.82e-14. All 40 baseline/intervention summaries reproduce exactly from saved logits, including losses, margins and rescue/harm. The audit is not an independent forward implementation.

The attention equations and [primary-source interpretation discussion](../positions/position-routing-results.md#sources-and-verification) remain applicable. This is a local intervention result, not a reproduced paper benchmark, general theory of keys, or proof of length generalization. Mixed-offset representations need not occur naturally.
