# Routing after varied-position training

Replacing block-2 Q/K from +4 largely rescues +8 predictions in all five varied-position models. Replacing V alone does not. However, unrelated donors with the same target position also rescue. This supports a routing intervention effect, not a complete account of content-specific retrieval.

## Method and math

Use all ten frozen [position-training models](position-training-results.md) and the same 7,200 test examples (400 mappings). Inspect +4 and +8. This set was already opened; the [diagnostic protocol](../../position-routing-protocol.md) is exploratory. No retraining or architecture change occurred.

Q/K determine attention weights A = softmax(QK^T/sqrt(8) + causal mask). V supplies vectors to combine. Each head has width 8. The two heads are concatenated and projected by W_O.

For a +8 recipient, a QK patch computes A from the same example at +4, then applies it to V from +8. A V patch keeps +8 weights and uses +4 values. QKV replaces both. The recipient residual path and downstream FFN, normalization and output remain active. Test the reverse direction too. All replacements stay within one model; different models can learn different coordinates.

Margin means correct-answer logit minus the largest competing logit. Positive means the answer wins, except exact ties. A high average margin can coexist with many errors.

## What attention inspection shows

Five-seed, two-head means. Value rows are 1/3/5; their preceding key rows are 0/2/4. The last column measures final-query attention to the correct value slot in block 2. Per-head results are retained in the raw report. Head labels are not assumed to have matching roles across runs.

| Training / offset | Block-1 pair weight, row 1 | Row 3 | Row 5 | Block-2 target weight |
|---|---:|---:|---:|---:|
| Fixed +4 | .899 | .211 | .229 | .396 |
| Fixed +8 | .738 | .610 | .122 | .451 |
| Varied +4 | .852 | .666 | .639 | .834 |
| Varied +8 | .858 | .492 | .243 | .558 |

In varied models, later value rows read their paired keys less at +8, and target attention falls. These are descriptive associations. They do not prove that block 1 caused the failure.

## Controlled replacements

Mean accuracy (%). Columns identify the recipient offset. Donors use the other offset. Baseline means no replacement.

| Replacement | Fixed +8 | Varied +8 | Fixed +4 | Varied +4 |
|---|---:|---:|---:|---:|
| Baseline | 59.50 | 77.65 | 56.92 | 99.98 |
| Matched QK | 50.92 | 99.44 | 62.35 | 78.36 |
| Matched V | 60.99 | 78.29 | 54.57 | 99.74 |
| Matched QKV | 54.84 | 99.95 | 62.75 | 78.42 |
| Unrelated QK | 48.83 | 98.02 | 60.02 | 75.91 |
| Unrelated V | 30.44 | 31.09 | 29.16 | 31.48 |
| Unrelated QKV | 28.41 | 31.29 | 29.71 | 30.73 |

The reverse QK patch damages the otherwise strong varied +4 model. That strengthens evidence for an effect of the replaced routing computation. Fixed models lack a reliable +4 donor: their mean QK patch effect is negative at +8. Therefore, +4 itself is not a universally corrective source.

| Varied seed | +8 baseline | Matched QK | Matched V | Matched QKV | Unrelated QK |
|---|---:|---:|---:|---:|---:|
| 461 | 89.50 | 99.82 | 88.44 | 99.88 | 99.63 |
| 487 | 84.96 | 99.69 | 85.17 | 99.93 | 99.26 |
| 509 | 81.79 | 99.82 | 81.36 | 99.99 | 99.79 |
| 541 | 61.81 | 99.40 | 64.50 | 100.00 | 97.32 |
| 571 | 70.19 | 98.47 | 71.96 | 99.96 | 94.11 |

Across 36,000 seed/example observations, matched QK rescues 7,870 predictions and harms 25. V rescues 1,320 and harms 1,091. QKV rescues 8,035 and harms 7. These counts are dependent across mapping variants and seeds, not independent trials. Mean +8 margin is 3.35 before replacement, 6.66 with QK, 3.71 with V and 7.51 with QKV.

## Why the unrelated control matters

Donors move forward one mapping (18 rows), preserving the order/query enumeration index. All 7,200 donors consequently retain the recipient's target slot. Their answer labels match in 31.5% of cases. This is a slot-aligned control, not a control with random target positions.

Unrelated QK gives 98.02% recipient accuracy. Its routing can point to the same slot even with different content. Unrelated QKV instead gives 31.29% recipient accuracy and 98.59% donor-answer agreement. That is consistent with transferring donor information when V also moves. Neither observation proves a complete binding algorithm.

## Bounded conclusion and next test

QK replacement is a strong rescue in this setting. It shows that the remaining +8 values and downstream paths can support high accuracy when supplied these donor routes. It does not prove V is unaffected, isolate Q from K, or identify how training produced the route. Mixed activations can also be outside the model's normal operating distribution.

Follow-up completed: [donor slot versus answer controls](../attention/donor-slot-results.md). Same-slot/different-answer donors retain 97.81% recipient accuracy; different-slot donors drive about 96.6% agreement with recipient values at the donor-selected slot. The rescue transfers routing and is not evidence that the recipient independently recovered the correct relationship. Q/K localization is also complete; the [joint-routing assessment](../../assessments/joint-routing-assessment.md) closes finer diagnostic patching and defines fresh-mapping confirmation.

## Sources and verification

[Jain and Wallace (2019)](https://aclanthology.org/N19-1357/) show cases where attention distributions do not uniquely explain predictions. [Wiegreffe and Pinter (2019)](https://aclanthology.org/D19-1002/) argue that explanation claims need explicit definitions and diagnostic controls. These motivate separating attention inspection from intervention evidence; our experiment does not reproduce their benchmarks.

[Code](../../../experiments/position_routing.py), [summary results](../../../results/position_routing.json), [example-level evidence](../../../results/position_routing_details.json), [audit](../../../results/position_routing_audit.json). Original accuracy/loss match the prior report. Causal normalization, reconstructed attention, sham outputs and QKV/donor-output equivalence passed; maximum control discrepancy was 4.93e-14. All checkpoint hashes remained unchanged. The separate audit exactly reconstructs accuracy, margins, attention summaries and rescue/harm counts for 140 conditions. It verifies provenance but does not independently recompute losses or rerun interventions. Checks share PyTorch and existing helpers.
