# Where do the remaining errors occur?

**Errors span every queried-pair rank. Repeated answers are associated with higher accuracy, but can hide incorrect retrieval.** These are descriptive patterns, not causes established by intervention.

## Short explanation

Query rank means whether the question asks about the first, second or third retained pair. It is not the final query token's position, which stays last.

Consider `A->2, B->2, C->2`. Returning 2 is correct even if the model reads the wrong pair. Compare `A->2, B->3, C->1`: now reading the wrong pair usually changes the answer. Thus higher accuracy with repeats does not by itself show better key/value binding.

## Method

The [grouping protocol](../../error-strata-protocol.md) was fixed before subgroup outcomes were computed. Reevaluate the five frozen models on the existing 5,120 confirmation crops at shifts 0/4/6/8. Keep original and later-two paired-key forcing unchanged. No training or new intervention. Preserve per-crop predictions.

Partition by rank and by value pattern, including the full rank-by-pattern table. Accuracy is correct/count within the subgroup; mean loss averages per-example cross-entropy. All seeds use the same subgroup counts. Mean seed accuracies below do not imply independent crop observations. Parent and crop overlap limitations from [confirmation](pair-confirmation-results.md#method-and-data-limits) remain.

## Results with the fixed intervention

Percent accuracy, averaged across five seeds. Zero-shift accuracy is >=99.96% in each listed group.

| Group | Crops per seed | Shift +4 | +6 | +8 |
|---|---:|---:|---:|---:|
| Query first pair | 1,778 | 94.30 | 92.20 | 83.60 |
| Query second pair | 1,696 | 93.05 | 91.16 | 86.23 |
| Query third pair | 1,646 | 89.91 | 89.55 | 88.26 |
| Three distinct values | 1,952 | 90.33 | 88.52 | 82.53 |
| Two distinct values; target appears once | 987 | 89.54 | 88.11 | 82.03 |
| Two distinct values; target appears twice | 1,910 | 95.23 | 93.95 | 90.00 |
| All three values equal | 271 | 99.26 | 98.67 | 96.68 |

Rank ordering changes with shift: third-pair accuracy is lowest at +4/+6, while first-pair accuracy is lowest at +8. There is no single consistently worst rank in these aggregates.

| Seed, +8 | Query first | Query second | Query third | Distinct values | Target twice | All equal |
|---|---:|---:|---:|---:|---:|---:|
| 137 | 91.28 | 91.98 | 98.54 | 92.57 | 95.81 | 99.63 |
| 163 | 91.96 | 95.75 | 99.21 | 94.26 | 97.23 | 97.79 |
| 191 | 90.55 | 96.05 | 96.66 | 92.78 | 95.81 | 97.79 |
| 223 | 68.45 | 70.40 | 73.03 | 63.22 | 79.84 | 96.31 |
| 257 | 75.76 | 76.95 | 73.88 | 69.83 | 81.31 | 91.88 |

The weaker seeds fail across ranks. Their difficulty is not confined to the first value position. Within each rank at +8, target-twice accuracy also exceeds distinct-value accuracy; the full cross-tabulation is saved. This reduces one simple aggregation concern but does not control all confounders.

## What wrong answers look like

Counts across 25,600 model/crop evaluations at each shift. Categories exhaust all errors; they are not independent trials.

| Shift | Wrong value supplied by another pair | Value absent from the crop | Invalid token |
|---|---:|---:|---:|
| +4 | 1,760 | 166 | 0 |
| +6 | 1,954 | 349 | 0 |
| +8 | 2,887 | 705 | 0 |

Most wrong answers are supplied values. That is compatible with selecting another pair, but prediction labels alone do not prove the model read that pair. Absent-value errors show that wrong-source selection is not a complete literal account. Even all-equal crops have errors at +8; because their only supplied value is correct, those errors cannot be explained merely by choosing a different supplied value.

## What is still unknown?

Repeated target values may make scoring easier without fixing retrieval. Distribution differences could also contribute. This analysis does not establish that repetition causes better internal routing. Subgroups share parent episodes and identities, and their sample sizes differ. No significance test or causal claim is made. The [existing attention source comparison](../attention/qkv-results.md#learn-the-operation) remains applicable; no benchmark or architecture claim is added.

## Verification and next question

[Script](../../../experiments/error_strata.py) and [raw report with predictions](../../../results/error_strata.json) retain all seeds, original/intervened outcomes, losses, cross-tabs and error counts. Group partitions cover all crops exactly once. Aggregate accuracy/loss reproduce confirmation results within 1e-10. Input/checkpoint hashes, finite outputs and causal/normalization checks passed. Source hashes and partition/error-count arithmetic were checked separately.

Follow-up completed: [matched value counterfactuals](value-counterfactual-results.md). All-equal inputs improve accuracy while average target-slot attention slightly falls; other routing proxies also change. This supports redundancy without proving a complete mediation mechanism.
