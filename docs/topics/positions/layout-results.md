# Fixed content, different positional layouts

**Pair-position offsets and gaps affect accuracy even when crop content is unchanged.** Original tail positions alone have a smaller effect in the compact-pair setting. These are conditional effects in frozen models, not a unique mechanism or architecture recommendation.

## Method and math

Plain explanation: the information and correct answer stayed the same. We changed only the position numbers supplied to the model. An offset moves where the pairs start. Gaps skip position numbers; they do not insert empty tokens. The tail is the marker and final query. Compact means consecutive positions 0..7. Accuracy changed even though the tokens and weights did not. The model's solution depends on layout. Moving pairs also changes their distance from the query, so these results alone do not separate absolute positions from relative distances.

The [predeclared protocol](../../layout-protocol.md) evaluates all eight offset/gap/tail combinations on the same crops used in [subset confirmation](../retrieval/subset-confirmation-results.md): 512 parent episodes per length, 3 or 10 crops each, five frozen models. Data are reused and exploratory; no training occurred.

For original retained key indices r_j, j=0,1,2, let a=r_0 and g_j=r_j-r_0-2j. Assign key position p_j=2j+A*a+G*g_j and value position p_j+1. A/G retain or remove the original pair offset/gaps. T selects original long tail positions instead of compact 6/7. Tokens, order, labels and causal mask stay fixed; only sinusoidal PE changes through the previously verified hook.

## All eight results

Mean accuracy / cross-entropy in nats across five seeds. Bits are A,G,T; 1 retains that original positional component. All-seed status concerns subset-average accuracy >=90%, not every crop being correct.

| A G T | Four-pair-derived crops | Six-pair-derived crops | Every seed >=90%, four / six |
|---|---:|---:|---|
| 000, compact | 100% / 0.0005 | 99.996% / 0.0005 | Yes / Yes |
| 001, tail only | 99.26% / 0.0328 | 97.50% / 0.1108 | Yes / Yes |
| 010, gaps only | 92.71% / 0.4106 | 92.17% / 0.4593 | Yes / Yes |
| 011, gaps + tail | 91.80% / 0.4309 | 89.00% / 0.6113 | No / No |
| 100, pair offset only | 94.73% / 0.3296 | 88.01% / 0.7040 | Yes / No |
| 101, offset + tail | 93.89% / 0.3500 | 85.90% / 0.7915 | Yes / No |
| 110, offset + gaps | 87.43% / 0.7396 | 87.66% / 0.7056 | No / No |
| 111, original positions | 86.43% / 0.7481 | 84.32% / 0.8763 | No / No |

At six-pair-derived crops, adding only offset to compact positions reduces mean accuracy by 11.99 percentage points; gaps alone by 7.82; original tail alone by 2.50. Effects depend on the other factors: adding gaps with offset present changes 88.01% to 87.66%, a much smaller difference. They cannot be treated as additive shares of one failure. Full paired rescue/harm counts for all 12 conditional comparisons and every seed are retained in the report.

| Seed | Six: compact pairs, original tail | Six: pair offset only |
|---|---:|---:|
| 137 | 98.96% | 89.47% |
| 163 | 99.04% | 87.29% |
| 191 | 99.12% | 88.07% |
| 223 | 91.43% | 86.56% |
| 257 | 98.93% | 88.65% |

## What is established and what is not

The compact-pair/original-tail result refutes the sufficiency of a tail-only account for these cropped inputs. Pair offsets and gaps have measurable conditional effects with content fixed. It does not establish whether absolute positions, pair-to-tail distances, positional collisions or their interaction drive those effects: moving pairs changes several such relationships. A is a pair-only offset, not a global shift. Crops use query-aware selection and remove original distractors; conclusions do not automatically transfer to unmodified long sequences.

Subsets share parent episodes and are dependent. No confidence interval or significance claim is made. This is a local follow-up to the [existing paper comparison](../training/depth-comparison-results.md#paper-comparison-and-remaining-question), not a new benchmark reproduction.

## Verification and next question

[Script](../../../experiments/position_layout.py), [raw report](../../../results/position_layout.json) and the [fixed input artifact](../../../results/subset_confirmation_splits.json) preserve provenance. All 000/110/111 metrics reproduced prior measurements within 1e-12. Compact sham checks passed below 1e-10; labels, finite outputs and unchanged checkpoint hashes passed. Source hashes and paired-effect arithmetic were separately checked. All per-seed cell metrics and robust-episode accuracies are recorded.

Follow-up completed: [global versus pair-only shifts](shift-results.md). Accuracy changes under global shifts despite preserved distances. Layer/path attribution remains unresolved; these are diagnostic results, not untouched confirmation.
