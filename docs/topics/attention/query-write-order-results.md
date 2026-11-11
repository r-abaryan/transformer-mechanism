# Later writes account for the native delta classification failures

Executed 2026-10-04. **All native delta answers are correct immediately after the queried write. Subsequent writes leave 479 of 1,152 answers wrong. Moving the queried write last recovers all 479, with zero harms.** Softmax remains 100% correct.

[Learning guide](query-write-order-guide.md) | [Frozen protocol](../../protocols/query-write-order.md) | [Source](../../../experiments/query_write_order.py) | [Raw report](../../../results/query_write_order/report.json) | [Every seed's metrics](../../../results/query_write_order/metrics.csv) | [All query-present curves](../../../results/query_write_order/curves.csv) | [Aggregates](../../../results/query_write_order/summary.json) | [Audit](../../../results/query_write_order/independent_audit.json).

## Method

Six unchanged association-load checkpoints: seeds 3323/3329/3331, softmax/delta, 615 parameters each. Use the 384-example `test_32` pool: 64 context families with six cyclic value renamings, shared across checkpoints. Every history has 32 different keys, one assignment each.

Evaluate native order and move the queried event to position 1/16/32. Keep every key/value pair, query, label and length unchanged; preserve other events' relative order. Record all prefix logits. Report accuracy/CE only after the queried key appears. Native final metrics match the archive.

CPU float64, one thread; zero training updates. The 24 cells contain 9,216 final and 294,912 prefix predictions. These reuse opened pools and checkpoints, not fresh training seeds or repair confirmation. Query-aware reordered inputs may differ from the training distribution.

## Delta results

Three-seed means. Immediate means the prefix ending at the queried write; final means all 32 events. Counts sum three seeds, 1,152 paired answers per row.

| Queried-write position | Immediate accuracy % | Final accuracy % | Final CE | Correct immediately, wrong finally |
|---|---:|---:|---:|---:|
| Native | 100.00 | 58.42 | 2.321675 | 479 |
| First: 1 | 100.00 | 29.51 | 4.219373 | 812 |
| Middle: 16 | 100.00 | 59.20 | 1.974401 | 470 |
| Last: 32 | 100.00 | 100.00 | 0.000036 | 0 |

Immediate mean CE is approximately 0.000034 to 0.000036 across variants. There are no immediate errors to rescue. An endpoint harm does not imply the trajectory declined monotonically; intermediate curves retain every measured step and count.

| Seed | Native final % | First final % | Middle final % | Last final % |
|---|---:|---:|---:|---:|
| 3323 | 61.20 | 33.59 | 65.36 | 100.00 |
| 3329 | 51.04 | 23.18 | 51.30 | 100.00 |
| 3331 | 63.02 | 31.77 | 60.94 | 100.00 |

Native post-write harms by seed: 149/188/142. All seeds have 100% immediate accuracy in every variant. Per-seed CE and full paired categories remain in the report/CSV.

## Reordering versus native final answers

| Intervention | Errors rescued | Correct answers harmed |
|---|---:|---:|
| Query first | 1 | 334 |
| Query middle | 117 | 108 |
| Query last | 479 | 0 |

These compare final answers with native final answers, not immediate answers. The middle position's small mean gain hides substantial changes in both directions.

Softmax: 100% immediate/final accuracy in every cell, final mean CE approximately 0.000279. Final logits change by at most 8.88e-15 under reordering, below the 1e-9 tolerance. Prefix logits need not be invariant because their event sets differ.

## Interpretation and limits

The native trace establishes that every observed final delta error arose after a correct queried read, while processing subsequent nonqueried writes. No extra information or training is needed to recover these answers under query-last order. The issue is retention of correct classification through these histories, rather than inability to classify the requested value immediately after its write.

This does not identify a unique offending event, prove literal information erasure, establish a hard state-capacity ceiling or explain why training learned inadequate retention. Query-last is a supplied task-aware intervention. It cannot answer every possible query by reading one unchanged final state. Earlier 100% training fit and held-out failure remain valid.

The guide connects this behavior to ordered delta corrections and softmax's permutation-invariant sums. Defining papers provide equations, not these measured scores or a reproduction of full-model results.

## Verification and stop

All four preflight methods pass, including a known suffix harm, intentionally frozen suffix, corrupted traces/scalars, labels/coverage, softmax invariance and actual initialized checkpoint evaluation/audit. The audit uses separately reconstructed order, matrix transitions, explicit exponentials and Python loss/category reductions. All 24 cells pass with maximum logit error 1.69e-14 and zero argmax disagreements. Shared data, weights and PyTorch limit independence.

All 197 protected earlier files remain unchanged. The diagnostic is complete. [Consolidated load and retention assessment](../../assessments/association-retention-assessment.md) closes this sequence without an automatic order grid or retraining.
