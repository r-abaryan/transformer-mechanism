# Orthogonal keys recover all queries, with twice the state storage

Executed 2026-10-04. **Replacing saved delta key geometry with orthogonal directions gives 100% accuracy for every query in every evaluated history. Zero-padding the original keys changes no logits.** No reordering or retraining occurs.

[Learning guide](key-geometry-retention-guide.md) | [Frozen protocol](../../protocols/key-geometry-retention.md) | [Source](../../../experiments/key_geometry_retention.py) | [Report](../../../results/key_geometry_retention/report.json) | [Every seed](../../../results/key_geometry_retention/metrics.csv) | [Aggregates](../../../results/key_geometry_retention/summary.json) | [Audit](../../../results/key_geometry_retention/independent_audit.json).

## Method

Three unchanged delta checkpoints, seeds 3323/3329/3331. Use the original 384 `test_32` histories, each containing all 32 known keys once. Prefill each history once and answer all 32 keys from that same final state. No query participates in prefill; no event moves.

Compare normalized learned keys (16 dimensions), the same keys padded with zeros (32 dimensions), and fixed identity keys (32 dimensions). Transform write and query keys together. Keep learned values, classifier and beta unchanged. The isolated evaluator permits variable state width because the archived Reader fixes it at 16; its native outputs must agree with the original implementation.

CPU float64, one thread; zero training updates. Nine seed/geometry cells, 12,288 predictions each, 110,592 total. Queries and six value renamings share 64 underlying context families; these are correlated evaluations, not independent experiments. The pools were already opened.

## Results

Three-seed means. A complete map passes only when all 32 queries are correct from its final state. Counts pool three checkpoints on the same histories.

| Geometry | All-query accuracy % | CE | Complete maps / evaluations | State payload per history |
|---|---:|---:|---:|---:|
| Native, 16 dimensions | 62.47 | 2.014674 | 0 / 1,152 | 1 KiB |
| Padded, 32 dimensions | 62.47 | 2.014674 | 0 / 1,152 | 2 KiB |
| Orthogonal, 32 dimensions | 100.00 | 0.000034 | 1,152 / 1,152 | 2 KiB |

The earlier 58.42% native result graded one selected query per history. This 62.47% grades all 32 queries. The original selected-query metrics still match the archive; the evaluation population changed, not the saved model.

| Seed | Native all-query % | Padded all-query % | Orthogonal all-query % | Orthogonal complete maps |
|---|---:|---:|---:|---:|
| 3323 | 62.87 | 62.87 | 100.00 | 384 / 384 |
| 3329 | 61.56 | 61.56 | 100.00 | 384 / 384 |
| 3331 | 62.98 | 62.98 | 100.00 | 384 / 384 |

Orthogonal geometry rescues **13,836** native wrong answers, with zero harms. Padding produces zero rescues/harms and exactly zero measured logit change. Orthogonal reads equal `beta*v` with zero measured maximum error. Learned beta values are approximately 0.9092/0.9412/0.9535; no beta=1 override is needed. Per-seed CE remains in the CSV.

Payload counts cover the float64 state tensor only: 1,024/2,048 bytes. They are not measured GPU allocator peaks, total model storage or speed results.

## What this establishes

Changing key geometry is sufficient to recover every tested association while preserving the learned values and classifier, in the expanded state. The padded control rules out a benefit from this particular width expansion when geometry stays unchanged. Together with the native [post-write damage](query-write-order-results.md), this supports cross-key interference as a removable mechanism in these saved readers.

It does not show that training can learn this geometry, that width 16 cannot classify the task, or that geometric overlap alone explains how the original solution was learned. Orthogonal keys impose dedicated slots for 32 known symbols and double the state. This is a supplied representation intervention, not a trained architecture comparison or arbitrary-vocabulary solution.

The [DeltaNet equation](https://arxiv.org/html/2406.06484v2), section 2.2, already implies orthogonal preservation. Earlier local arithmetic verified that property. The new evidence is simultaneous complete-map recovery using saved learned decoding. No defining-paper quality or speed benchmark is reproduced.

## Verification and stop

Four preflight methods pass: geometry/labels/coverage, padding inverse error, one-bad-query complete-map control, corrupted logits/scalars/counts, actual initialized saved reload/audit, original Reader agreement and orthogonal reversal/direct-read controls.

All nine cells pass independent matrix equations and Python loss, complete-map and paired-count reductions. Maximum logit error is 1.42e-14; zero argmax disagreements. The direct read identity is independently checked too. Data, weights and PyTorch are shared dependencies.

All 210 protected earlier files remain unchanged. Both frozen criteria pass: padded-logit invariance and all-seed orthogonal complete-map correctness. Close this diagnostic. Learned retention at the original budget and transfer beyond fixed known symbols remain open; no automatic geometry grid or retraining follows.
