# Donor-free handling of global position offsets

For this task, resetting position IDs gives 99.98% mean accuracy without a donor or activation patch. This closes global-offset handling for independent, consecutive eight-token inputs under the stated task contract. It does not solve general transformer routing or length transfer.

## What changed and why it is allowed

The answer depends on the supplied key-value pairs, not their absolute starting position. Before one ordinary forward pass, replace position IDs p_i with p_i - p_0 + 4. Consecutive inputs then always use positions 4..11. Tokens, order and causal mask remain unchanged. No answers or donor activations are used.

For a global shift s, (p_i+s)-(p_0+s)+4 = p_i-p_0+4. Offset invariance follows from the preprocessing equation. Identical predictions verify this identity in the implementation; they are not evidence that the model learned invariance.

Canonical start +4 was chosen from prior results and fixed before the fresh test. It is a familiar position setting for these models. The adapter rejects gapped/repeated position IDs. This is task-specific input normalization, not a new architecture or an isolated QK change: the whole model receives canonical positions.

## Fresh evaluation

Five frozen varied-position models. Exclude all 5,200 mappings in the two prior training/split corpora and the joint-routing confirmation set. Generate 400 new mappings, balanced over value triples, answers and query ranks; enumerate all 18 order/query variants. Save inputs and provenance before predictions. No training, tuning or filtering. [Protocol](../../canonical-position-protocol.md).

| Supplied offset | Ordinary accuracy | Canonical accuracy | Ordinary loss | Canonical loss |
|---|---:|---:|---:|---:|
| 0 | 99.981% | 99.978% | .0020 | .0018 |
| +4 | 99.978% | 99.978% | .0018 | .0018 |
| +8 | 77.600% | 99.978% | 1.1141 | .0018 |
| +12 | 60.311% | 99.978% | 1.9351 | .0018 |

Five-seed means; loss is cross-entropy in nats. Canonical accuracy is repeated because its actual inputs are identical across these conditions. Fully correct mappings average 99.60%, compared with 5.95% for ordinary +8 and 1.00% for ordinary +12. The 7,200 sequences are 400 dependent mapping groups.

| Seed | Canonical accuracy at every tested offset |
|---|---:|
| 461 | 99.986% |
| 487 | 99.931% |
| 509 | 99.972% |
| 541 | 100% |
| 571 | 100% |

All predeclared checks pass: every seed >=99%, offset-0 accuracy loss <=1 point, and identical canonical logits across offsets and against ordinary +4. Worst loss relative to ordinary 0 is .0139 percentage point. High accuracy is not perfection; eight errors remain across 36,000 seed/example observations per canonical condition.

## What is resolved, and what is not

Resolved within scope: a donor is unnecessary when global position origin carries no task information and we may reset it. One forward per example constructs Q/K/V normally from the canonical input. The diagnostic donor pass and extra activation-patched execution are unnecessary. No latency or FLOP benchmark was run.

Not resolved: preserving meaningful absolute positions, nonconsecutive layouts, streaming caches, longer sequences, other tasks, or learning intrinsically robust routing. Resetting position IDs would change the problem in some of those settings. This experiment also cannot attribute the benefit exclusively to Q/K because every position-dependent path changes.

The earlier joint-QK results remain useful mechanism evidence. Canonicalization is a simpler scoped input-handling solution, not proof that its computation matches the patch or that joint routing is globally solved. New mappings share the same task grammar, models and implementation. No novelty claim.

Stop this global-offset cycle here. Further research should name a distinct unresolved setting and reserve its evaluation, rather than adding more patches or offset tests to this task.

## Reproducible evidence

[Adapter and experiment](../../../experiments/canonical_positions.py), [fresh inputs](../../../results/canonical_positions_inputs.json), [results and provenance](../../../results/canonical_positions.json), [full logits](../../../results/canonical_positions_scores.json), [audit](../../../results/canonical_positions_audit.json).

All 40 condition summaries reproduce exactly from saved logits. Canonical logits agree exactly across tested offsets; ordinary explicit/library discrepancy is at most 1.17e-13. Mapping exclusion, complete variants, balance, immutable checkpoint/source hashes, finite outputs and criteria passed. Adapter checks reject gaps and repeated positions and verify the normalization formula at additional integer offsets. Those extra checks exercise position IDs only, not additional model evaluations. The audit shares PyTorch and is not an independent implementation.
