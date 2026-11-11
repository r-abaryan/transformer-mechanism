# Key geometry and simultaneous retention: frozen diagnostic

2026-10-04. Question: can removing cross-key overlap preserve every association in one final state, without query-aware reordering? User authorized pursuing retention. Saved-model diagnostic, no training or learned architecture comparison.

Use three saved delta association-load confirmation checkpoints (3323/3329/3331), original `test_32` histories (384 examples, 64 families with six value renamings). Keep original order, events, value embeddings, classifier and scalar beta fixed. Three global, query-independent geometries: original normalized 32x16 key table; zero-pad it to 32x32; replace it by identity keys 32x32. Transform both write keys and queries consistently. No labels enter the transformation. Orthogonal keys are imposed for known symbols, not learned from this test.

Prefill each history once, then query all 32 present keys from the same final matrix. Labels are independently recovered from the event map. Report per-seed all-query accuracy/CE, complete-map accuracy (all 32 correct), rescues/harms against native all-query predictions, and state bytes. Original selected-query native predictions must match the saved report.

Count: 3 seeds * 3 geometries = 9 cells; 384 * 32 = 12,288 predictions each, 110,592 total. Queries/renamings share histories; they are not independent seeds. CPU float64, one thread, zero updates. Pools are opened; results are explanatory interventions, not fresh confirmation.

Native state 16x8 float64 is 1,024 bytes/history; padded and orthogonal state 32x8 is 2,048 bytes. This is tensor payload, not measured GPU allocator or runtime. Orthogonality removes overlap while expanding the original state. The padded geometry controls width without changing dot products. Their same-width comparison does not prove width 16 cannot solve this six-class task.

Require independent equations within 1e-9 and zero argmax disagreements. Padding must preserve all native logits within 1e-9. For identity keys written once from zero, verify reads `beta*v` within 1e-9. Report whether all orthogonal complete maps are correct in every seed, without assuming the learned value/classifier decoding passes. No adoption threshold or geometry selection.

Four preflight methods: unit/Gram/padding contracts with a deliberately nonzero padding error; all-query labels and one-bad-query complete-map control; actual initialized saved-checkpoint audit with wrong scalar/map count/logit controls; original Reader agreement, direct identity and reversal invariance. The isolated evaluator uses dimension-aware delta allocation because the archived Reader fixes width 16. Audit uses independent matrix transitions, explicit scalar CE and Python complete-map/paired counts. Shared data, weights and PyTorch remain dependencies.

Freeze sources/tests/protocol/import hashes, preserve earlier reports/checkpoints and stop after nine cells/audit. No retraining, arbitrary-symbol claim, optimized-kernel benchmark or automatic geometry grid. [DeltaNet 2024, section 2.2](https://arxiv.org/html/2406.06484v2) defines the recurrence; orthogonal preservation/direct reads follow by local derivation. [Prior query-order evidence](../topics/attention/query-write-order-results.md).
