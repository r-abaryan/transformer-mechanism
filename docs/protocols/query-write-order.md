# Query-write order: frozen saved-model diagnostic

2026-10-04. Question: does an initially correct queried read become wrong after other keys are written? User authorized this diagnostic. No training or architecture change.

Use six unchanged association-load confirmation checkpoints, seeds 3323/3329/3331 and softmax/delta arms. Evaluate all 384 examples in the already-opened `test_32` pool: 64 contexts with six value renamings. Each history contains 32 unique keys with consistent assignments.

Four variants: native order, queried event moved to position 1/16/32 (one-based). Remove and insert that event, preserving all other events' relative order. Keep map, query, label, length and multiplicity fixed. Reordering is query-aware and may create unfamiliar order distributions; it is not a learned repair.

Record all 32 prefix logits, but report task quality only from the queried event onward. Before its arrival, the requested association is absent. Measure immediate and final accuracy/CE, correctness curves indexed by number of subsequent writes, immediate-to-final and native-to-reordered paired rescues/harms. Variable-count curves disclose counts; do not treat them as matched trajectories across offsets. Native final scores must match the parent archive. Do not infer loss of literal stored information solely from a wrong class.

Count: 3 seeds * 2 arms * 4 variants = 24 cells, 9,216 final predictions and 294,912 prefix predictions. CPU float64, one thread; zero updates. No accuracy adoption threshold: report every seed and all categories. Require equations within 1e-9 with zero argmax disagreements and softmax final logit differences below 1e-9 under all full-history permutations.

Four preflight methods verify event/label contracts, coverage, softmax invariance, query-present metric boundaries, a known delta suffix harm, intentional frozen-suffix/trace/scalar errors, and actual initialized saved-checkpoint evaluator/audit entry. Audit independently reconstructs event order with slices, evaluates matrix transitions/explicit exponential normalization and Python CE/category reductions. Shared weights/data/PyTorch remain common dependencies.

Hash execution sources, tests, protocol and imported helpers before evaluation. Preserve parent and repetition sources/reports/data/checkpoints. Stop after 24 cells and audit. No extra order grid, training, width search or architecture adoption follows automatically. Then consolidate the load, repetition and order evidence.

Math basis: [DeltaNet 2024, section 2.2](https://arxiv.org/html/2406.06484v2), ordered correction rule; [Vaswani et al. 2017, section 3.2.1](https://arxiv.org/html/1706.03762v7), softmax sums. Papers define operations; they do not predict local scores. [Previous repetition results](../topics/attention/history-repetition-results.md).
