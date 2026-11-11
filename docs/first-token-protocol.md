# First-token recall: protocol fixed before training

Question: can the existing one-block decoder learn to retrieve the first symbol from a sequence, and does that behavior extend to longer sequences?

Task: symbols 0-3 followed by query token 4. Example: `[2,0,3,1,0,1,4] -> 2`. Only the final query output receives loss. The answer is visible earlier in the input, but is never supplied after the query. This is a narrow retrieval task, not full sequence copying or associative key-value recall.

Data: enumerate all 4^6 length-six symbol sequences and shuffle with seed 20260929. Use 2,048 training, 512 validation, and 512 test examples, with exact sequence disjointness checked. The unused 1,024 examples are reserved. Separately generate 512 unique length-ten examples with seed 20260930. The splits are identical across training seeds. Balance is measured, not assumed.

Model: reuse `TinyDecoder`, vocabulary 5, width 16, heads 2, FFN width 32, one block, default trainable biases and LayerNorm, sinusoidal positions, dropout zero. CPU float64 matches the verified arithmetic regime. Train with SGD, learning rate 0.1, batch 32, exactly 600 updates. Seeds: 11, 29, 47, 73, 101. No learning-rate search, early stopping, checkpoint selection or test-driven retry. Use the final weights.

Primary criterion: test accuracy at least 95% for every seed. Separate length-transfer criterion: length-ten accuracy at least 95% for every seed. These are project thresholds, not literature benchmarks. Report each seed, means and sample standard deviations; no population confidence claim from five runs. Cross-entropy includes all five logits, including the query token.

Baselines: uniform choice among four symbols (expected accuracy 25%, loss log(4)); constant symbol selected by training frequency; copy the most recent symbol; and exact first-symbol lookup (100% accuracy by task definition). Report empirical accuracy of deterministic baselines. The rule-based oracle shows this task does not require a neural model.

Record split hashes, source hashes, versions, parameter count, examples/tokens processed, per-seed training time, validation learning curves every 100 updates, final test metrics and model checkpoints. Timing is descriptive CPU training time, not a hardware benchmark; memory and FLOPs are not measured. Failure to meet a criterion is retained, not tuned away. Generalization beyond this task remains unknown even on success.
