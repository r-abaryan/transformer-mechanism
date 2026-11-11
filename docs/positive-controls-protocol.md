# Associative recall positive controls

Fixed before execution. Keep TinyDecoder(13,16,2,32), CPU float64, SGD learning rate 0.1, no dropout, and seeds 11/29/47/73/101. Do not modify the previous associative checkpoints.

Control A, pipeline: enumerate all 32 single-pair episodes `[key,value,query,key] -> value`. Train on all 32 for 3,000 full-batch updates. Target values are balanced. This task can be solved by copying the only stored value; it does not require selecting a binding.

Control B, small-set fit: select the first eight episodes for each target value from the original three-pair training split (32 total). Train fresh models on this fixed balanced subset for 3,000 full-batch updates. Report subset fit and original held-out three-pair test accuracy separately. This can demonstrate memorization, not a learned general binding rule.

Criterion for each control: final training accuracy at least 99% for every seed (with 32 examples, effectively 100%). Keep final weights, record curves every 500 updates, losses, parameter counts, seeds, source/data/protocol/checkpoint hashes. No tuning or retry after results. Evaluate only final checkpoints; no early stopping or best-checkpoint selection.

Interpretation: A pass shows basic learning works under the tested setting. B pass rules out a blanket inability to fit this tiny set, not capacity/optimization limits on the full task. B failure does not distinguish those limits. Comparing full-batch small-set training with earlier sampled training is diagnostic, not a controlled optimizer comparison.
