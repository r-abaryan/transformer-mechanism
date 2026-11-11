# Associative recall protocol

Fixed before execution. Question: can the unchanged one-block decoder retrieve a value using a queried key rather than a fixed position?

Encoding: keys 0-7, values 8-11, query marker 12. An episode `[k1,v1,k2,v2,k3,v3,12,q]` predicts the value associated with q. Keys are unique per episode, values are independent uniform draws (duplicates allowed), pair order is random and q is sampled uniformly from the keys. Only the final query-key output receives loss. No padding; no answer appended. Query marker distinguishes the final query from stored keys.

Generate 8,192 train, 1,024 validation and 1,024 test episodes with three pairs; generate 1,024 episodes each with four and six pairs. Use Python RNG seeds 20261040-20261044 respectively. Canonical identity is the sorted mapping plus query, so reordered copies of the same query/mapping cannot cross same-size splits. Rejection sampling removes duplicates and prior canonical identities. These are unseen episodes, not held-out individual key/value pairs. Record split hashes and target/query-position counts.

Model: existing TinyDecoder, one block, width 16, two heads, FFN width 32, pre-norm, sinusoidal positions, dropout zero. Vocabulary changes from 5 to 13; parameters otherwise retain the same architecture and default initialization. CPU float64. Seeds 11/29/47/73/101. SGD learning rate 0.1, batch 64, exactly 3,000 updates. Validation every 500 updates; final checkpoint only. No tuning, early stopping or test-driven retry.

Primary learning criterion: at least 90% three-pair test accuracy for every seed. Transfer criterion: at least 90% for every seed at both four and six pairs. Compare uniform value guessing (expected 25%, loss log(4)), training-majority value, first stored value, last stored value and exact dictionary lookup (100%). Loss covers all 13 output classes, including invalid key/marker predictions.

Report every seed, accuracy, cross-entropy, mean/sample SD, train metrics, and loss curves. Save checkpoints, data/protocol/source hashes, versions, parameters, examples/tokens processed and descriptive training times. Failure is a result, not permission to alter the recipe. Synthetic results do not establish language understanding or architectural superiority.
