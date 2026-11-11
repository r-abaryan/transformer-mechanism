# FFN branch challenge

Exploratory, fixed before execution: five frozen varied models, existing canonical_positions_inputs.json, three pairs, canonical +4. Ordinary, unchanged-output sham, zero block-1 FFN, zero block-2 FFN, zero both, and separately permuted block-1/block-2 FFN outputs. All token rows affected; downstream computation remains live. One permutation from seed 20261226, shared across models, saved before evaluation; retain and report fixed points and answer overlap. No performance threshold or training.

Record accuracy, loss, margin, rescue/harm, donor-answer agreement, branch norms and ReLU active fraction. Preserve full logits. Verify source/checkpoint/input hashes, explicit/library baseline and sham agreement below 1e-10.

Math checks: reconstruct f as the sum of activated W2 columns plus bias at every token. Test the analytical derivative of the FFN with respect to its normalized input z, W2 diag(W1 z+b1>0) W1, against central directional differences on the first example's final token. Choose epsilon=min(1e-5,min absolute preactivation/(4 max absolute directional preactivation change)); assert ReLU gates do not change, compare within 1e-6. This excludes LayerNorm's derivative and ReLU kinks, explicitly.

Challenge the shortcut that the final FFN changes logits by W_out f. Compare it against the actual delta W_out[LN(u+f)-LN(u)] on the final token. Report discrepancy; do not expect equality. Preserve mean FFN-output norm and residual norm without treating magnitude as importance. A drop from zeroing does not establish global necessity, and permutation mixes input contexts. Stop at a bounded branch-level answer, then move to the next defined question without neuron fishing.
