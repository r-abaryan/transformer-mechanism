# Donor-free canonical positions

Fixed before fresh input generation. Scope: eight-token, three-pair recall where a global position offset is irrelevant to the answer. Use one ordinary model forward with positions 4..11 for every input, regardless of supplied global offset. This is explicit preprocessing using the task contract, not learned invariance, activation replacement or a novel architecture. Canonical start +4 was chosen from earlier results, not from this new set.

Use all five frozen varied-position checkpoints. Exclude all mappings in reserved_cycle_splits.json, position_training_splits.json and joint_routing_confirmation_inputs.json. Generator seed 20261224; sorted enumeration per value triple, shuffle remaining mappings and take 20 per triple. Expand 400 mappings into 7,200 order/query sequences. Save inputs, manifest and hashes before predictions. Verify exclusions and balance.

Compare ordinary positions at offsets 0/4/8/12 against canonical positions at each offset. Canonicalization is allowed only after checking positions are consecutive, with the same global offset in each row; reject gapped layouts. Subtract each row's first position and add 4. Tokens and causal mask do not change. No labels or donor activations enter this adapter. The implementation uses the existing position-aware forward helper.

Predeclared endpoint: canonical accuracy >=99% for every seed, and no more than one percentage point loss relative to ordinary offset 0 for any seed. Separately require canonical logits to agree across all tested offsets within 1e-10 and to equal ordinary +4 within 1e-10. These latter properties follow from identical position IDs and verify the adapter; they are not empirical evidence of learned generalization.

Report all seed accuracies, losses and mapping correctness; ordinary +8/+12 versus canonical gains. Check ordinary explicit/library parity, finite outputs, checkpoint/provenance hashes and rejection of gapped positions. Save full logits for metric audit. One forward per prediction versus the donor method's additional donor computation; no latency or FLOP benchmark claim. No training. Stop after the five-seed evaluation; retain failures without changing the chosen start.

This can resolve global-offset sensitivity for independent sequences under this contract. It does not apply automatically to meaningful absolute positions, gapped positions, streaming caches, longer inputs or other tasks. Those remain untested.
