# Position-training routing diagnostic

Exploratory on the opened 400-mapping test set. Freeze all ten position-training checkpoints. No training, architecture edits or new confirmation claim.

Inspect offsets +4/+8 in each model. Measure block-1 attention from each value row to its preceding key, per head and row; block-2 final-query attention to the target value, per head; answer accuracy, loss and target-minus-best-other-logit margin. Attention alone is descriptive.

Within each model, replace block-2 QK, V or QKV from the same example at the other offset. Test both directions (4 into 8, 8 into 4), both heads, all rows. Recipient residual path and downstream FFN/normalization/output remain live. Use its own offset as a sham. Use donors shifted by 18 dataset rows modulo the dataset size as an unrelated-example control; this moves to another mapping while retaining the order/query enumeration index. Report donor-answer agreement and actual same-answer fraction, since unrelated donors need not have different answers.

Do not transfer tensors between independently trained models: their learned coordinates may differ. Compare within-model effects across the matched training groups instead. No threshold selected after results. Report every seed, rescue/harm counts, means and margins; no p-values treating mapping variants as independent. A donor at +4 is not assumed correct, especially in fixed models.

Verify original accuracy/loss against the saved training report, attention reconstruction and sham logits <1e-10, finite outputs, causal normalized weights, immutable checkpoint/source/split hashes. Save predictions, margins and per-example route measures for audit. QKV replacement must reproduce whole attention-output replacement from the donor. These checks share PyTorch; they are not independent replication.
