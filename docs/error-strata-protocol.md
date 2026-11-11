# Remaining-error stratification

Fix groups before computing their outcomes. Five frozen models, existing confirmation parents/crops; shifts 0/4/6/8; original and frozen later-two paired-key forcing only. No training, new interventions or threshold tuning. This is post-hoc descriptive analysis of known failures, not causal confirmation.

Group by queried pair rank 0/1/2 and four exhaustive value patterns: all three distinct; two values with target appearing once; two values with target appearing twice; all three equal. Also report their full 3x4 cross-tabulation, counts, accuracy, mean per-example cross-entropy and paired rescued/harmed counts. Record wrong predictions that equal a supplied non-target value versus a value absent from the crop versus an invalid token. A correct answer does not prove correct retrieval when values repeat.

Preserve predictions per crop/seed/condition. Verify group partitions cover all 5,120 crops exactly once, parent hash and checkpoints, baseline accuracy/loss reproduction (<1e-10), finite logits, causal masks and source/protocol hashes. Subgroups share parents and identities. No significance tests or claims that rank/repetition causes failure. Use all seeds; report losing and inconsistent patterns.
