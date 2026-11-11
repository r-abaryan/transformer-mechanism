# Query versus key offset diagnostic

Exploratory on the already opened 7,200 test sequences (400 mappings). Use the five frozen varied-position models only. Within each model and the same example, replace block-2 Q only, K only, or QK from offset +4 into +8, and reverse +8 into +4. Both heads and all rows; recipient V, residual and downstream operations remain live. No training or donor selection by predictions.

Report accuracy, loss, correct-minus-best-other-logit margin, rescue/harm and per-seed results. No post-hoc pass threshold. Compare Q/K effects with joint QK and compute the descriptive accuracy interaction: joint gain minus Q-only gain minus K-only gain. Accuracy is nonlinear; this is not a mechanistic decomposition or mediation fraction.

Check unchanged checkpoint/source/split hashes, original and joint QK metrics against position-routing results, causal attention reconstruction and sham logits <1e-10. Save full logits and audit metrics from them. Changing all K does not isolate target versus distractor keys, and an offset donor is not a normal model operation. These results localize intervention sensitivity, not the complete upstream cause. Fresh mappings are still needed for confirmation.
