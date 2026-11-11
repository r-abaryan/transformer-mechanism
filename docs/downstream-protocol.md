# Downstream residual investigation protocol

Fixed before execution. Use all five saved checkpoints and the same 1,024 examples from the pathway experiment. This is exploratory decomposition, not another fresh-data confirmation. Keep attention output fixed at its original position-ten computation.

Let u be the original post-attention residual and delta=PE(6)-PE(10). Define f(u)=FFN(LN2(u)), b=u+f(u), and z=W_vocab LN_final(b). Compare baseline, full residual rescue u+delta+f(u+delta), skip-only u+delta+f(u), and FFN-only u+f(u+delta). The last two recombine original and donor activations, so they may be off-distribution.

For baseline and rescue, record true-token logit minus largest competing logit (including query-token competitor). Measure paired accuracy, loss, corrected/broken predictions and margin on original errors. No intermediate output is a calibrated prediction: projections before the FFN/final norm are diagnostic margins only.

Decompose final normalization into centering, variance scale, learned feature scale gamma and bias beta. For each patched b, compare normal normalization with using the original row's mean/standard deviation, then vary mean and standard deviation separately. Preserve gamma/beta. These are diagnostic interventions, not replacement architectures. Report the direct algebraic contributions of centering and scaling to each baseline/rescue true-token versus original strongest competitor logit gap. The competitor is fixed for this decomposition so contributions add exactly.

Verify the sum of components matches logits within 1e-10 and full residual rescue matches the prior pathway implementation. Keep checkpoints fixed, log hashes, and retain every seed and intervention. No claim that an individual stage is the sole cause follows just from a rescue.
