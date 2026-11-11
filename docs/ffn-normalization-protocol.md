# Next question: final normalization after an FFN edit

Fixed before this follow-up execution. Same five frozen varied models and opened canonical three-pair inputs, +4. Focus only on removing block-2 FFN at the final token. Let u be its input residual, f its FFN output and b=u+f. Compute original final LayerNorm statistics mu_b and sigma_b=sqrt(var(b)+epsilon), population variance over 16 features.

Compare ordinary logits Wout LN(b), actual removal Wout LN(u), and a diagnostic frozen-statistics removal Wout[gamma*(u-mu_b)/sigma_b+beta]. Frozen statistics are taken from the unmodified example. This is a counterfactual diagnostic, not a proposed model.

Verify original minus frozen logits equals Wout[gamma*f/sigma_b] within 1e-10. Decompose ordinary-minus-actual-removal into this direct term plus frozen-minus-actual-removal (the statistics-change term). Report accuracy/loss of all three, RMS direct and statistics-change terms, and identity errors. Neither term alone is a context-free neuron attribution. All offsets/weights stay fixed. No thresholds for performance, no tuning. Save logits and recompute metrics. Close the bounded question after reporting the decomposition and limits.
