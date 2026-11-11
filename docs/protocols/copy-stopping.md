# Copy stopping diagnostic

Frozen before evaluation. Question: does withholding early EOS recover correct content on the saved copy models?

- Reuse seeds2371/2377/2381, all eight original pools and final checkpoints. CPU float64, one thread, no training or selection. Preserve original files by hashes.
- Teacher-forced scores: per-position correct-token counts, EOS argmax counts, and mean margin `logit(EOS)-logit(correct)`. Include terminal EOS separately; its margin is identically zero.
- Intervention: forbid only EOS for the first n generated tokens, where n is the known source length. Thereafter permit every class and stop at first EOS, with the original n+3 limit. Do not force correct content or terminal EOS. Preserve ordinary decoding as the baseline.
- Record content-prefix exact counts (first n tokens equal source), complete sequence exact counts (source followed immediately by EOS), per-position generated correctness, and EOS timing. Missing positions count incorrect. Report every seed, pool and distinct/repeated group.
- Independent audit: manual model equations and separate single-example intervention loop; exact generated agreement, logit/margin tolerance1e-9. Scripted early-EOS and off-by-one suppression controls must detect incorrect controllers.
- Interpretation rule: call content recovery reliable only if every seed/group reaches95% content-prefix exact at a given length. Complete-generation recovery additionally requires95% full exact in every group. These descriptive diagnostic thresholds do not replace the original criteria. Already inspected pools are not fresh confirmation.
- Stop after this fixed evaluation and audit, regardless of scores. No retraining, decoder tuning, new seeds or architecture change. Oracle length makes this a diagnostic, not an improved ordinary-generation benchmark.
