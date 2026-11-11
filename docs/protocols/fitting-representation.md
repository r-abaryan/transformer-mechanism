# Routing versus value/decoder after direction-controlled fitting

Question: which part of the successful saved computation carries its training-error advantage? The direction study established a fitting benefit but did not identify its representation. Prior beta/key swaps used different, incompletely fitted Adam endpoints; this comparison uses the audited paired direction endpoints and their adequate-fitting controls.

Exploratory parameter inspection before this protocol found large effective-decoder norm growth and small changes in aggregate key overlap. Those observations selected the question, not a success threshold. This diagnostic reuses opened pools and is not fresh confirmation.

Frozen before hybrid evaluation, 2026-10-05:

- Seeds 3473/3479/3481, final L-BFGS and negative-gradient checkpoints from `results/fitting_direction/report.json`; no intermediate-stage selection.
- Routing donor R supplies `key.weight` and `write_logit`, including the tied queries. Decoder donor D supplies `value.weight`, `output.weight` and `output.bias`. All five parameters have exactly one donor.
- Evaluate the Cartesian 3 seeds x 2 routing donors x 2 decoder donors x 3 splits x 4 loads = **144 cells**. Each load has 768/192/384 histories and exhaustive present-key queries. Retain all accuracy/CE/maps and nonempty unique/repeated groups. Same task, zero-start writes, CPU float64, one thread.
- Native self-donor cells must reproduce saved endpoint logits within 1e-9, identical decisions and metrics. Inputs, source artifacts and RNG remain unchanged.
- Routing-only sufficiency predicts that L-BFGS routing with the negative-gradient decoder has no more **total training errors** than native L-BFGS in every seed. Decoder-only sufficiency is the symmetric prediction. Report both conjunctions and every seed regardless of failure. These are finite-pool endpoint predictions, not training-cause or paper claims.
- Report both conditional routing effects and both conditional decoder effects on equal-load CE and total errors. Report their factorial interaction; do not assign additive causal percentages.
- Save all prediction logits. Independently check labels, matrix-recurrence logits and scalar metrics. Check exact intervention scope, initialized saved runtime, tied logits/empty groups and deliberate scope/logit/label/count/decision/missing-cell defects before formal execution.
- Implementation control: the exact value-basis factorization must match the native computation. Arithmetic checks are reusable tests, not separate research findings.
- Record all saved parameter milestones: beta, off-diagonal normalized-key Gram RMS, effective-decoder norm, positive scalar projection and residual versus the common warm start. No inferred causal ranking from parameter norms.
- Stop after these 12 fixed hybrids and audit. Zero optimizer/gradient/solver calls, no new training, donor variants, seed replacements, temperature fitting or model selection. Held-out results are descriptive.

The value embeddings and classifier are kept together because their product defines the effective class map. Cross-arm hybrids still break possible co-adaptation with routing. Success or failure applies to these donor coordinates and endpoints; it does not prove necessity in every representation, identify a training trajectory cause, or demonstrate a learned repair.

Source inspected: [Yang et al. (2024), section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2) for delta writes/reads. Our scalar-beta, tied-key/query reader differs from full DeltaNet. The value-basis factorization, donor split and sufficiency predictions are local derivations/design choices. No source predicts these local scores.
