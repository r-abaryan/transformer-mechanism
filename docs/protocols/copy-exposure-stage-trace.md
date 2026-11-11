# Trace the matched exposure effect

2026-10-08. No training or activation intervention. Trace the shared update-6,000 checkpoint and both branches at 6,500/7,000/7,500/8,000, using frozen test6 and test10 sources and identical correct prefixes. These teacher-forced positions include counterfactual positions after a model's generated early stop.

Freeze length-10 cohorts using saved update-8,000 complete-generation outcomes: recovered (control wrong/intervention right), persistent failure (both wrong), preserved success (both right), harmed (control right/intervention wrong). Include absent cohorts explicitly; do not redefine them at earlier checkpoints. Test6 is a separate short control. No cohort is fresh confirmation.

Capture pre-cross residual, projected cross output, cross residual sum, post-cross LayerNorm, FFN residual sum and final LayerNorm. Apply each model's own final classifier at intermediate stages as a diagnostic projection only. Content margin is EOS minus correct content; terminal margin is EOS minus that model's strongest final non-EOS class, held fixed across stages. Intermediate projections are not calibrated predictions; between-model differences also reflect changed readout weights.

Report stage-margin means and mean paired branch differences for all content positions, terminal positions, and the control's first generated premature-stop position (when present). Report cross-output norms, final correctness and positive-margin fractions. Preserve per-example/position tensors. Verify the signed stage increments telescope to the final margin; this arithmetic is not causal attribution. Do not compare head identity or infer usefulness from norm alone.

Before trained scoring, reload an initialized model and compare all stages with explicit equations, final logits with native/reference forward, and scalar classifier projections. Reject a deliberately corrupted stage. Check absent cohorts. Bind existing evidence, protocol and code hashes. Audit persisted tensors and summaries. Stop after this fixed observation; no automatic intervention, stage sweep or training follows. A candidate intervention requires a distinction beyond correlated downstream amplification.

Primary equation source: [Vaswani et al. (2017), §§3.1–3.3](https://arxiv.org/html/1706.03762v7#S3), residual connections, normalization and attention output. Applying the endpoint readout to intermediate states and selecting retrospective cohorts are local diagnostic choices, not claims from that paper.
