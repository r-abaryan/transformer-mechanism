# Readout control for the existing exposure trace

Extension of the stage diagnostic, 2026-10-08. No training, model execution on new examples, activation replacement or new empirical study. Preserve the original trace code/protocol/tensors/report. Archive the educational report before adding this control.

Use the two saved branches at 6,500/7,000/7,500/8,000 on test6/test10. Keep the original final-outcome cohorts and control-stop anchors. The shared update-6,000 state supplies an initialized-identical-pair boundary only, not another contrast. Content competitor is the correct token. Terminal competitor is the control model's strongest final non-EOS token at each checkpoint/example, fixed for both branches and all stages. Record the difference from the intervention's own strongest terminal competitor as a separate rival-change term.

For each saved stage, score both branches through (a) that checkpoint's control classifier and (b) that checkpoint's intervention classifier. These are fixed-classifier diagnostic projections; intermediate scores are not predictions from a separately trained intermediate head. A fixed classifier removes its variation from the within-pair difference but does not eliminate learned-coordinate compatibility concerns.

Let `a = w_EOS - w_competitor`, `m = a^T h`, `delta_h = h_B - h_A` and `delta_a = a_B - a_A`. Exact symmetric identity:

`m_B - m_A = ((a_A+a_B)/2)^T delta_h + delta_a^T ((h_A+h_B)/2)`.

Call these representation and classifier terms. They split the bilinear interaction evenly, a local bookkeeping convention, not causal shares. Retain both ordered identities as sensitivity checks: `a_A^T delta_h + delta_a^T h_B` and `a_B^T delta_h + delta_a^T h_A`. Add the rival-change term to reconcile the previous native-margin difference. Do not turn signed terms into percentages when cancellation is possible.

Report all cohorts, content/terminal/control-stop slices and the short control at every paired checkpoint. Empty cohorts/slices are explicit null. Primary descriptive check, fixed endpoint only: for recovered control-stop examples, test whether the mean paired margin difference increases in magnitude from cross LayerNorm to FFN residual and from FFN residual to final LayerNorm under **each** fixed classifier. Report both booleans and per-example fractions without treating this as a causal or repair gate. No automatic training follows.

Before scored saved traces: supplied interacting vectors, identical/readout-only/state-only cases, common logit-shift invariance, unequal terminal rivals, missing terms/empty groups and deliberate numerical corruption. Exercise actual saved initialized native traces through the same scorer and scalar audit, then reload derived tensors and audit every cell. Verify protected hashes before/after; compare native differences with the original report.

Equation source inspected: [Vaswani et al. (2017), §3.4](https://arxiv.org/html/1706.03762v7#S3.SS4), linear output transformation. The symmetric/ordered decompositions, retrospective cohorts, rivals and fixed-readout checks are local algebra and diagnostics; the paper does not predict these scores.
