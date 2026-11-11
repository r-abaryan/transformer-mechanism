# LayerNorm versus RMSNorm: frozen training comparison

Decision: does removing normalization centering improve longer-sequence recall under the accepted MQA recipe? This completes the normalization investigation after the mathematical study. Freeze this document and the runner, verifier and auditor before training or held-out evaluation.

## Matched design

Five paired seeds1511/1523/1543/1567/1597. Two arms: `ln` and `rms`. Both train from scratch on CPU float64, one thread, deterministic operations. Vocabulary19, width16, two pre-norm blocks, two query heads of width8, one shared K/V head, RoPE offset4, ReLU FFN width32, no dropout. Replace all five normalization sites (two per block plus final readout normalization).

Both normalizations learn gamma, initialize it to1, have no beta and use epsilon1e-5 explicitly. LN computes gamma*(x-mean(x))/sqrt(mean((x-mean(x))^2)+epsilon); RMS uses gamma*x/sqrt(mean(x^2)+epsilon). Both have4,464 parameters. Removing the original five LayerNorm biases avoids confounding centering with bias parameters. This baseline therefore differs from the prior4,544-parameter MQA study; historical percentages are not paired baselines.

For each seed start both arms from the identical parameter state and record its hash. Only the normalization operation differs. Verify all ten initialized models' logits/parameter gradients against explicit normalization, rotation matrices and SDPA at <=1e-10. Check causal prefix preservation and an intentionally incorrect centered evaluation of RMSNorm as a negative control before training. Initialize no inactive parameters.

## Data and exposure

Generator20270110. Exclude the union of repeated-key, GELU, FFN-width, query-exposure and MQA-training families. Four pairs:800/100/200 fresh train/validation/test families. Six pairs:200 fresh test families only. Reuse the same three-pair160/20/40 split because its finite support is exhausted. Every family has exactly one repeated key and distinct values;36 order/reversal/cyclic-value variants. Expand every context into queries for every key. Record family identities, prior hashes, labels and data checks before training.

Adam learning rate0.001, batch64,6,000 updates; alternate three-pair odd updates and four-pair even updates. No warmup, clipping, stopping, rate search or budget extension. Each seed's two arms receive identical index streams from generators seed+1000 and seed+2000. Advance both generators every update, hash all drawn indices and record actual repeated/unique exposure. Per completed run:3,456,000 input tokens and31,488,000 squared attention-position pairs; this is an exposure count, not a FLOP measurement.

Validation at updates0/1000/.../6000. Record repeated/unique accuracy and cross-entropy, mean sampled training loss per1,000-update window, finite-loss/gradient checks and loop elapsed time. Report the first sampled validation checkpoint reaching99% for both query types at three and four pairs, or none. This is a sampled learning milestone, not an exact convergence time. Loop elapsed time is descriptive and does not establish normalization speed.

Train all ten attempts before opening test scores. Use the final6,000-update checkpoint only. A nonfinite run is retained as failed. Reserve all four/six test mappings before execution; do not choose seeds, checkpoints or criteria using their outcomes.

## Adoption rule and audit

Adopt RMSNorm for this controlled recipe only if all ten runs finish with finite training and all conditions hold:

- Every RMS seed has >=99% repeated and unique accuracy at three/four pairs.
- Every RMS seed has >=90% repeated and unique accuracy at six pairs.
- Mean paired six-pair unique gain is >=1 percentage point; no unique seed loss exceeds1 point.
- Mean paired six-pair repeated loss is <=0.5 point; no repeated seed loss exceeds1 point.

Equal parameters are checked explicitly. Cross-entropy, context correctness and sampled learning curves remain descriptive and cannot substitute for a failed criterion. Failure means retain centered normalization under this comparison's rule, not universal RMSNorm inferiority.

The independent auditor checks sources, fixtures, fresh-family exclusion, identical initial parameter hashes, index streams, optimizer update counts and final checkpoint hashes. Reload all trained checkpoints and recompute all30 length evaluations with separate explicit normalization/rotation/SDPA arithmetic. Require logit/metric agreement <=1e-10. Independently derive the adoption decision and query counts. Archive raw final logits and labels plus all validation curves.

Write one concise educational results document, retain failures, and consolidate normalization findings in an assessment. Close after ten attempts and the audit. No additional seeds, epsilon/bias searches, training extension or automatic task sweep. Speed, other precision/hardware and task transfer remain outside this comparison.
