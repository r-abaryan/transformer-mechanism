# Recurrent state: GRU passes the bounded comparison

Completed and documented2026-10-02. **The fixed GRU performance rule passes.** All three GRU seeds reach100% final-answer and operation-state accuracy on every evaluated training, validation and test group, through24 operations. The Transformer controls still have short-task fitting errors and weak long-input results. The separate fit-comparability check fails.

[Educational guide and gate equations](recurrent-state-guide.md) · [Frozen protocol](../../protocols/recurrent-state.md).

## Method

Three fresh seed pairs2081/2099/2111; six3,000-update runs. Transformer: original width16/two-block MQA, RoPE, ReLU FFN32 and LayerNorm;4,080 parameters. GRU: embedding width16, one hidden28 recurrent layer, bias-free two-class readout;4,032 parameters (1.18% fewer). Each GRU sequence starts with zero hidden state and processes the initial-state token, operations and QUERY.

Pairs share initial embedding values and sampled input batches. Other initial weights and hidden-state dimensions differ. Both use Adam.001, batch64, alternating8/12 operations, CPU float64, one thread, PyTorch2.11.0+cu128. Both minimize final QUERY cross-entropy plus mean operation-state cross-entropy, with coefficients1/1. No target state is fed into either model. Each run sees192,000 final targets,1,920,000 operation targets and2,304,000 input tokens.

The same previously opened state-tracking data are reused. Train/validation/test sequences remain disjoint, but this is not fresh confirmation. Whole training/validation pools are measured every500 updates. Final training/validation scores and checkpoint hashes were frozen before test scoring. No model or checkpoint selection followed those scores.

## Test results

Final-answer accuracy (%), mean across three seeds. Counts per group are64/256/512/512 examples at8/12/16/24 operations. Every GRU seed, not just its mean, scores100% in every cell. Constant prediction scores50%; the exact symbolic oracle scores100% by construction.

| Operations | Group | Transformer | GRU |
|---|---|---:|---:|
|8|No resets|93.75|100.00|
|8|Long tail|100.00|100.00|
|8|Mixed resets|98.44|100.00|
|12|No resets|72.92|100.00|
|12|Long tail|80.21|100.00|
|12|Mixed resets|96.09|100.00|
|16|No resets|49.48|100.00|
|16|Long tail|54.04|100.00|
|16|Mixed resets|96.94|100.00|
|24|No resets|50.46|100.00|
|24|Long tail|49.74|100.00|
|24|Mixed resets|94.08|100.00|

## Training fit and seed variation

The following averages weight no-reset and long-tail groups equally. Long test additionally weights16/24 equally. GRU training, validation and long-test scores are100% for each seed.

| Seed | Transformer train12 | Transformer validation12 | Transformer long test | GRU long test | Long-test gain, pp |
|---|---:|---:|---:|---:|---:|
|2081|81.25|77.73|53.03|100.00|46.97|
|2099|81.15|79.69|47.61|100.00|52.39|
|2111|72.75|67.19|52.15|100.00|47.85|

Mean Transformer training/validation accuracy at12 operations is75.00%/72.40% without resets,81.77%/77.34% for long tails, and98.31%/95.05% with mixed resets. Every GRU short training/validation cell is100%. Thus the comparison contains both fitting and longer-input differences; it cannot attribute the entire gap to extrapolation.

| Predefined check | Outcome |
|---|---|
| All six finite attempts complete |Pass|
| Every GRU short training/validation/test cell>=99%, long test cell>=90% |Pass|
| Every paired seed's hard long-input mean gain>=10pp |Pass,46.97 to52.39pp|
| Both architectures fit all short training/validation cells>=99% |Fail|

The scoped performance decision passes the first three checks. Fit comparability is a separate interpretation constraint, not a condition silently removed after seeing results. Thresholds are operational criteria, not significance tests.

## Resources and verification

Dense forward MAC estimates, with total token length`L = operations+2`, are Transformer`3616L+64L^2` and GRU`3752L`. At8/12 operations these are42,560/63,168 versus37,520/52,528 per example. These exclude nonlinearities, normalization, gate products and backward work. Equal parameter count or update budget would not imply equal computation.

Recorded training loops total126.52seconds for Transformer and56.43seconds for GRU, including periodic training/validation scoring and excluding setup, final evaluation and audit. These sequential timings are execution records, not a controlled speed benchmark.

Preflight checks99,840 prefix labels and all six models' logits/gradients, causality and batch isolation. Maximum gradient discrepancy5.00e-16. GRU chunked-state carry agrees with full execution; deliberately clearing the hidden state each token changes outputs and is detected.

The independent audit passes144 final-checkpoint cells:39,168 final and599,040 operation-state predictions, zero argmax differences. Maximum logit error1.61e-13; final-loss error9.99e-16. It uses explicit GRU gate unrolling and the separate Transformer forward. Initialization, streams, optimizer steps, counts, hashes, pretest snapshot and decision agree. These observations come from three seeds per architecture on shared examples, not hundreds of thousands of independent trials.

## Meaning and stopping boundary

This GRU package learns and transfers better on the tested binary-state task under the fixed supervision and budget. Its recurrent update matches the form of a state-update problem, which is a plausible design rationale, not a measured internal mechanism. Gates, state width, normalization, positional treatment, initialization and computation differ too; recurrence alone is not isolated.

The result does not establish arbitrary-length correctness, general language superiority, a Transformer impossibility or a novel architecture. [Cho et al.2014](https://aclanthology.org/D14-1179.pdf) supplies the recurrent/gated-state precedent; our unit uses [PyTorch2.11's different reset placement](https://docs.pytorch.org/docs/2.11/generated/torch.nn.GRU.html). Neither source's translation benchmark is reproduced. **Study closed:** no automatic longer-length tests, extra seeds or architecture sweep.

[Raw results](../../../results/recurrent_state.json), [independent audit](../../../results/recurrent_state_audit.json), [pretest snapshot](../../../results/recurrent_state_pretest.json), [full logits](../../../results/recurrent_state_scores.pt), [runner](../../../experiments/recurrent_state.py), [auditor](../../../experiments/audit_recurrent_state.py). Complete training/validation/test losses, per-label scores and curves remain in the raw report. The [original data report](../../../results/state_tracking.json) retains measured ignore-FLIP baselines.
