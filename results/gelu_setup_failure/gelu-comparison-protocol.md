# ReLU versus GELU: frozen protocol

Question: with RoPE fixed, does changing the FFN activation improve latest-value recall under the same budget? This is a new training comparison, not an activation-patching experiment.

## Fixed task and data

Use the repeated-key rule, vocabulary, families and 36-example expansion from the [previous protocol](repeated-key-protocol.md). Only the single duplicated key is queried; distinct values make its earlier value wrong. This cannot establish query-token dependence or unrestricted recall.

New data seed20270105. Exclude every family in `results/repeated_key_data.json`, including its train/validation/test splits; record that artifact's hash. Split new families: for each of3/4 pairs,800 train/100 validation/200 test; for6 pairs,200 test only. Keep full families disjoint across new splits and prior data, modulo order and cyclic value relabeling. Smaller fragments and other relabelings may remain familiar. Same exact data in both arms.

## Fixed models and training

Both arms use standard RoPE at offset+4, two pre-norm blocks, width16, two heads of width8, FFN32, vocabulary19, dropout0, final LayerNorm,5,088 parameters. Change only both FFN activations: ReLU versus exact GELU (`approximate='none'`), with no added gate/projection. Reuse positional attention code. Pair every learned initial tensor.

Seeds1103/1123/1151/1171/1193. CPU float64, one thread, deterministic algorithms. Adam lr.001, betas.9/.999, eps1e-8; no decay, clipping or warmup. Batch64,6,000 updates. Alternate3-pair odd/4-pair even; advance both index generators every update with seed+1000/+2000. Equal3,456,000 tokens per run; GELU changes arithmetic, so not equal FLOPs. Record elapsed loop time without a speed claim.

Save latest resumable model/Adam/RNG checkpoints and validation curves at0/1000/.../6000. Use final weights only. Evaluate tests after all ten attempts. Retain nonfinite failures without replacement. No early stopping, learning-rate search or extra training.

## Verify and decide

Before training, check exact GELU against `x*(1+erf(x/sqrt(2)))/2` and its derivative `Phi(x)+x*phi(x)` within1e-12; ReLU derivative away from zero, with library convention0 at zero. Match ReLU logits/gradients to the unchanged RoPE implementation within1e-10/1e-9. Check one GELU model parameter by finite differences within1e-6. Confirm switching activation actually changes the synthetic model output. Audit labels, reversals and family exclusions before training.

After training, independently reconstruct matrix rotations, SDPA and GELU's erf expression; verify30 saved-logit evaluations and metrics within1e-10. Audit hashes, checkpoints, initialization and regenerated streams. Report per-seed accuracy/CE at3/4/6, six-pair family/reversal correctness, obsolete-value rate and latest-position accuracy. Validation curves are descriptive, not an alternate checkpoint selection rule.

Keep GELU for this recipe only if all five seeds reach six-pair >=90%, mean paired gain >=2 percentage points, no six-pair seed worsens, every GELU seed reaches >=99% at both3/4 pairs, and mean loss relative to ReLU is <=.5 points at each short length. The2-point minimum is chosen before evaluation because the prior RoPE mean already exceeded93%; it is an engineering threshold, not a significance test. Otherwise retain ReLU for this recipe and document partial gains or losses.

Stop after ten attempts and audit. No alternatives, new seeds or tuning based on the opened test. Failure does not imply GELU is generally inferior. See the [learning guide](ffn-activation-guide.md) for the math and primary source.
