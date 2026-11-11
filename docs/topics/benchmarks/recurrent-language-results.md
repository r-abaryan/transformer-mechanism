# Language prediction: Transformer leads within the fixed budget

Executed and closed2026-10-02. The Transformer has **33.60% lower geometric mean validation perplexity**,396.64 versus597.34, with improvement in every pair. All six models beat the unigram baseline. This is a fixed-budget comparison; every validation loss was still falling during the final1,000 updates.

[Educational guide and diagram](recurrent-language-guide.md) · [Frozen protocol](../../protocols/recurrent-language.md).

## Method

Three fresh seed pairs2183/2203/2213; six6,000-update runs on the native Windows RTX4060. FP32, Adam0.001, batch8/context64, no TF32/AMP/dropout/clipping. Every next-word position is supervised:3,072,000 targets per run. Identical sampled windows and initial embedding/output values within each pair. Other native initializations differ.

Two-block MQA Transformer: width16, RoPE, LayerNorm, ReLU32;1,068,832 parameters. Two-layer hidden16 GRU with zero initial state per window and bias-free untied readout:1,068,160 parameters. Total difference0.063%; processing cores3,936 versus3,264, a17.07% difference hidden by the large vocabulary matrices. This GRU differs from the single-layer GRUs used in the earlier tasks.

Reuse audited word-level WikiText-2,33,278 vocabulary entries, train-only vocabulary and fixed original splits. Training fit uses a predeclared65,536-target subset spread across the stream, about3.14% of its target positions. Full validation217,645 and test245,568 targets are scored once per context setting, including unknown/row-end words. Context resets each window; no GRU state carry across windows. Final training/validation scores and checkpoint hashes were saved before test scoring. Previously opened data, no checkpoint selection or extra training.

## Results

Losses are means across three seeds; perplexities are exp(mean loss), hence geometric means. Lower is better. Training-subset and held-out token distributions differ; the unigram also scores better on validation/test, so these absolute cross-pool differences do not diagnose fitting by themselves.

| Pool | Targets per model | Transformer loss / PPL | GRU loss / PPL | Unigram PPL |
|---|---:|---:|---:|---:|
| Training subset64 |65,536|6.0796 /436.85|6.5688 /712.53|1159.98|
| Validation64 |217,645|5.9830 /396.64|6.3925 /597.34|967.62|
| Test64 |245,568|5.9086 /368.19|6.3354 /564.22|914.39|
| Test128 |245,568|5.9061 /367.26|6.3346 /563.76|914.39|

Per-seed perplexity preserves the variation:

| Seed | Model | Training subset | Validation64 | Test64 | Test128 |
|---|---|---:|---:|---:|---:|
|2183|Transformer|446.95|407.30|377.78|377.35|
|2183|GRU|700.07|590.29|559.89|559.42|
|2203|Transformer|431.48|389.93|362.64|361.12|
|2203|GRU|711.33|595.44|561.94|561.52|
|2213|Transformer|432.30|392.91|364.33|363.51|
|2213|GRU|726.44|606.41|570.88|570.40|

Validation GRU/Transformer PPL ratios1.4493,1.5271 and1.5434. The declared2% mean improvement/every-pair rule passes, as does the at-least0.05-nat improvement over unigram on training-subset/validation/test64. Test64 favors the Transformer by34.74%, descriptively. These percentages are perplexity reductions, not accuracy-point gains.

Validation determines the winner's direction and margin. Test64 also participates in the frozen baseline eligibility guard; its model-to-model ranking remains descriptive. Neither test context selected checkpoints or triggered tuning.

Mean validation top1 accuracy is16.89% Transformer versus13.45% GRU. This measures whether the highest-scoring word matches the corpus target; it is separate from perplexity and does not measure broader reasoning.

## Groups and longer context

Validation geometric mean perplexity, same targets for both architectures:

| Group | Targets | Transformer | GRU |
|---|---:|---:|---:|
| Early positions0..7 |27,208|409.39|592.46|
| Middle8..31 |81,624|398.30|601.22|
| Late32..63 |108,813|392.28|595.67|
| Unknown word |11,718|38.84|37.72|
| Row ending |3,759|4.20|5.88|
| Ordinary word |202,168|493.88|763.97|

The aggregate Transformer advantage includes ordinary-word improvement. The GRU has better mean unknown-word PPL, so improvement is not uniform across groups. Full per-seed/group losses, PPL and top1 accuracy are in the [raw report](../../../results/recurrent_language.json).

Context128 lowers test PPL only0.25% for Transformer and0.08% for GRU. It reuses the same words with fewer resets and different available prefixes/position exposure. This is a descriptive context-window check, not independent replication or isolated length-transfer evidence.

## Verification

All24 final-checkpoint pools and4,645,902 token losses pass the independent CPU-double audit. Maximum token-loss error4.54e-5 nats, below1e-4; checkpoint/score/source hashes, paired initial values, sampled windows, optimizer steps, targets and group metrics match. Six preflights also verify forward/gradient agreement, causal prefixes and batch isolation, and detect unshifted labels and the intentional reset-at-every-token error.

There are four GPU-FP32/CPU-double top1 differences: seed2203 GRU has one each in validation/test64/test128; seed2213 Transformer has one in test64. Their exact causes were not investigated. Loss agreement passes without claiming identical predictions across precision. Three training seeds per architecture, reused words and repeated context settings are not4.65 million independent experiments.

A [separate decision check](../../../results/recurrent_language_decision_verification.json) independently derives the rule and bounds the audited validation ratio between1.506006923 and1.506006962, preserving the decision. Deliberately reporting the opposite winner is detected.

[Full audit](../../../results/recurrent_language_audit.json) · [Pretest snapshot](../../../results/recurrent_language_pretest.json) · [Training source](../../../experiments/recurrent_language.py) · [Audit source](../../../experiments/audit_recurrent_language.py) · [Decision-check source](../../../experiments/verify_recurrent_language_decision.py).

## Interpretation

Both recipes improve next-word probabilities beyond the fixed unigram baseline. The Transformer package learns more effectively within this6,000-update budget. The cause of the gap and the eventual converged ranking remain open. Processing-core size, nonlinearities, normalization and initial feature distributions differ; this does not isolate attention as the cause.

[Merity et al., section5.4/Table3](https://arxiv.org/html/1609.07843v1) evaluated20-21M-parameter two-layer LSTMs with hidden width650. Our roughly1.07M models, width16, fixed resets and budget differ substantially. No published quality result is reproduced. The [guide's primary sources](recurrent-language-guide.md) support the mechanism definitions, not a predicted winner.

State tracking favored the earlier GRU package; retrieval and this language experiment favor the tested Transformer packages. Different tasks, supervision, model dimensions and budgets prevent combining these into an overall architecture ranking. Broader reasoning and other language regimes remain untested. Study closed after the six attempts and audits; no tuning or automatic extension.
