# Saved language-model context use

Frozen before evaluation, 2026-10-04. Question: do the three saved Transformer language models rely on earlier words, recent word order, or both? The earlier 64/128 comparison changed window resets and cannot isolate this.

Use seeds 2183, 2203 and 2213 from the completed recurrent-language study. No GRU comparison, training, recipe selection or new test split. Evaluate the already opened WikiText-2 validation stream. Every complete, non-overlapping 64-word window supplies one next-word target. Counts: floor(217645/64) = 3400 windows; 3 seeds x 4 conditions x 3400 = 40800 predictions. The remaining 45 target positions are excluded. This samples 1.56% of validation targets at fixed window endpoints; it is not full-validation perplexity.

| Condition | Input | Position control |
|---|---|---|
| Original | All 64 words | Original RoPE indices 4..67 |
| Recent only | Latest 32 words | Keep original indices 36..67 |
| Earlier shuffle | Permute first 32 words | Latest 32 and final word unchanged |
| Recent shuffle | Permute positions 32..62 | Earlier 32 and final word unchanged |

Targets never change. Use one deterministic permutation per window and region, generators 202610041 and 202610042, shared across seeds. A permutation may leave repeated words unchanged; report actual changed-window counts. No selected examples or alternative shuffle draws. Cropping rebuilds remaining representations without the prefix; this intervention does not isolate one head or one attention layer.

Producer: native CUDA FP32, TF32 off, deterministic operations, batch 8; existing chunk decoder with sliced rotation tables and fresh caches. Auditor: CPU float64 applied to saved FP32 weights, explicit normalization and rotary matrices, independent attention implementation and metric reductions. Maximum per-target CE disagreement <= 1e-4; mean/group/paired CE disagreement <= 1e-5; fixture logit disagreement <= 1e-3. Record every argmax disagreement rather than silently treating near ties as identical. Check original endpoints against archived losses and targets at flat indices 63,127,...; the same CE tolerance applies.

Retain paired token losses, predictions, targets, positions, permutations and two full-vocabulary logit fixtures per cell. Report every seed and target group: ordinary words, unknown word, row-end token. Missing groups have count zero and undefined metrics. Main metrics: mean CE, exp(CE), paired delta CE, exp(delta CE), improved/worsened/equal-loss counts and correct-answer rescues/harms. No significance test, architecture winner or guaranteed control effect is declared. Three seeds are three trained models, not 40800 independent experiments.

Before formal evaluation, exercise labels, actual evaluator and audit entry on supplied logits and an initialized saved checkpoint, including missing groups, both shortened and shuffled inputs, and intentional target/metric errors. Freeze new executed sources, tests and protocol together with the referenced parent sources, stream, reports and checkpoints. Preserve existing evidence. Stop after the four conditions and audit; no retraining or automatic extension. The next project phase is RL.

Source inspected: [O'Connor and Andreas, 2021](https://aclanthology.org/2021.acl-long.70.pdf), section 3.3 and discussion. Their distinction between inference sensitivity and usable information motivates our interpretation. We perturb saved models only; we do not reproduce their retraining experiments or infer their scores for our models.
