# Why lower training loss need not improve reset invariance

Executed 2026-10-03. Three saved independent-exposure checkpoints, all training pools, no new training. [Protocol](../../protocols/state-training-gradients.md) | [Source](../../../experiments/state_training_gradients.py) | [Evidence](../../../results/state_training_gradients/report.json) | [All group metrics](../../../results/state_training_gradients/metrics.csv).

**The expected training loss still has a local descent direction in every seed. In weak seed 3019, that direction improves disagreeing reset examples while worsening three other groups. It also slightly increases initial-bit probability sensitivity at length 12.** This identifies a local tradeoff, not the cause of the historical training failure.

## What a gradient tells us

CE measures how much probability the model assigns to the correct answer. For training group j, `g_j = gradient(L_j)` points toward increasing its mean CE. Moving against it locally reduces that loss.

The original sampler implies `L = sum(w_j L_j)` and `g = sum(w_j g_j)`. Each length has half the exposure. Within each length, reset-free and mixed groups each have weight 1/6; reset agreement and disagreement each have weight 1/12. Enumerating both augmentation outcomes for every base example verifies these weights. Giving all eight cells equal weight would change the objective.

We probe the unit direction `d = -g / ||g||`. First-order group change is:

`L_j(theta + h*d) - L_j(theta) ≈ h * (g_j dot d)`.

A negative slope helps that group; a positive slope hurts it. The aggregate slope is `-||g||`. A negative cosine between two group gradients means their directions oppose locally. It does not mean every combined update hurts one of them.

## Remaining signal and group tradeoffs

Each checkpoint has 4,080 parameters. All 2,560 training-support inputs are evaluated; gradients cover complete pools rather than sampled batches. Losses below use the original expected sampling weights.

| Seed | Weighted CE | Gradient norm | Negative cosine pairs, out of 28 | Groups hurt by d |
|---|---:|---:|---:|---:|
| 3011 | 0.018471 | 0.556167 | 3 | 0 |
| 3019 | 0.141932 | 2.175332 | 15 | 3 |
| 3023 | 0.021689 | 2.079267 | 4 | 0 |

These norms depend on parameter coordinates. Their magnitude is not a universal measure of learning difficulty. Seed 3023 also has a substantial gradient despite much better fitting.

| Training group | Slope, seed 3011 | Slope, seed 3019 | Slope, seed 3023 |
|---|---:|---:|---:|
| 8, reset-free | -0.221017 | -1.726056 | -4.074849 |
| 8, mixed | -0.009617 | +0.060991 | -0.060377 |
| 8, reset agree | -0.234920 | +0.126679 | -0.138219 |
| 8, reset disagree | -1.135652 | -5.447306 | -6.442657 |
| 12, reset-free | -0.821351 | -8.098543 | -0.341094 |
| 12, mixed | -0.395705 | -0.569267 | -1.697944 |
| 12, reset agree | -0.323937 | +0.546506 | -2.714151 |
| 12, reset disagree | -2.084119 | -0.664112 | -3.307647 |

Weak-seed agreement/disagreement gradients have cosine -0.420401 at length 8, but +0.132810 at length 12. The length-12 tradeoff therefore cannot be described simply as opposition between those two cells; the aggregate includes six others.

## Loss reduction and cue dependence are different

Temporarily displace parameters by `h*d`, where `h = 1e-7 * ||theta||`, about 1.7 to 1.9 millionths in parameter-vector norm. Check both signs at h and h/10, always starting from the saved weights, then restore exactly. These are numerical probes, not optimizer updates.

After SET, paired inputs have identical reset/suffix/answer but opposite initial bits. Sensitivity is their mean absolute difference in predicted probability of class 1. Smaller means more probability invariance; equal predicted classes alone do not imply equal probabilities.

| Seed | Weighted CE change at +h | Length-12 sensitivity before | Sensitivity change at +h |
|---|---:|---:|---:|
| 3011 | -1.01323e-6 | 0.0147571 | -1.93674e-6 |
| 3019 | -3.79051e-6 | 0.2138905 | +3.47981e-6 |
| 3023 | -3.90499e-6 | 0.0263108 | -3.00054e-6 |

Both radii give these same signs. All original class predictions remain unchanged across the four probes, so no accuracy repair is demonstrated. The weak seed's CE reduction with increased sensitivity is a small, verified counterexample to assuming the two metrics must improve together. It does not predict behavior after larger or repeated steps.

## Verification and relation to papers

All 24 full-group gradients and 38,400 baseline/displaced example evaluations agree with independent normalization, RoPE, causal SDPA and binary-loss equations. Maximum logit error 1.01e-13, gradient error 1.77e-14, scalar reduction error 6.04e-14; zero native/reference argmax changes. Archived baseline scores match exactly. All 24 fine-radius derivative checks pass, maximum slope error 1.08e-8. Wrong-label gradient controls and reversed-slope controls detect the deliberate errors. Saved gradients/logits support independent reductions; parameter restoration is exact and targeted prior artifacts are unchanged.

[Yu et al., NeurIPS 2020, sections 2.1–2.2](https://arxiv.org/html/2001.06782v4) define gradient conflict using negative cosine, but explicitly state conflict alone need not be detrimental. Their training intervention and curvature conditions are not tested here. Our groups are strata of one state task, not their benchmark tasks. [Kingma and Ba, Adam](https://arxiv.org/abs/1412.6980v9) use adaptive moment estimates; our expected raw-gradient direction is not the actual next minibatch Adam update.

The [transport study](state-block1-transport-results.md) explains where the cue travels. This diagnostic shows remaining training signal and a local loss/invariance tradeoff. Historical optimization, sustained improvement, attainable full fitting and long-input transfer remain unresolved. Stop here; no gradient surgery, new training or architecture change is selected automatically.
