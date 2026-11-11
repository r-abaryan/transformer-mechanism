# Can optimization changes fit state tracking?

Executed 2026-10-03. Eight fresh training attempts completed. Selection used training data only. [Frozen protocol](../../protocols/state-recipe-development.md) | [Source](../../../experiments/state_recipe_development.py) | [Raw report](../../../results/state_recipe_development/report.json) | [All group metrics](../../../results/state_recipe_development/metrics.csv).

**Warmup plus cosine decay at peak .001 improves training fit in both paired seeds, but no recipe reaches 99% in every group. No recipe is selected for confirmation.** This is useful evidence about learning adequacy in the unchanged architecture; it does not identify its learned algorithm.

## Why this phase is different

Earlier comparisons fixed a recipe and reported its capability boundary. Here we deliberately search a small declared optimization grid using training fit. The architecture, labels, supervision and examples stay fixed. All attempts remain visible. Validation/test scores are neither computed nor used to select a recipe. A successful development recipe would still need fresh confirmation.

Each seed starts all four recipes with identical tensors and uses identical minibatch indices. The scheduled arms warm up for 200 updates, then decay to 10% of the peak by step 6,000. Adam .001/.003 otherwise stays constant. The schedule changes early and late optimization together; this experiment cannot attribute its effect to warmup alone.

## Every attempt

Entries are the minimum training-group accuracy at the stated update. The gate is >=99% in every group/length in both seeds, using final weights only.

| Recipe | Seed | Worst group at 3,000 | Worst group at 6,000 | Worst final CE | Fit gate |
|---|---|---:|---:|---:|---|
| constant_001 | 2903 | 50.00% | 88.28% | 0.23232 | Fail |
| constant_001 | 2909 | 50.39% | 52.93% | 0.70804 | Fail |
| constant_003 | 2903 | 51.76% | 54.30% | 0.68172 | Fail |
| constant_003 | 2909 | 50.20% | 53.32% | 0.68906 | Fail |
| cosine_001 | 2903 | 54.69% | 93.95% | 0.17243 | Fail |
| cosine_001 | 2909 | 54.49% | 95.12% | 0.10526 | Fail |
| cosine_003 | 2903 | 53.32% | 56.05% | 0.68036 | Fail |
| cosine_003 | 2909 | 49.61% | 49.61% | 0.69734 | Fail |

The .001 schedule improves final worst-group accuracy over its paired constant control by **5.66 and 42.19 percentage points**. The .003 arms remain weak on sustained updates, with final worst-group accuracy 49.61?56.05%. A higher rate is not a reliable repair in this grid. These are two development seeds, not eight independent recipe replications.

## Where the best candidate still fails

All cells below are training accuracies for cosine_001. Counts are 128/group at eight operations and 512/group at twelve.

| Seed | No resets, 8 | Long tail, 8 | Mixed resets, 8 | No resets, 12 | Long tail, 12 | Mixed resets, 12 |
|---|---:|---:|---:|---:|---:|---:|
| 2903 | 98.44% | 99.22% | 100.00% | 95.12% | 93.95% | 98.83% |
| 2909 | 100.00% | 100.00% | 100.00% | 95.12% | 95.12% | 99.61% |

Twelve-operation sustained updates remain below threshold in both seeds. Mixed resets and shorter cases fit much better. This narrows the remaining learning problem, but does not isolate gradient conflict, representational capacity or a particular state circuit. No plateau or convergence claim follows from the final budget.

## Budget and verification

Same 4,080-parameter, two-block MQA Transformer, width 16, ReLU FFN32 and RoPE. Original 1,920 training examples, final-query CE, no intermediate labels. Each run has 6,000 updates at batch 64: 384,000 examples/targets, 3,840,000 operations and 4,608,000 input tokens. Total exposure across eight runs is 3,072,000 examples. This is twice the old per-run update budget; old seeds/scores are historical references, not paired controls.

CPU float64, one thread, Python 3.14.7, PyTorch 2.11.0+cu128. A discarded full-step profile allocated all 36 Adam state entries, 65424 tensor bytes. GPU allocated/reserved memory is zero; CPU peak is unmeasured. Training plus periodic full-training evaluations took 780.05 seconds across eight loops, excluding profile, process startup and final audit. These are execution records, not speed comparisons.

Checks: all labels match independent parity/last-reset equations; constant baselines are 50%. Ignoring FLIPs produces 816 label errors. Four initialization output/gradient cases agree with independent equations, shifted-label gradients are detected, and schedule boundaries/missing-warmup controls pass. Final independent forward checks all 48 training cells and 15,360 predictions with zero argmax changes; maximum logit error is 5.15e-14. Independent scalar accuracy/CE error is at most 2.78e-16. All eight optimizer step counts, paired initializations and replayed index streams match. Targeted historical sources/data/checkpoints stay unchanged. Artifacts are contained in [one run folder](../../../results/state_recipe_development/).

## What we now know

The paired schedule treatment materially changes learned performance while keeping the architecture fixed. That weakens an architecture-only explanation of the earlier weak scores. It does not establish that optimization is the sole cause, that this model can master the full task under some budget, or that it generalizes. Historical failures remain valid for their original recipes.

We stop the grid here and select no recipe. A useful next decision would examine the remaining errors or training gradients of these improved saved models against their paired controls, with a specific explanatory hypothesis. Increasing the grid or relaxing 99% after seeing the result would not turn this into successful confirmation. [Prior input diagnostic](state-sensitivity-results.md) supplies the earlier behavioral evidence.
