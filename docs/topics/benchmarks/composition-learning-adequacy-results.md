# GPU graph lookup: learning-adequacy results

Executed2026-10-02. Closed at the frozen budget. Both architectures fail the99% one-step fit gate. The composition-specific comparison is therefore ineligible. Observed one-step scores improve relative to the historical runs, but reliable fitting and the cause remain unresolved.

[Learning guide and diagram](composition-learning-adequacy-guide.md) ? [Frozen protocol](../../protocols/composition-learning-adequacy.md)

## Methods

Three new pairs:2267/2273/2281. Native Windows, RTX4060 Laptop GPU,8GB, PyTorch2.11.0+cu128. Transformer:2 width64 pre-norm blocks,2 Q heads/1 KV head, ReLU128, RoPE. GRU:one hidden110 layer, embedding64. Parameters60,672/60,690, shared initial embeddings and paired sample streams. Adam0.001, FP32, TF32 off, deterministic algorithms, no AMP/dropout/clipping.

Batch1,024;3,000 updates and3,072,000 final-answer targets per run, split equally across hop counts and4/6-edge contexts. Six runs total18,432,000 sampled targets. Full train/validation curves at500/1,000/2,000/3,000; all attempts and the pretest snapshot precede final model test scoring.

The existing416-family/104,448-example dataset is reused and independently checked. These are opened diagnostic pools, not new confirmation. Family splits group cyclic symbol renamings; graphs retain cycle topology. The8-edge test changes context length and positions, not hop depth. Relative to the old recipe, capacity, batch, update count, exposure, initializer and precision change together. Historical scores are not paired controls.

## Mean results

Equal-weight means over three seeds. Each cell is **accuracy / cross-entropy loss**. Percentages use half-up rounding; decisions use unrounded values.

| Pool | Transformer1-step | GRU1-step | Transformer2-step | GRU2-step |
|---|---:|---:|---:|---:|
| train_4 | 99.43% / 0.021 | 75.90% / 0.679 | 87.24% / 0.343 | 39.34% / 1.087 |
| validation_4 | 98.84% / 0.033 | 66.95% / 0.787 | 82.62% / 0.439 | 28.47% / 1.212 |
| test_4 | 99.00% / 0.028 | 66.54% / 0.795 | 83.17% / 0.432 | 28.69% / 1.219 |
| train_6 | 93.42% / 0.156 | 45.17% / 1.335 | 45.47% / 1.283 | 27.39% / 1.586 |
| validation_6 | 91.15% / 0.216 | 35.11% / 1.495 | 33.55% / 1.590 | 20.16% / 1.710 |
| test_6 | 91.44% / 0.218 | 35.81% / 1.483 | 33.98% / 1.594 | 20.03% / 1.709 |
| test_8 | 77.85% / 0.742 | 20.21% / 2.003 | 18.72% / 2.742 | 13.42% / 2.111 |

One-step fitting remains incomplete at6 edges: training means93.42% Transformer/45.17% GRU. The Transformer nearly fits the4-edge one-step pool, but validation mean98.84% also misses99%. Two-step scores cannot yet isolate a failure to combine otherwise reliable lookups.

## Every seed and group

Forward means the second required edge occurs later in the context; reverse means earlier. These groups have equal counts. They are sequence-order controls, not different graph topologies.

| Seed | Model | Pool |1-step acc/loss |2-step acc/loss |Forward acc/loss |Reverse acc/loss |
|---|---|---|---:|---:|---:|---:|
| 2267 | Transformer | train_4 | 98.96% / 0.031 | 90.30% / 0.256 | 91.98% / 0.228 | 88.62% / 0.284 |
| 2267 | Transformer | validation_4 | 98.14% / 0.048 | 84.44% / 0.356 | 86.78% / 0.317 | 82.10% / 0.395 |
| 2267 | Transformer | test_4 | 98.47% / 0.036 | 85.32% / 0.350 | 85.48% / 0.344 | 85.16% / 0.356 |
| 2267 | Transformer | train_6 | 93.04% / 0.170 | 51.95% / 1.139 | 55.79% / 1.054 | 48.10% / 1.223 |
| 2267 | Transformer | validation_6 | 91.49% / 0.217 | 39.69% / 1.444 | 42.84% / 1.373 | 36.55% / 1.515 |
| 2267 | Transformer | test_6 | 91.88% / 0.215 | 39.43% / 1.460 | 43.23% / 1.348 | 35.63% / 1.573 |
| 2267 | Transformer | test_8 | 79.23% / 0.667 | 21.92% / 2.135 | 22.66% / 2.106 | 21.19% / 2.163 |
| 2267 | GRU | train_4 | 76.76% / 0.668 | 39.31% / 1.090 | 41.37% / 1.070 | 37.24% / 1.109 |
| 2267 | GRU | validation_4 | 68.59% / 0.764 | 29.10% / 1.208 | 30.86% / 1.186 | 27.34% / 1.229 |
| 2267 | GRU | test_4 | 67.58% / 0.771 | 30.18% / 1.213 | 32.29% / 1.189 | 28.06% / 1.237 |
| 2267 | GRU | train_6 | 45.62% / 1.329 | 26.81% / 1.590 | 28.68% / 1.563 | 24.95% / 1.618 |
| 2267 | GRU | validation_6 | 35.94% / 1.474 | 20.03% / 1.713 | 22.22% / 1.675 | 17.84% / 1.750 |
| 2267 | GRU | test_6 | 36.35% / 1.469 | 20.18% / 1.710 | 21.92% / 1.679 | 18.45% / 1.741 |
| 2267 | GRU | test_8 | 21.48% / 1.990 | 13.35% / 2.119 | 15.04% / 2.060 | 11.65% / 2.178 |
| 2273 | Transformer | train_4 | 99.58% / 0.019 | 86.52% / 0.369 | 88.75% / 0.323 | 84.28% / 0.415 |
| 2273 | Transformer | validation_4 | 98.70% / 0.032 | 81.97% / 0.468 | 85.87% / 0.398 | 78.06% / 0.538 |
| 2273 | Transformer | test_4 | 98.93% / 0.030 | 82.88% / 0.449 | 84.83% / 0.401 | 80.92% / 0.498 |
| 2273 | Transformer | train_6 | 92.98% / 0.160 | 43.33% / 1.338 | 45.81% / 1.261 | 40.84% / 1.414 |
| 2273 | Transformer | validation_6 | 90.30% / 0.222 | 30.75% / 1.667 | 35.33% / 1.544 | 26.17% / 1.790 |
| 2273 | Transformer | test_6 | 89.95% / 0.243 | 30.73% / 1.666 | 32.47% / 1.590 | 28.99% / 1.742 |
| 2273 | Transformer | test_8 | 73.37% / 0.920 | 17.06% / 3.528 | 17.55% / 3.386 | 16.57% / 3.669 |
| 2273 | GRU | train_4 | 75.76% / 0.682 | 40.02% / 1.082 | 41.39% / 1.071 | 38.66% / 1.092 |
| 2273 | GRU | validation_4 | 66.18% / 0.796 | 28.81% / 1.220 | 29.88% / 1.207 | 27.73% / 1.232 |
| 2273 | GRU | test_4 | 66.67% / 0.795 | 28.35% / 1.220 | 29.04% / 1.208 | 27.67% / 1.233 |
| 2273 | GRU | train_6 | 44.69% / 1.339 | 28.37% / 1.576 | 29.43% / 1.557 | 27.32% / 1.594 |
| 2273 | GRU | validation_6 | 34.94% / 1.500 | 19.86% / 1.708 | 19.84% / 1.690 | 19.88% / 1.725 |
| 2273 | GRU | test_6 | 35.74% / 1.484 | 19.64% / 1.716 | 20.57% / 1.695 | 18.71% / 1.737 |
| 2273 | GRU | test_8 | 19.38% / 2.006 | 13.46% / 2.120 | 14.71% / 2.075 | 12.21% / 2.165 |
| 2281 | Transformer | train_4 | 99.75% / 0.015 | 84.90% / 0.403 | 89.83% / 0.319 | 79.96% / 0.487 |
| 2281 | Transformer | validation_4 | 99.67% / 0.020 | 81.45% / 0.494 | 87.50% / 0.380 | 75.39% / 0.607 |
| 2281 | Transformer | test_4 | 99.61% / 0.018 | 81.32% / 0.497 | 86.46% / 0.407 | 76.17% / 0.587 |
| 2281 | Transformer | train_6 | 94.24% / 0.140 | 41.14% / 1.373 | 43.22% / 1.303 | 39.06% / 1.442 |
| 2281 | Transformer | validation_6 | 91.67% / 0.207 | 30.21% / 1.658 | 34.64% / 1.542 | 25.78% / 1.774 |
| 2281 | Transformer | test_6 | 92.47% / 0.195 | 31.79% / 1.656 | 33.46% / 1.591 | 30.12% / 1.721 |
| 2281 | Transformer | test_8 | 80.94% / 0.640 | 17.17% / 2.564 | 19.86% / 2.377 | 14.49% / 2.750 |
| 2281 | GRU | train_4 | 75.17% / 0.687 | 38.68% / 1.089 | 40.35% / 1.074 | 37.01% / 1.105 |
| 2281 | GRU | validation_4 | 66.08% / 0.802 | 27.51% / 1.209 | 29.56% / 1.190 | 25.46% / 1.227 |
| 2281 | GRU | test_4 | 65.36% / 0.817 | 27.54% / 1.222 | 29.49% / 1.204 | 25.59% / 1.241 |
| 2281 | GRU | train_6 | 45.21% / 1.338 | 26.97% / 1.591 | 28.18% / 1.574 | 25.76% / 1.608 |
| 2281 | GRU | validation_6 | 34.46% / 1.512 | 20.59% / 1.709 | 22.35% / 1.672 | 18.84% / 1.745 |
| 2281 | GRU | test_6 | 35.33% / 1.497 | 20.27% / 1.701 | 20.18% / 1.689 | 20.36% / 1.712 |
| 2281 | GRU | test_8 | 19.76% / 2.014 | 13.44% / 2.094 | 14.68% / 2.049 | 12.21% / 2.138 |

## Learning and the gate

Six-edge one-step validation accuracy across the fixed checkpoints. Trajectories differ by seed; the sampled curves do not identify a learned mechanism.

| Seed | Model |500 updates |1,000 |2,000 |3,000 |Final6-edge loss change |
|---|---|---:|---:|---:|---:|---:|
| 2267 | Transformer | 16.51% | 18.08% | 87.98% | 91.49% | -0.0529 |
| 2267 | GRU | 21.38% | 19.92% | 24.13% | 35.94% | -0.0546 |
| 2273 | Transformer | 16.99% | 84.85% | 89.65% | 90.30% | 0.0373 |
| 2273 | GRU | 20.31% | 20.90% | 23.50% | 34.94% | -0.0478 |
| 2281 | Transformer | 20.64% | 85.55% | 89.30% | 91.67% | 0.0087 |
| 2281 | GRU | 20.29% | 19.49% | 22.68% | 34.46% | -0.0384 |

Loss change is3,000 minus2,000, averaged over both hop counts. All three GRU losses still fall; two Transformer losses increase in this interval. Neither convergence nor an eventual ranking is established. No checkpoint was selected by validation score.

| Frozen rule | Transformer | GRU |
|---|---|---|
| One-step train+validation:every seed ?99% at4/6 |Fail |Fail |
| One-step transfer, conditional on fitting |Not qualified |Not qualified |
| Two-step fitting, descriptive ?99% in every short group |Fail |Fail |
| Two-step transfer, conditional on fitting |Not qualified |Not qualified |

The dictionary-walk oracle scores100%. A one-step-only oracle scores100% on one-step and0% on two-step queries. Returning the query scores0%; returning the final edge destination scores25%/16.67%/12.50% at4/6/8 edges. Balanced constant or uniform12-node guessing is8.33%. Baseline-like accuracy does not prove a learned shortcut.

## GPU memory and elapsed time

All18 synthetic full-step profiles complete. The largest declared candidate, batch1,024, fits the6GiB cap. Values below are MiB, not total device usage.

| Model | Profile allocated/reserved peak |Run allocated/reserved peak |Run elapsed range |
|---|---:|---:|---:|
| Transformer | 197.68/228.00 | 174.85/216.00 | 37.01 to 37.20 seconds |
| GRU | 322.88/474.00 | 277.45/460.00 | 21.06 to 21.12 seconds |

Run time includes training, scheduled train/validation evaluations and checkpoint saving; excludes preflight and final test scoring. These timings are descriptive. Three synthetic updates are a memory profile, not a sustained throughput comparison. Contexts of11/15 training tokens explain why these models use much less than8GB; available memory alone does not justify larger batches or real-world claims.

## Independent verification

All42 pools and626,688 final predictions pass independent CPU float64 attention/GRU equations and metric checks. Maximum logit error5.74e-05; per-example loss error3.24e-05, both below the frozen1e-3 limits. Three cross-precision argmax changes are retained; every threshold classification remains stable under their worst-case error bounds.

CPU output/all-parameter-gradient checks, GPU precision checks, causal prefixes, changed-future invariance, batch isolation and state repeatability pass. Unmasked-attention, per-token-reset, wrong-hop-label and altered-gate controls are detected. These checks support the tested arithmetic and data contracts, not every possible implementation detail.

All58 recorded historical reports/checkpoints and167 frozen files retain their hashes. Paired streams/initial embeddings, optimizer updates, saved logits and the independent decision checks agree.

| Seed/model/pool | Row | Target |GPU FP32 prediction |CPU float64 prediction |
|---|---:|---:|---:|---:|
|2267/GRU/test_8 |5791 |11 |11 |10 |
|2273/GRU/train_4 |7133 |1 |10 |1 |
|2273/Transformer/train_6 |26379 |10 |0 |10 |

## Interpretation and closure

**Observed:** the new Transformer recipe achieves stronger mean lookup and two-step scores than the new GRU recipe on these pools. The one-step fit gate still fails for both; two-step training fit is also incomplete. Longer-input losses coexist with short-task failures.

**Unknown:** which recipe changes caused the gains, why fitting is incomplete, the learned mechanism, converged performance, and transfer to fresh graph distributions or real workloads. A finite failed threshold is not architectural impossibility.

The original composition paper comparisons remain in the [earlier guide](recurrent-composition-guide.md#relation-to-papers). This remains an input-context cycle task, not a reproduction of those different reasoning setups. The model conventions follow the primary sources linked in the [learning guide](composition-learning-adequacy-guide.md); no new literature claim is inferred from these scores.

The study is closed. Do not extend these attempts, replace seeds or automatically sweep widths/batches. The fitting cause is unresolved, but another open-ended tuning cycle is not selected.

## Artifacts

[Report](../../../results/composition_learning_adequacy.json) ? [Memory profile and frozen configuration](../../../results/composition_learning_adequacy_profile.json) ? [Independent audit](../../../results/composition_learning_adequacy_audit.json) ? [Precision details](../../../results/composition_learning_adequacy_precision_details.json) ? [Pretest snapshot](../../../results/composition_learning_adequacy_pretest.json) ? [Producer](../../../experiments/composition_learning_adequacy.py) ? [Independent equations](../../../experiments/audit_composition_learning_adequacy.py)
