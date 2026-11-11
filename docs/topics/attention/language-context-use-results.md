# Saved language-model context-use results

Completed 2026-10-04. Recent-context shuffling raises mean endpoint loss in every seed. Earlier removal/shuffling has a much smaller aggregate effect. This is a saved-model sensitivity result, not an attention capacity limit.

## Method

Three existing, approximately 1.07M-parameter MQA Transformer word models, seeds 2183/2203/2213. Four interventions, zero training updates. Use all 3400 complete non-overlapping 64-word validation windows, predicting one fixed next word after each window. The subset covers 1.56% of validation target positions; these perplexities are not the earlier full-validation scores. Prior fitting/evaluation evidence remains in the [language comparison](../benchmarks/recurrent-language-results.md).

Cropping keeps the recent words' original RoPE positions. Both shuffles preserve the word multiset and final input word. One fixed shuffle per window/region is shared across seeds; all 3400 windows change in each shuffle condition. Recent shuffle moves positions 32..62, leaving position 63 fixed. See the [protocol](../../protocols/language-context-use.md) and [educational guide](language-context-use-guide.md).

## Paired findings

CE is negative log probability of the correct word. Lower CE/perplexity is better. PPL change compares the same targets with each seed's original context. Rescues/harms count correct top predictions gained/lost.

| Seed | Condition | CE (nats) | Perplexity | PPL change | Rescues / harms |
|---|---|---:|---:|---:|---:|
| 2183 | Original 64 | 6.05014 | 424.17 | +0.00% | 0 / 0 |
| 2183 | Recent 32 only | 6.04734 | 422.99 | -0.28% | 24 / 15 |
| 2183 | Shuffle earlier 32 | 6.04976 | 424.01 | -0.04% | 16 / 22 |
| 2183 | Shuffle recent 31 | 6.21747 | 501.43 | +18.21% | 45 / 143 |
| 2203 | Original 64 | 5.97226 | 392.39 | +0.00% | 0 / 0 |
| 2203 | Recent 32 only | 5.97168 | 392.16 | -0.06% | 15 / 16 |
| 2203 | Shuffle earlier 32 | 5.97078 | 391.81 | -0.15% | 8 / 13 |
| 2203 | Shuffle recent 31 | 6.19647 | 491.01 | +25.13% | 73 / 193 |
| 2213 | Original 64 | 5.99407 | 401.04 | +0.00% | 0 / 0 |
| 2213 | Recent 32 only | 5.99237 | 400.36 | -0.17% | 10 / 15 |
| 2213 | Shuffle earlier 32 | 5.99445 | 401.20 | +0.04% | 10 / 11 |
| 2213 | Shuffle recent 31 | 6.24314 | 514.47 | +28.28% | 55 / 187 |

Cropping does not give a uniform benefit: 46 previously correct predictions are harmed across the three seeds, while 49 are rescued. Recent shuffle harms 523 and rescues 173. The average cropping effect does not show that earlier words are ignored or can safely be removed for every case.

Each cell has 3167 ordinary words, 172 unknown tokens and 61 row-end tokens. Positive subgroup delta means worse mean loss. All original subgroup scores and all accuracy/count metrics are in the [48-row CSV](../../../results/language_context_use/metrics.csv).

| Seed | Intervention | Ordinary delta CE | Unknown delta CE | Row-end delta CE |
|---|---|---:|---:|---:|
| 2183 | Recent 32 only | -0.00282 | -0.02616 | +0.06457 |
| 2183 | Shuffle earlier 32 | -0.00017 | -0.01543 | +0.03133 |
| 2183 | Shuffle recent 31 | +0.16626 | +0.01925 | +0.64047 |
| 2203 | Recent 32 only | +0.00052 | -0.02010 | -0.00281 |
| 2203 | Shuffle earlier 32 | -0.00086 | -0.01846 | +0.01410 |
| 2203 | Shuffle recent 31 | +0.23304 | +0.02847 | +0.31776 |
| 2213 | Recent 32 only | -0.00295 | +0.02162 | -0.00258 |
| 2213 | Shuffle earlier 32 | +0.00055 | -0.00373 | +0.00301 |
| 2213 | Shuffle recent 31 | +0.25656 | +0.00523 | +0.54801 |

For example, cropping slightly improves aggregate loss in seed 2203 while worsening ordinary-word loss. Row-end changes differ across seeds. Keep these groups visible rather than deriving a universal rule from the overall mean.

## Verification and limits

Four preflight methods passed, including actual CPU/CUDA evaluator and initialized-checkpoint audit reload, supplied-score/missing-group cases, position controls and intentional errors. Formal CUDA FP32 evaluation took 17.36 seconds, batch 8, TF32 off. Peak allocated/reserved memory was 167.56/188.00 MiB. This is one inference session, not a speed benchmark.

The independent CPU float64 audit checks all 12 cells and 40,800 predictions. Maximum token CE error is 1.205e-5; fixture logit error is 2.793e-6; no argmax disagreements. Original endpoint CE matches archived scores within 9.537e-7. All 180 locked source/data/evidence files remain unchanged. The [preflight](../../../results/language_context_use_preflight.json), [report](../../../results/language_context_use/report.json), [audit](../../../results/language_context_use/audit.json) and [lock](../../../results/language_context_use/lock.json) retain evidence. A Windows tempfile write issue was corrected before freezing. PowerShell later flagged the existing missing-NumPy warning as NativeCommandError; complete artifacts and the independent audit verify that evaluation finished. No rerun occurred.

These are three trained seeds on opened validation endpoints. Shuffling changes order, distance and contextual representations, and creates unfamiliar inputs. We do not isolate one head/layer, measure usable information under retraining, certify statistical significance or reproduce a large language model's long-context behavior.

Source comparison: [O'Connor and Andreas, 2021](https://aclanthology.org/2021.acl-long.70.pdf), section 3.3 and discussion inspected, distinguishes inference sensitivity from transformed-data training. Our procedure follows the sensitivity question. Their long-context usable-information conclusions and scores are not a prediction or reproduction of ours.

Close this diagnostic and the current attention phase. The next focus is [reinforcement learning](../rl/rl-roadmap.md), including fundamentals, applications, PPO, RLHF and named GRPO variants.
