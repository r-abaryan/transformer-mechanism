# Learned MoE balancing results

Executed 2026-10-03 and closed. **Six actual training runs**, not numerical fixtures. [Guide](moe-balance-training-guide.md) | [Protocol](../../protocols/moe-balance-training.md) | [Source](../../../experiments/moe_balance_training.py) | [Raw report](../../../results/moe_balance_training.json) | [Group metrics](../../../results/moe_balance_training_metrics.csv).

## What was compared

Three paired seeds, 4,306 parameters, four experts. CE alone versus CE + .01 balancing loss. Identical initial weights and minibatch streams. Each run completed 3,000 Adam updates with batch 128: 384,000 supervised examples, 2,304,000 across all six. Full final train/validation/test pools contain 1,024/512/512 examples. Two equally represented sign-rule groups share the input distribution across exact-input-disjoint splits. No sequence or language task.

CPU float64, one thread, Python 3.14.7 and PyTorch 2.11.0+cu128. Full optimizer-step profile preceded the frozen protocol. GPU allocated/reserved memory: zero; CPU peak not measured. Training loops took 9.03-11.82 seconds per run, excluding process startup and final evaluation. These tiny classifiers train quickly; times are not a controlled performance comparison.

## Per-seed results

All training groups reached 100%. Validation aggregate accuracy is 99.805% for seeds 2819/2837 and 100% for 2833, in both arms; every group is at least 99.609%. All test groups are 100%. Both frozen fitting gates pass.

| Seed | Arm | Train CE | Validation CE | Test CE | Validation overflow | Test overflow |
|---|---|---:|---:|---:|---:|---:|
| 2819 | off | 0.001194 | 0.006089 | 0.001676 | 24.80% | 25.00% |
| 2819 | on | 0.002998 | 0.004996 | 0.003817 | 5.86% | 5.47% |
| 2833 | off | 0.001316 | 0.001443 | 0.001688 | 19.73% | 18.16% |
| 2833 | on | 0.003563 | 0.002404 | 0.003650 | 6.64% | 4.30% |
| 2837 | off | 0.001251 | 0.003283 | 0.001170 | 20.51% | 15.62% |
| 2837 | on | 0.002109 | 0.017777 | 0.001522 | 7.42% | 3.91% |

Off/on means auxiliary balancing disabled/enabled. Routing TV equals overflow fraction for these fixed divisible batches and unit capacity factor; these are not independent outcomes. Every pair exceeds the frozen .05 TV and .02 overflow reduction thresholds on validation and test.

**Adoption fails on quality.** At least one group in every validation/test pair exceeds the allowed 5% CE increase. No group loses more than one accuracy point. For seed 2837, validation group 0 CE rises from .001763 to .033567 while accuracy changes from 100% to 99.609%. Near-zero baseline losses make relative thresholds strict; this threshold was frozen, so it remains a failure. Small absolute losses and the routing benefit are retained alongside that decision.

## Verification

Independent label equations and 50% constant/sign-u baselines pass; oracle is 100%. Full-model initial output/gradient checks pass for every seed, and omitting probability weighting is detected. Final dense-all-expert equations independently check all 18 pools and 12,288 predictions: zero argmax changes, identical choices/accepted masks, maximum logit error 2.13e-14. Metric recomputation differs by at most 2.22e-16. This audit shares the metric reducer; a separate scalar verification accompanies the report. Checkpoints contain all 3,000 optimizer steps. Prior JSON files and executed math sources remain unchanged.

## Meaning and limits

The added objective improved learned assignment balance in every tested pair. It did not satisfy the full quality-preservation rule. This is compatible with the earlier mathematical counterexamples: a loss need not certify perfect balance to influence learning usefully.

The [Switch paper, section 2.2](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf) motivates the objective. This synthetic classifier does not reproduce its language-model results. The residual can solve parts of the task without experts; assignments alone do not establish specialization. Fixed evaluation grouping, a shared synthetic distribution, three seeds and one coefficient/budget limit scope. No convergence, speed, novelty or overall architecture ranking claim. No tuning or automatic follow-up.
