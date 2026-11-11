# First-moment removal changes a bad step, but fails the joint repair rules

Executed 2026-10-04. **The reset reverses seed 3463's immediate error-loss deterioration. After four updates, both seeds have lower error CE, but one loses training accuracy. Neither all-seed benefit rule passes.** Both final test-32 accuracies lose slightly.

[Short guide](memory-first-moment-reset-guide.md) | [Frozen protocol](../../protocols/memory-first-moment-reset.md) | [Training source](../../../experiments/memory_first_moment_reset.py) | [Full report](../../../results/memory_first_moment_reset/report.json) | [All groups](../../../results/memory_first_moment_reset/metrics.csv) | [Independent audit](../../../results/memory_first_moment_reset/audit.json) | [Audit recovery](../../../results/memory_first_moment_reset/audit_recovery.json).

## Method and budget

Two reused seeds, 3461/3463, from the already audited before-step-11004 states. Each pair has unchanged control and a one-time memory exp_avg reset. Preserve exp_avg_sq, counters, classifier moments, model, objective and history/query RNG states. Run exactly steps 11004–11007, loads 32/2/8/16, with batch 64 and every key supervised. No held-out selection.

Formal execution makes **16 optimizer calls**: eight control-reproduction updates and eight counterfactual updates. Nothing runs past the historical 12000-step budget. Per arm: 256 sampled histories, 8192 writes, 3712 answer targets; total 1024/32768/14848. These are optimizer updates, not an inference-only diagnostic.

Baseline/final evaluation covers every train/validation/test pool and load; immediate evaluation covers train 32. All 76 cells, 566,016 answers and complete-map metrics are saved. Original pools have stable values per key; load 32 is unique, other loads repeat. Error membership is fixed at the common start, so arms score identical cohorts: 287/378 errors for seeds 3461/3463.

CPU float64, one thread. A complete synthetic preflight step with 64 histories and 2048 targets took 0.017427 seconds in one measurement. Model/optimizer tensor payloads were 4920/9860 bytes; these are not measured peak CPU memory or a throughput benchmark. No GPU use; allocated/reserved GPU memory is zero. Hardware capacity was not the limiting scope.

## Immediate and four-update outcomes

Reset-minus-control differences. Negative CE means reset is better on the declared cohort/objective. Accuracy differences are percentage points.

| Seed | Immediate error CE difference | Four-update error CE difference | Four-update equal-load training CE difference | Train-32 accuracy difference | Immediate rule | Four-update rule |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 3461 | +0.000197661 | -0.000162496 | -0.000046842 | +0.048828 | Fail | Pass |
| 3463 | -0.002126436 | -0.007291908 | -0.000055002 | -0.012207 | Pass | Fail |

The immediate rule requires lower fixed-error CE in both seeds. The four-update rule also requires lower equal-load training CE and no train-32 accuracy loss in both seeds. Both joint rules fail. No effect is rounded away or treated as a new selection opportunity.

| Seed | Starting error CE | Immediate control | Immediate reset | Final control | Final reset |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3461 | 1.656940058 | 1.649732024 | 1.649929685 | 1.636092363 | 1.635929867 |
| 3463 | 1.601535396 | 1.602450338 | 1.600323903 | 1.607171783 | 1.599879875 |

Seed 3463's immediate reset changes worsening to improvement relative to the start. Identical starting weights/gradients/streams and scoped optimizer changes support a causal claim **for this first-step intervention**. Seed 3461 shows why this is not uniformly beneficial. Later outcomes include divergent gradients and second moments.

## Final accuracy and losing cases

Each entry is control → reset, percent. Full CE and every group are in the CSV; absent groups are count zero/null. Only two training seeds are independent runs.

| Split | Load | Seed 3461 % | Seed 3463 % |
| --- | ---: | ---: | ---: |
| Train | 2 | 100.0000 → 100.0000 | 100.0000 → 100.0000 |
| Train | 8 | 100.0000 → 100.0000 | 100.0000 → 100.0000 |
| Train | 16 | 100.0000 → 100.0000 | 100.0000 → 100.0000 |
| Train | 32 | 98.8200 → 98.8688 | 98.5067 → 98.4945 |
| Validation | 2 | 100.0000 → 100.0000 | 100.0000 → 100.0000 |
| Validation | 8 | 100.0000 → 100.0000 | 100.0000 → 100.0000 |
| Validation | 16 | 99.3164 → 99.4141 | 98.7305 → 98.6654 |
| Validation | 32 | 88.5742 → 88.6068 | 86.1979 → 86.2467 |
| Test | 2 | 100.0000 → 100.0000 | 100.0000 → 100.0000 |
| Test | 8 | 99.6094 → 99.6094 | 100.0000 → 100.0000 |
| Test | 16 | 98.6654 → 98.6654 | 97.3633 → 97.3633 |
| Test | 32 | 86.4990 → 86.4746 | 87.3372 → 87.2965 |

Final train-32 reset gives 12 more correct answers in seed 3461 and three fewer in seed 3463. Test 32 loses three/five correct answers respectively, although CE falls from 0.888850684 to 0.887919930 and 0.817767246 to 0.817301926. Seed 3463 validation-16 accuracy also loses; seed 3461 validation-32 and test-16 CE worsen. These reused held-out pools are descriptive, not selection or fresh transfer confirmation. No arm meets the exact 99% full-training gate.

| Seed/arm | Starting errors rescued after four updates | Starting correct answers harmed | Final training complete maps / 768 |
| --- | ---: | ---: | ---: |
| 3461 control | 12 | 15 | 532 |
| 3461 reset | 10 | 1 | 542 |
| 3463 control | 19 | 8 | 474 |
| 3463 reset | 19 | 11 | 474 |

Seed 3461's accuracy gain comes from fewer new harms despite fewer rescues. A lower conditional loss on old errors is not complete retention or a solved task.

## Verification and recorded audit defect

Two preflight methods exercise actual saved training/evaluator/audit paths, both arms, scalar Adam, labels/streams, tied logits and absent groups. The synthetic fixture performs 12 optimizer calls, separate from the formal 16. It detects altered second moments, gradients, parameters, labels and RNG draws.

The original formal audit stopped on a **list-comparison implementation defect**, after reaching the decision summary. Its scalar/dictionary comparator did not support the per-seed criteria list. Preserve [the failure record](../../../results/memory_first_moment_reset/audit_failure.json) and original executed sources. A [separately frozen recovery](../../protocols/memory-first-moment-audit-recovery.md) adds exact list length/order checks, exercises inverse criteria defects and the saved fixture, then runs the complete original audit entry. Recovery preflight performs another 12 synthetic fixture updates; formal training is not rerun and no tolerance, metric or decision changes.

Recovered audit passes all 16 formal updates and 76 cells. Controls match the archived first full state exactly and every original beta log over the four steps. Independent matrix gradients/scores, scalar Adam, reset scope, paired streams, fixed cohorts and independently computed criteria pass. Maximum errors: evaluation logits 8.526513e-14, training logits 5.684342e-14, gradients 5.278005e-15, scalar Adam 1.776357e-15. All 197 original protected files plus recovery-bound files remain unchanged.

Sources inspected: [Kingma and Ba, Algorithm 1](https://arxiv.org/pdf/1412.6980) and [PyTorch 2.11 Adam](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html), moment/update equations. Local outcomes are not source-predicted scores. Targeted starting states and short horizon limit generality.

**Conclusion:** stored first moments cause a local update reversal, but removing them is not a reliable cross-seed accuracy repair. Close this test. No additional reset variant, longer horizon or optimizer recommendation follows. Next, review the remaining representation-learning hypotheses against existing evidence before selecting another experiment. RL remains parked.
