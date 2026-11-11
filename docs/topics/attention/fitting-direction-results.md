# Fitting improves when the direction rule changes

The frozen direction hypothesis **passes in all three seeds**. With the same full objective and line-search policy, L-BFGS produces lower training CE and fewer errors than the negative-gradient branch. Its final objective is 2.96 to 18.45 times lower. Held-out CE is worse, so this is not a generalization repair.

## Method and meaning

A direction tells the optimizer where to move. A line search chooses how far. See the [short mathematical guide](fitting-direction-guide.md) and [frozen protocol](../../protocols/fitting-direction.md).

| Fixed | Intervention |
|---|---|
| Same saved Adam-12000 weights, seeds 3473/3479/3481 | L-BFGS direction versus negative gradient |
| Same 615-parameter reader, zero-start normal writes | One guarded direction assignment before the directional derivative |
| Same training pools and equal-load exhaustive CE | No supervision, architecture or data change |
| Same native search, initial trial policy, counters and tolerances | Accepted lengths and trial counts may differ |
| Same cap of 128 full-gradient evaluations, including rejected trials | Actual counts are reported below |

These are three paired initialization seeds on one previously opened dataset, not six independent seeds or a fresh dataset confirmation. Each load has 768 training, 192 validation and 384 test histories. Six related value shifts per base history are not independent examples. Values stay constant for a key within each history; this is not changing-value overwrite evidence.

## Training results

The primary objective averages CE equally across loads 2/8/16/32. Errors and complete maps below are load 32; every lower-load training answer and map is correct in both branches. Both pass the existing 99% all-training-group gate. Perfect fitting still fails for L-BFGS seed 3481.

| Seed | Direction | All-load objective | Train-32 accuracy | Train-32 CE | Errors | Complete maps |
|---|---|---:|---:|---:|---:|---:|
| 3473 | L-BFGS | 0.0004363 | 100.0000% | 0.001639 | 0 | 768/768 |
| 3473 | Negative gradient | 0.0080508 | 99.5036% | 0.030709 | 122 | 652/768 |
| 3479 | L-BFGS | 0.0020771 | 100.0000% | 0.007988 | 0 | 768/768 |
| 3479 | Negative gradient | 0.0100521 | 99.2839% | 0.038454 | 176 | 605/768 |
| 3481 | L-BFGS | 0.0035381 | 99.9756% | 0.013711 | 6 | 762/768 |
| 3481 | Negative gradient | 0.0104868 | 99.1699% | 0.040849 | 204 | 588/768 |

Complete maps require every queried key in a history to be answered correctly. Thus the negative-gradient branch exceeds 99% answer accuracy but completes only 588 to 652 of 768 load-32 maps. The L-BFGS six-error exception remains six related shifts of one family, exactly as in confirmation.

The predeclared primary rule requires lower all-load CE by more than 1e-8 and no more errors in each seed. It passes without selecting checkpoints or changing settings. All saved training milestones and nonempty unique/repeated-key groups are in the [CSV](../../../results/fitting_direction/evaluation.csv); absent groups remain explicit.

## Held-out results, descriptive

These sets were already opened. The training-only decision was written before this evaluation.

| Seed | Direction | Validation-32 accuracy / CE | Test-32 accuracy / CE | Validation maps | Test maps |
|---|---|---:|---:|---:|---:|
| 3473 | L-BFGS | 86.6536% / 4.0680 | 87.4023% / 4.2212 | 6/192 | 0/384 |
| 3473 | Negative gradient | 87.3047% / 1.0518 | 88.6719% / 1.0055 | 0/192 | 18/384 |
| 3479 | L-BFGS | 87.0280% / 2.9364 | 88.6637% / 2.6227 | 6/192 | 12/384 |
| 3479 | Negative gradient | 87.6628% / 1.0015 | 88.6719% / 0.8838 | 6/192 | 12/384 |
| 3481 | L-BFGS | 86.2305% / 2.3911 | 87.5895% / 2.2299 | 0/192 | 6/384 |
| 3481 | Negative gradient | 87.6953% / 0.9693 | 87.4268% / 0.9787 | 0/192 | 6/384 |

L-BFGS validation-32 accuracy is lower in every seed. Test accuracy effects are mixed. Its validation/test CE is higher at loads 16 and 32 in every seed; lower loads remain in the CSV. Neither branch establishes reliable unseen complete maps. Loss/accuracy disagreement alone does not establish a calibration mechanism.

## What the step traces establish

First optimizer steps are identical in every pair. Later directions differ. Every control trial and final checkpoint is tensor-exact with the earlier native confirmation control. The audit checks negative-gradient directions independently as minus the gradient and L-BFGS directions through a separate two-loop calculation.

L-BFGS makes 63 updates per seed; negative gradient makes 39. Both use the same gradient-evaluation cap. More trials per negative-gradient update consume that allowance. All 306 accepted steps satisfy sufficient decrease and curvature conditions. No accepted step violates either condition; the overall evaluation budget still limits progress. We have not isolated direction orientation from its scale or identified which representation changes improve fitting.

## Execution and verification

| Seed | Direction | Gradient evaluations | Optimizer calls | Training answer targets |
|---|---|---:|---:|---:|
| 3473 | L-BFGS | 127 | 63 | 5,657,088 |
| 3473 | Negative gradient | 128 | 39 | 5,701,632 |
| 3479 | L-BFGS | 128 | 63 | 5,701,632 |
| 3479 | Negative gradient | 128 | 39 | 5,701,632 |
| 3481 | L-BFGS | 127 | 63 | 5,657,088 |
| 3481 | Negative gradient | 128 | 39 | 5,701,632 |

Actual formal totals: **766 gradients, 306 optimizer calls, 34,120,704 answer targets**, 2,353,152 history presentations and 75,300,864 input events. Repeated presentations are exposure, not independent observations. Warm-up was reused and is excluded. End-to-end producer time was 178.35 seconds, excluding preflight, profiling and audit. CPU float64, three workers, one thread each; GPU unused and peak CPU memory unmeasured. This is not a speed benchmark.

Two runtime preflight methods passed in 12.519 seconds, including deliberate fault rejection. Four complete profile calls used 11 extra gradients and are excluded from formal counts. The [audit](../../../results/fitting_direction/audit.json) verifies every archived gradient/step and all 168 evaluation cells (1,536,768 repeated evaluation answers); 34 protected files stayed unchanged. Maximum discrepancies: gradient 1.36e-15, direction 2.85e-16, logits 4.55e-13. The audit shares PyTorch's line-search engine, rather than providing a second independent search implementation. The [supplement](../../../results/fitting_direction/supplementary_verification.json) independently reconstructs the primary contrast from saved logits and checks all 504 CSV rows. [Raw report](../../../results/fitting_direction/report.json).

## Conclusion and remaining question

The direction rule contributes to the observed fitting advantage **under this objective, search policy and budget**. Removing minibatch sampling and keeping the same search does not remove the gap. This does not attribute the entire Adam comparison to curvature, prove convergence, or establish a universal optimizer winner.

Stop this contrast without extra budgets or optimizer variants. Next, use the saved trajectories to formulate one representation-level prediction: which change in memory writes/reads or their classifier improves training margins while unseen maps remain weak? A trajectory difference would be descriptive; claiming a cause needs a controlled intervention. Copy/state stay deferred and RL stays parked.

Sources inspected: [PyTorch 2.11 documentation](https://docs.pytorch.org/docs/2.11/generated/torch.optim.LBFGS.html) and [implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/lbfgs.py) for the tested direction/search machinery; [Liu and Nocedal (1989), author abstract](https://users.iems.northwestern.edu/~nocedal/Abstracts/limited-memory.html) for historical method context, not a neural convergence guarantee. Equations are explained in the guide. All scores and causal-scope statements here are local measurements and interpretations, not paper forecasts.
