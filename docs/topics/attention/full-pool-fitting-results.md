# Perfect finite-pool fitting, incomplete generalization

Executed 2026-10-05. Full-pool L-BFGS produces **100% training accuracy and complete maps at every load in both seeds**, using the unchanged normal-write reader. Exposure-matched extra Adam training also passes the exact 99% fitting gate, but retains errors. The architecture can therefore fit these finite pools. Held-out performance remains weak at load 32.

## Method

The [frozen protocol](../../protocols/full-pool-fitting.md) compares two procedures from identical step-12000 weights per seed. Adam restores its moments and data streams; L-BFGS starts with empty curvature history. Optimizer, batching and stored history change together. This does not isolate a curvature or gradient-noise cause. The [guide](full-pool-fitting-guide.md) explains the objective, directions and budget mathematics.

The reader has 615 parameters, a 16-by-8 memory, normalized tied Q/K, scalar write strength and six values/classes. Each history contains 32 events. Loads 2/8/16/32 have 768 training, 192 validation and 384 test histories each. Every present key is queried. Values stay constant within each history, including repeated keys; value-changing overwrite is outside this study. At loads 2/8/16 every key repeats; at load 32 every key occurs once. Empty groups remain explicit in the raw records.

## Training result

Load-32 accuracy and complete training maps; each pool contains 24,576 queried answers and 768 histories:

| Seed | Starting accuracy | Final L-BFGS accuracy / maps | Final Adam accuracy / maps |
|---|---:|---:|---:|
| 3461 | 98.9990% | 100% / 768 | 99.8454% / 730 |
| 3463 | 98.5636% | 100% / 768 | 99.3490% / 621 |

Both L-BFGS checkpoints answer all 44,544 training queries correctly across 3,072 histories each. Minimum true-class margins are 1.6977 and 0.8560, respectively, calculated from saved logits. These are numerical finite-pool witnesses through normal writes, stronger than the earlier supplied-state oracle. They are not universal representability proofs.

Adam leaves 38/160 load-32 errors. Seed 3461 also loses six previously correct load-16 answers. Both procedures pass every nonempty training-group gate. The original 246/353 errors and failed original budget remain valid historical results. Additional exposure helps Adam too; L-BFGS is not required to reach 99% under this extension.

## All final pools

Cells show **accuracy / mean cross-entropy**. A complete map means every queried key in one history is correct. All final groups and intermediate measurements are in the [336-row CSV](../../../results/full_pool_fitting/evaluation.csv).

| Seed | Procedure | Load | Training | Validation | Test | Complete test maps |
|---|---|---:|---:|---:|---:|---:|
| 3461 | L-BFGS | 2 | 100.0000% / 0 | 100.0000% / 0 | 100.0000% / 0 | 384/384 |
| 3461 | L-BFGS | 8 | 100.0000% / 6.9283e-06 | 100.0000% / 7.6883e-10 | 100.0000% / 0 | 384/384 |
| 3461 | L-BFGS | 16 | 100.0000% / 8.8267e-05 | 98.8281% / 0.37045 | 98.8281% / 0.39841 | 312/384 |
| 3461 | L-BFGS | 32 | 100.0000% / 0.0018771 | 87.4837% / 3.5671 | 86.7513% / 3.8747 | 13/384 |
| 3461 | Adam | 2 | 100.0000% / 0 | 100.0000% / 0 | 100.0000% / 0 | 384/384 |
| 3461 | Adam | 8 | 100.0000% / 3.1032e-05 | 100.0000% / 2.0644e-13 | 100.0000% / 2.9588e-08 | 384/384 |
| 3461 | Adam | 16 | 99.9512% / 0.0012553 | 99.0234% / 0.18105 | 98.8281% / 0.17892 | 312/384 |
| 3461 | Adam | 32 | 99.8454% / 0.019252 | 87.8906% / 1.3935 | 86.5967% / 1.5661 | 6/384 |
| 3463 | L-BFGS | 2 | 100.0000% / 0 | 100.0000% / 0 | 100.0000% / 0 | 384/384 |
| 3463 | L-BFGS | 8 | 100.0000% / 4.3266e-10 | 100.0000% / 0 | 100.0000% / 4.0795e-11 | 384/384 |
| 3463 | L-BFGS | 16 | 100.0000% / 0.00022672 | 98.2422% / 0.55702 | 96.8750% / 1.1846 | 246/384 |
| 3463 | L-BFGS | 32 | 100.0000% / 0.0065731 | 85.4492% / 3.2964 | 86.7757% / 3.1861 | 0/384 |
| 3463 | Adam | 2 | 100.0000% / 0 | 100.0000% / 0 | 100.0000% / 0 | 384/384 |
| 3463 | Adam | 8 | 100.0000% / 1.3873e-05 | 100.0000% / 3.2851e-13 | 100.0000% / 1.476e-05 | 384/384 |
| 3463 | Adam | 16 | 100.0000% / 0.00073388 | 98.2422% / 0.26781 | 97.0703% / 0.42679 | 234/384 |
| 3463 | Adam | 32 | 99.3490% / 0.030051 | 85.7096% / 1.362 | 87.0280% / 1.3516 | 6/384 |

Zero/tiny float64 CE does not prove probability exactly one or convergence. Validation/test pools were previously opened and excluded from selection. They measure exploratory held-out histories at the same length, not longer-input transfer or fresh confirmation.

L-BFGS test-32 accuracy gains 19 answers in seed 3461 and loses 31 in seed 3463 versus Adam. Its validation-32 accuracy is lower in both seeds. Held-out CE is much higher at loads 16/32. Perfect training fit has not produced reliable unseen maps or a held-out winner; CE alone does not identify why.

## Executed budget and verification

| Per seed | L-BFGS | Adam |
|---|---:|---:|
| Optimizer calls | 63 | 6,096 |
| Gradient evaluations | 127 full-objective closures | 6,096 minibatches |
| History exposures | 390,144 | 390,144 |
| Input events | 12,484,608 | 12,484,608 |
| Answer targets | 5,657,088 | 5,657,088 |
| Run time, seeds 3461 / 3463 | 79.61 / 75.54 s | 131.49 / 131.44 s |

Target/history/event counts match separately at every load, including L-BFGS trials. They do not match FLOPs, gradient variance or optimizer cost. Four formal runs total 12,318 optimizer calls, 12,446 gradients and 22,628,352 answer targets. These are repeated exposures, not independent experiments. CPU float64, one thread; GPU unused, CPU peak memory unmeasured. Run times include trajectory recording and training snapshots, excluding setup, final held-out evaluation and audit; this is not a speed benchmark.

Two runtime preflight methods passed, including initialized saved-model evaluator/audit paths and intentional faults. The [formal audit](../../../results/full_pool_fitting/audit.json) passes every recorded gradient, optimizer step and all 112 evaluation cells / 1,024,512 repeatedly evaluated answers. It independently checks labels, matrix gradients, scalar Adam equations, L-BFGS two-loop directions and accepted displacements. L-BFGS trial replay uses the same installed optimizer engine with independently verified archived gradients; it is not a second independent L-BFGS implementation. Maximum gradient/logit discrepancies are approximately 6.34e-15 / 4.55e-13; all 31 protected files remain unchanged.

[Supplementary saved-artifact checks](../../../results/full_pool_fitting/supplementary_verification.json) verify original starting weights/moments, counters, all-group fitting, per-load exposure and positive training margins without new model evaluations. Raw [report](../../../results/full_pool_fitting/report.json), trajectories and checkpoints remain preserved. Audit replays are verification work, separate from formal training counts.

## What changes our understanding

| Question | Answer within this study |
|---|---|
| Can the unchanged architecture perfectly fit these finite training pools? | Yes, two normal-write saved parameter settings do |
| Is the original Adam endpoint a hard fitting ceiling? | No; more Adam exposure passes the 99% gate |
| Does L-BFGS improve final fitting at matched exposure? | Yes, all training errors disappear; optimizer/batch/history causes are not separated |
| Does perfect fitting settle generalization or the learned mechanism? | No; unseen-map reliability fails and the learning cause remains open |

Earlier [fixed-read affine certificates](shared-readout-recovery-results.md) still apply to their original reads. Here the writer and reads change jointly with the classifier, so perfect fitting does not contradict those certificates. No from-scratch, universal-task or full-DeltaNet claim follows.

This bounded development attempt stops here. Next, lock a successful fitting recipe and test its repeatability on fresh seeds/pools before causal trajectory comparisons. State the competing predictions before inspecting new trajectories. Copy/state stay deferred; RL stays parked.

## Sources and inspected scope

[Yang et al. (2024), section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2) supplies the delta correction/read equations; our tied Q/K and scalar-beta reader is a restricted local model. [Liu and Nocedal (1989), author abstract](https://users.iems.northwestern.edu/~nocedal/Abstracts/limited-memory.html) supplies limited-memory quasi-Newton context, with convergence scope that does not automatically cover this neural objective. [PyTorch 2.11 L-BFGS](https://docs.pytorch.org/docs/2.11/generated/torch.optim.LBFGS.html) and installed step/strong-Wolfe source were inspected for closures, directions and evaluation limits. Equally weighted loads, exposure matching and the finite-pool interpretation are local derivations. Papers do not predict our scores.
