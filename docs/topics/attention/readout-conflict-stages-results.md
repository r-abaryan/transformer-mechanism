# Joint affine conflict predates training in both seeds

Executed 2026-10-05. **Both seeds have exact affine-infeasibility certificates at initialization, step 3000 and step 12000.** Native accuracy improves substantially. The remaining learning problem is incomplete removal of conflict, not its first creation by training.

[Short guide](readout-conflict-stages-guide.md) | [Frozen protocol](../../protocols/readout-conflict-training-stages.md) | [Producer](../../../experiments/readout_conflict_stages.py) | [Auditor](../../../experiments/audit_readout_conflict_stages.py) | [Report](../../../results/readout_conflict_stages/report.json) | [Every native/candidate group](../../../results/readout_conflict_stages/metrics.csv) | [Execution log](../../../results/readout_conflict_stages_execution.log) | [Independent audit](../../../results/readout_conflict_stages/audit.json)

## Method

Use seeds 3461/3463 from the same exhaustive-feedback trajectory. Reload actual saved initial/step-3000 checkpoints; verify initial state hashes and continuation parent hashes. Reuse the already certified step-12000 endpoints without another solve or recurrent prefill. The older assigned-query/resampled trajectories are not substituted.

At each stage evaluate the same training loads 2/8/16/32 together: 768 histories per pool, 44,544 queries per seed/stage. Four fresh stages supply 178,176 native answers; two reused endpoints add 89,088. Total reported: 267,264 answers/18,432 history evaluations. Data and seeds are reused, so these counts are not independent experiments. No held-out evaluation or new training occurs.

Each fresh stage gets one unchanged [full-coordinate recovery](shared-readout-recovery-guide.md): retain eight directions; 80-digit transform; one margin LP; exact certificate or original-read positive witness checks. Four new solver calls, capped at 60 seconds/100,000 iterations each; no retries or added stages.

## Native training behavior

Accuracies are percentages. CE and complete maps below refer to load 32. A complete map requires all 32 queries correct.

| Seed | Step | Load 2 % | Load 8 % | Load 16 % | Load 32 % | Load-32 CE | Complete maps / 768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3461 | 0 | 0.0000 | 1.0905 | 6.2663 | 10.6242 | 2.082097 | 0 |
| 3461 | 3000 | 100.0000 | 100.0000 | 99.9674 | 94.4499 | 0.192271 | 149 |
| 3461 | 12000 | 100.0000 | 100.0000 | 100.0000 | 98.9990 | 0.040355 | 561 |
| 3463 | 0 | 50.0000 | 46.9889 | 37.7360 | 29.0405 | 1.740846 | 0 |
| 3463 | 3000 | 100.0000 | 100.0000 | 99.9919 | 94.9259 | 0.175781 | 181 |
| 3463 | 12000 | 100.0000 | 100.0000 | 100.0000 | 98.5636 | 0.049351 | 510 |

The CSV preserves CE and complete maps for every load too. Loads 2/8/16 contain repeated keys only; load 32 contains unique keys only. Present-group metrics equal the all-query metrics; absent groups stay null. Initial native accuracy varies markedly by seed; do not assume a uniform chance score for a particular initialization.

## Affine feasibility at those stages

| Seed | Step | Evidence | New solver calls | Exact certificate rows |
|---|---:|---|---:|---:|
| 3461 | 0 | Certified infeasible | 1 | 46 |
| 3461 | 3000 | Certified infeasible | 1 | 46 |
| 3461 | 12000 | Reused certified endpoint | 0 | 46 |
| 3463 | 0 | Certified infeasible | 1 | 46 |
| 3463 | 3000 | Certified infeasible | 1 | 46 |
| 3463 | 12000 | Reused certified endpoint | 0 | 46 |

All certificates cancel every original class/feature coefficient exactly with nonnegative rational weights summing to one. Each passes the first-index tie condition. All include smaller-load comparisons, so they establish joint conflict, not load-32-only infeasibility. They give no minimum error count. Four fresh solvers report status zero and margin zero; the exact checks, not those messages, establish the conclusion.

Every returned search candidate is a zero head: 16.6667% accuracy and CE 1.791759 in every present seed/stage/load group, with 21 complete maps at load 2 and zero otherwise. These are degenerate diagnostic candidates, not trained replacements. The native endpoint's 246/353 errors remain unchanged.

## Verification and limits

Two runtime preflight methods pass: initialized saved evaluation/audit, singular reads remaining unresolved, ties/absent groups, exact certificate controls, and deliberate read/label/coordinate/head/step/count/CSV defects. The initial preflight failed its RNG-preservation check because model reload constructs random weights before restoring the checkpoint. Isolating reloads with `fork_rng` preserved the check and fixed it before formal evaluation. [Development record](../../../results/readout_conflict_stages_development.json) preserves that failure; no formal run was repeated.

Independent audit passes all six stages/48 native and candidate cells, original historical metrics, initialization/continuation lineage, full labels and affine matrix recurrence, transformed constraints and exact certificates. It verifies 178,176 fresh native queries and the two bound reused endpoints. Maximum checked equation discrepancy is 1.421085e-14; 35 scoped protected files remain unchanged. Cached certificates are checked again without solving. This scope does not audit every historical script.

CPU float64; torch one thread; NumPy 2.5.3, SciPy 1.18.1, torch 2.11.0+cu128. Saved HiGHS 1.12.0 logs report serial simplex. Producer time is 27.12 seconds excluding preflight/audit, not a performance comparison. Four new solves, 32 batched native prefills including fidelity passes, zero optimizer/training/autograd calls. Fixtures are separate.

**Established:** conflict exists before optimization and at both later sampled stages, while the current model's accuracy/CE improves greatly. **Unknown:** whether intermediate stages are feasible, whether conflicts involve the same queries, and why learning does not completely remove them. Keys, values, beta and head change jointly; this comparison isolates no causal parameter and measures no convergence or architectural limit.

Stop this three-stage comparison. Further chronology is not automatic. A next explanatory comparison should follow the same histories' writing/interference across stages, with successful-history controls, before proposing a training change. RL remains parked.

Sources inspected: [Yang et al. section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2), delta recurrence; [Boyd and Vandenberghe slides 5.33-5.35](https://stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf), exact alternatives; [NumPy SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) and [SciPy HiGHS dual simplex](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ds.html), numerical search. The measured chronology is local, not a paper reproduction or published causal explanation.
