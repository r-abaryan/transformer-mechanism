# Fixed-read affine infeasibility is now certified

Executed 2026-10-05. **Both saved models' training reads admit exact certificates excluding a globally perfect affine classifier across loads 2/8/16/32.** This resolves the earlier numerical uncertainty under the stored-read arithmetic contract. It does not prove lost information or explain historical learning.

[Educational guide](shared-readout-recovery-guide.md) | [Frozen recovery protocol](../../protocols/shared-readout-numerical-recovery.md) | [Original unresolved attempts](shared-readout-feasibility-results.md) | [Producer](../../../experiments/shared_readout_recovery.py) | [Auditor](../../../experiments/audit_shared_readout_recovery.py) | [Report](../../../results/shared_readout_recovery/report.json) | [Candidate groups](../../../results/shared_readout_recovery/metrics.csv) | [Execution log](../../../results/shared_readout_recovery_execution.log) | [Independent audit](../../../results/shared_readout_recovery/audit.json)

## Method and numerical diagnosis

Same exhaustive-feedback seeds 3461/3463 at step 12000. Reuse every original saved read, label and native state across four training pools, without a new recurrent prefill or held-out evaluation. Per seed: 3,072 histories, 44,544 queries, 46 free optimization variables and 222,721 inequalities including the margin cap. These are two reused trained seeds, not independent experiments per query.

Keep all eight directions. Center the reads, use the complete SVD right basis and power-of-two scales, and compute transformed coordinates at 80 digits before passing float64 coefficients to HiGHS. No dimensional reduction, label selection or norm constraint. The [guide](shared-readout-recovery-guide.md#why-the-solver-needed-help) explains why this preserves the affine hypothesis class in exact arithmetic. Verify certificates on the original reads, not rounded transformed features.

| Seed | Largest centered singular value | Sixth | Seventh | Eighth | Maximum residual outside value span |
|---|---:|---:|---:|---:|---:|
| 3461 | 577.772157 | 49.224956 | 8.099075e-13 | 5.952114e-13 | 4.993785e-15 |
| 3463 | 508.182631 | 65.128761 | 7.133125e-13 | 1.253519e-13 | 4.217947e-15 |

This agrees with the local exact-arithmetic span argument for six value embeddings. Tiny singular values are numerical indicators, not exact-rank proofs. The recovered search succeeds after changing coordinates; it does not isolate the internal reason for the earlier HiGHS error or attribute every failure to conditioning.

## Recovered outcomes

| Seed | Reported solver status | Iterations | Solver seconds | Margin | Exact certificate rows | Independent outcome |
|---|---:|---:|---:|---:|---:|---|
| 3461 | 0 | 228 | 4.588 | 0 | 46 | Certified infeasible |
| 3463 | 0 | 165 | 4.042 | 0 | 46 | Certified infeasible |

All certificate weights are nonnegative, sum exactly to one, and cancel every original class/feature coefficient exactly. Both pass the first-index tie condition. This certifies the failure of a perfect real-affine classifier on the stored binary float64 reads, not merely a failure to exceed the 1e-7 acceptance threshold. It does not certify arbitrary floating-point implementations that exploit rounding artifacts.

| Seed | Supported rows: load 2 | Load 8 | Load 16 | Load 32 |
|---|---:|---:|---:|---:|
| 3461 | 0 | 2 | 8 | 36 |
| 3463 | 0 | 0 | 1 | 45 |

Rows are class comparisons, not distinct queries or independent seeds. These certificates establish joint conflict and include smaller-load constraints. They are neither minimal conflicts nor proof of load-32-only infeasibility.

Both returned primal heads have zero weights/biases. They produce ties, predict class zero, and score 16.6667% accuracy/CE 1.791759 in every seed/load and present unique/repeat group. Complete maps are 21 at load 2 and zero at loads 8/16/32. These degenerate search candidates are not trained models or proposed replacements. No positive witness is accepted.

## Original evidence and verification

Native scores are unchanged: smaller loads remain 100%; load-32 accuracy is 98.9990%/98.5636%, CE 0.040355/0.049351, complete maps 561/510 of 768, and errors 246/353. See [every original native group](../../../results/shared_readout_feasibility/metrics.csv). The certificate gives no minimum error count and does not make every original error inevitable. Earlier [correct supplied states](shared-state-feasibility-results.md) remain valid.

Three preflight methods pass, including initialized saved evaluator/audit, full-coordinate/cancellation checks, singular-transform rejection, conflicting labels, missing groups and deliberate transformation/head/certificate/count defects. One formal recovery solve per seed, no retries. Independent audit verifies exact transform nonsingularity, 100-digit coordinate recomputation, every transformed constraint row at a deterministic probe, original input identity, actual Reader candidate scores, group/support counts, CSV and exact all-class certificates. All 89,088 source queries and 89,088 candidate reads pass; maximum checked equation discrepancy is 1.065814e-14. All 29 scoped protected files remain unchanged, including the original failed attempts.

CPU float64; NumPy 2.5.3, SciPy 1.18.1, torch 2.11.0+cu128, bundled HiGHS 1.12.0. Torch one thread; HiGHS default configuration reports serial simplex in the saved log. Same 60-second/100,000-iteration caps and 1e-9 tolerances. Producer execution takes 17.51 seconds excluding preflight/audit; this is provenance, not a benchmark. Zero training updates, optimizer calls, autograd or new recurrent prefills. The original two solves and these two recovery solves are separate diagnostic attempts.

Stop the classifier feasibility branch. The useful next question is how training produces reads with this joint conflict. A bounded comparison of saved training stages can investigate its emergence; no additional head search, nonlinear classifier, larger memory or training budget follows automatically. RL remains parked.

Sources inspected: [NumPy SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html), right basis; [SciPy HiGHS dual simplex](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ds.html), solver bounds/options/statuses; [Boyd and Vandenberghe slides 5.33-5.35](https://stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf), exact alternatives; [Yang et al. section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2), delta equations. Local derivations and certified checkpoint results are not paper-score reproductions or universal architecture claims.
