# Cold-start write credit assignment: results

[Learning guide](write-credit-assignment-guide.md) | [Frozen protocol](../../protocols/write-credit-assignment.md) | [Executed source](../../../experiments/write_credit_assignment.py) | [Raw report](../../../results/write_credit_assignment/report.json) | [All positions](../../../results/write_credit_assignment/metrics.csv) | [Independent audit](../../../results/write_credit_assignment/audit.json)

**Error feedback survives early transport. Shared-value gradients exhibit net temporal opposition after pooling histories. Neither establishes the cause of incomplete fitting.**

## Method

Seeds 3461/3463, exhaustive step 12000; all 768 existing train-32 histories per seed. Zero initial state, fixed saved weights, final all-query CE. Final correctness fixes correct/error cohorts. Each cohort contributes masked CE sum / (4 H m), preserving this load's original objective weight. Other loads are excluded. All formal queries are unique; repeat-group metrics are absent, not zero accuracy.

CPU float64, one thread. Three backward partitions reuse each seed's forward graph. Actual Reader state/logit fidelity passes. Zero training or optimizer calls; no held-out evaluation or new oracle search. Primary execution/save takes 0.78 seconds, excluding startup, preflight and audit. This is provenance, not a speed benchmark.

## Fitting and error-signal transport

Each seed grades 24,576 answers; earlier native fitting failures reproduce exactly. Norm ratios compare the entire history-batch gradient tensor at each state, not every history individually.

| Seed | Train accuracy % | Errors | Complete maps / 768 | Error G0 / G32 % | Error G1 / G32 % |
|---|---:|---:|---:|---:|---:|
| 3461 | 98.9990 | 246 | 561 | 69.4082 | 70.5814 |
| 3463 | 98.5636 | 353 | 510 | 71.9846 | 73.0545 |

G1 is the state after the first write; G0 is the fixed zero-start sensitivity. Error state gradients remain nonzero at every measured position. Early value-activation norms are not consistently smaller: first/last event norms are 6.017e-5 / 7.518e-5 in 3461 and 7.236e-5 / 6.433e-5 in 3463. This does not establish universally good gradient flow.

## Shared-value aggregation

R is norm of the summed position gradient / sum of separate position norms. The supplemental J/E column compares squared resultant norm with the sum of squared position norms. J/E below one establishes negative net cross terms.

| Seed | Cohort | R % | J/E | Negative / positive position pairs |
|---|---|---:|---:|---:|
| 3461 | Correct | 14.3059 | 0.589053 | 265 / 231 |
| 3461 | Error | 13.1602 | 0.466757 | 254 / 242 |
| 3461 | Total | 10.7369 | 0.319651 | 252 / 244 |
| 3463 | Correct | 11.9481 | 0.390680 | 266 / 230 |
| 3463 | Error | 11.1291 | 0.359508 | 253 / 243 |
| 3463 | Total | 10.0323 | 0.296866 | 251 / 245 |

All six cross-term sums are negative. No pair is unresolved under the 1e-14 guard. The error embedding gradients remain nonzero: norms 2.148e-4 / 2.531e-4. R does **not** mean that 87-89% of learning or accuracy was lost; directional spread also affects it. These position contributions pool histories, so they do not establish opposition inside each individual sequence. Q/K routes and actual Adam movements were not measured here.

The [signed-cross-term supplement](../../../results/write_credit_assignment/cross_terms.json) was explicitly declared **after opening primary results**, because R alone does not isolate opposition. Its [protocol](../../protocols/write-credit-cross-terms.md) and [source](../../../experiments/annotate_write_credit_assignment.py) preserve primary evidence. Three aligned/orthogonal/opposing controls and independent scalar pair sums pass. No additional model evaluation or gradient computation ran.

## Verification and limits

Three preflight methods exercise initialized saved reload/evaluator/audit, supplied ties/absent cohorts, finite differences, transport controls and intentional label/logit/mask/gradient/group/summary defects. One pre-freeze write-count correction preserves the initial passing record; no formal recovery or rerun occurred.

Independent affine forward and analytic reverse recurrence pass all **198 state rows, 192 event positions and 49,152 unique answers**. Maximum discrepancy: 4.263257e-14; 32 scoped protected files unchanged. The primary diagnostic uses 49,152 writes and native fidelity 98,304, totaling 147,456. Supplemental reduction checks 2,976 correlated position pairs. Counts are not independent seeds.

Compared with [Yang et al., sections 2.2/3.1](https://arxiv.org/html/2406.06484v2#S2.SS2), the local forward follows the delta recurrence in transposed orientation. Backward transport, cancellation statistics and measured scores are local evidence, not reproduced paper performance.

**Supported:** partial state-gradient attenuation and net opposition among pooled shared-value position contributions. **Unknown:** within-history versus cross-history origin, learning trajectory, causal repair benefit and other parameter routes. Stop this diagnostic. A useful next bounded reduction would separate within-history from cross-history cross terms using these saved gradients, before proposing a training change. RL remains parked.
