# Shared-classifier feasibility remains unresolved

Executed 2026-10-05. **Neither seed produced a verified classifier or an exact infeasibility certificate.** The diagnostic completed and its independent audit passed; the mathematical question remains unanswered.

[Learning guide](shared-readout-feasibility-guide.md) | [Protocol](../../protocols/shared-readout-feasibility.md) | [Producer](../../../experiments/shared_readout_feasibility.py) | [Auditor](../../../experiments/audit_shared_readout_feasibility.py) | [Report](../../../results/shared_readout_feasibility/report.json) | [All groups](../../../results/shared_readout_feasibility/metrics.csv) | [Audit](../../../results/shared_readout_feasibility/audit.json)

## Method

Use exhaustive-feedback seeds 3461/3463 at step 12000. Freeze keys, value embeddings, beta, native final states and reads. Search for one affine six-class head per seed across all four existing training loads, with unrestricted weights and positive class gaps. The [guide](shared-readout-feasibility-guide.md#the-mathematics) derives the program.

Each seed supplies 768 histories per load and 44,544 queries: $768(2+8+16+32)$. Each program has 46 variables and 222,721 inequalities including the margin cap. These are reused training pools and two saved seeds, not fresh confirmation or 89,088 independent experiments. No validation/test scores were inspected here.

CPU float64; torch one thread, HiGHS default solver threading; NumPy 2.5.3, SciPy 1.18.1, torch 2.11.0+cu128. Two formal primal solves, each capped at 60 seconds/100,000 iterations, with 1e-9 feasibility tolerances. Formal producer time: 11.83 seconds excluding preflight/audit. This is execution provenance, not a performance benchmark. Zero training updates, optimizer calls or autograd.

## Solver outcomes

| Seed | Solver result | Margin | Solver seconds | Certificate check | Permitted conclusion |
|---|---|---:|---:|---|---|
| 3461 | Status 0, reported optimal; 218 iterations | 0 | 5.432 | 36 positive dual rows; exact rational system inconsistent | Unresolved |
| 3463 | Status 4; HiGHS status 0, Not Set; 0 reported iterations | Not returned | 2.822 | No primal/dual candidate returned | Unresolved |

For 3461, the floating dual stationarity residual is 4.440892e-16. The exact check rejects it as a proof. The returned candidate's guarded margin is -2e-45, below the frozen positive threshold of 1e-7. For 3463, the reported solver error supplies no feasibility evidence. Neither run timed out. The reason for that solver error was not diagnosed in this bounded execution.

No accepted replacement head means no witness accuracy or CE exists. Do not describe the solver failures as a failed trained repair, or infer that a nonlinear head or wider memory is necessary.

## Native training results remain unchanged

Each cell has 768 histories. Complete maps require every present key to be answered correctly.

| Seed | Load | Queries | Accuracy % | Mean CE | Complete maps / 768 |
|---|---:|---:|---:|---:|---:|
| 3461 | 2 | 1,536 | 100.0000 | 0.000000 | 768 |
| 3461 | 8 | 6,144 | 100.0000 | 0.000243 | 768 |
| 3461 | 16 | 12,288 | 100.0000 | 0.001477 | 768 |
| 3461 | 32 | 24,576 | 98.9990 | 0.040355 | 561 |
| 3463 | 2 | 1,536 | 100.0000 | 0.000000 | 768 |
| 3463 | 8 | 6,144 | 100.0000 | 0.000051 | 768 |
| 3463 | 16 | 12,288 | 100.0000 | 0.001608 | 768 |
| 3463 | 32 | 24,576 | 98.5636 | 0.049351 | 510 |

Loads 2/8/16 contain repeated keys only; load 32 contains unique keys only. Present-group metrics equal the corresponding all-query metrics. Absent groups remain null in the CSV, not zero accuracy. Load-2 CE is float64 zero, not proof of exact certainty. The original 246/353 wrong answers remain.

## Verification and stopping boundary

Three preflight methods pass: known separable/conflicting inputs, free negative weights, exact certificate/tie controls, supplied missing groups, initialized saved-model evaluator/audit, and deliberate read/label/head/group/count defects. Preflight solves are separate from the two formal calls.

Independent audit checks labels, affine matrix recurrence, native reads/logits, every constraint row at a deterministic probe against direct class scores, historical metrics, counts, CSV and hashes. It passes all eight native cells/89,088 queries; maximum checked equation discrepancy is 8.526513e-14. All 19 scoped protected files remain unchanged. **Audit success verifies these computations and records; it does not certify the solver's optimum or resolve feasibility.** No formal run was repeated and no executed source was overwritten.

Stop at the declared two solves. Any numerical recovery needs a separately declared method and verification on the same fixed features and all four loads. There is no basis here for choosing an architecture change. The earlier per-history [shared-state witnesses](shared-state-feasibility-results.md) remain valid; whether the actual reads admit one universally correct affine head, and why training leaves them as they are, remain open. RL stays parked.

Sources inspected: [Boyd and Vandenberghe, slides 5.33-5.35](https://stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf), certificate principle; [SciPy HiGHS dual simplex](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ds.html), bounds, marginals, tolerances and statuses. Our margin program and measured outcomes are local, not published predictions. In particular, the sources do not justify treating an approximate zero-margin result as exact infeasibility.
