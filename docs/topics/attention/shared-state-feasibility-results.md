# Correct shared states exist at the saved weights

Executed 2026-10-04. **Every tested training history admits one positive-margin shared memory with the learned keys and classifier fixed.** The original width is sufficient for these particular histories under supplied states. The native write process still makes errors.

[Short learning guide](shared-state-feasibility-guide.md) | [Frozen protocol](../../protocols/shared-state-feasibility.md) | [Source](../../../experiments/shared_state_feasibility.py) | [Raw report](../../../results/shared_state_feasibility/report.json) | [Every history](../../../results/shared_state_feasibility/metrics.csv) | [Independent audit](../../../results/shared_state_feasibility/audit.json).

## Method and execution

Use exhaustive-feedback seeds 3461/3463 at step 12000. Evaluate the same 768 train-32 histories per seed. Each history has all 32 keys once, with six possible value classes. These are 128 context families with six cyclic renamings, reused across seeds. They are not 1536 independent runs or fresh confirmation. Validation/test are not evaluated or used for selection here.

Native baseline: actual saved-model prefill/read. Intervention: one free 16-by-8 state per history, chosen using all correct labels. Keep normalized queries and classifier fixed. Solve the [guide's margin inequalities](shared-state-feasibility-guide.md#what-a-margin-means) using one capped-margin linear program per history. State size stays 1 KiB in float64; this is tensor payload, not allocator memory.

CPU float64, torch one thread, NumPy 2.5.3, SciPy 1.18.1, HiGHS dual simplex. All 1536 calls finish successfully without retry. Formal execution takes 46.32 seconds excluding preflight/audit; this is provenance, not a speed comparison. **Zero optimizer calls or training updates.** Actual native/oracle reads each produce 49,152 answers, totaling 98,304 answer outputs. Preflight uses separate initialized fixtures.

## Findings

Each seed has 24,576 queries. Complete maps require all 32 answers correct from the same state.

| Seed | Native accuracy % | Native complete maps / 768 | Positive shared-state witnesses | Oracle accuracy % | Native errors recovered | Nonpositive / unresolved |
|---|---:|---:|---:|---:|---:|---:|
| 3461 | 98.9990 | 561 | 768 | 100.0000 | 246 | 0 / 0 |
| 3463 | 98.5636 | 510 | 768 | 100.0000 | 353 | 0 / 0 |

All 465 incomplete native maps recover under the supplied states; all native successes remain correct. This recovers 599 wrong answers with zero harms. Native fit failures are preserved, not replaced by oracle scores.

| Seed | Positive witnesses among native complete maps | Positive witnesses among native incomplete maps | Smallest verified native-read oracle margin | Exact target ties |
|---|---:|---:|---:|---:|
| 3461 | 561 / 561 | 207 / 207 | 0.9999999999989592 | 0 |
| 3463 | 510 / 510 | 258 / 258 | 0.9999999999983675 | 0 |

All independently guarded scalar margins also exceed the frozen 1e-7 threshold. No formal history needed a nonpositive certificate. The conflicting-query certificate is a preflight control, not an observed learned-model failure.

## State size and loss caveats

| Seed | Native CE | Oracle CE | Mean native state norm | Mean oracle state norm | Maximum oracle/native norm ratio |
|---|---:|---:|---:|---:|---:|
| 3461 | 0.040355 | 0.686312 | 13.683692 | 4.517430 | 2.002761 |
| 3463 | 0.049351 | 0.680806 | 11.821715 | 4.694160 | 2.221971 |

Oracle CE is worse despite perfect accuracy. The program caps its common-margin objective at 1 and does not minimize CE. Mean norm is smaller, but some states are larger than their native counterpart. These are selected feasible states, not minimum-norm solutions, calibrated predictions or states shown to be reachable.

## Verification and interpretation

Three preflight methods pass: feasible/conflicting fixtures, necessary negative entries, tied logits/absent groups, initialized saved reload, actual evaluator/audit entry and deliberate label/state/logit/certificate/group/summary defects. Before the formal freeze, Windows denied access to a newly created `TemporaryDirectory`; switching to the repository's existing unique `results/test_runs` fixture pattern resolved it. [Development record](../../../results/shared_state_feasibility_development.json). No formal run was interrupted or repeated.

Independent audit checks all 1536 states using 80-digit scalar reads, actual Reader calls, separate labels, matrix native equations, historical baseline agreement, all CSV entries and independent counts. Maximum logit discrepancy is 4.973799e-14. All 16 scoped protected files remain unchanged. This scope binds relevant inputs/code and historical report/audit pairs; it is not a rerun or certification of every earlier study.

**Supported:** the fixed learned queries/classifier permit correct simultaneous decoding for every tested training history at the original width. The recurrence's native output misses available positive-margin states.

**Unknown:** whether these states are reachable or stable under learned writes; whether every possible history is feasible; why joint training leaves the observed errors. Classification feasibility does not establish exact value reconstruction or an architectural impossibility elsewhere.

Sources inspected: [Yang et al. section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2), memory equations; [Boyd and Vandenberghe slides 5.33-5.35](https://stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf), certificate principle; [SciPy HiGHS](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html), solver bounds/options/sign conventions. The margin reduction is local; these papers do not predict our scores or provide this trained repair.

Stop this feasibility study. The next distinct question is whether the learned write rule preserves an already correct supplied shared state when replaying the same history. That needs a separate protocol and cannot, by itself, rule out other stable states. RL remains parked.
