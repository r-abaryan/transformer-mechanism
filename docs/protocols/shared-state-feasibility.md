# Frozen shared-state feasibility diagnostic

Question: with the saved normalized query vectors and classifier fixed, can one 16-by-8 memory give a strictly correct winner for every query in each train-32 history?

Scope: exhaustive-feedback seeds 3461/3463, step 12000, all 768 original train-32 histories per seed. No validation/test selection, training, optimizer calls, changed keys/head, reachability claim or new confirmation. Both native successes and failures are included. Stable-value labels are reconstructed independently. The 1536 histories include six cyclic renamings of 128 families per seed; they are correlated evaluations, not 1536 independent runs.

## Method fixed before evaluation

Reuse the actual saved Reader. Native prefill/read is the baseline. Materialize its normalized query vectors q and keep classifier W,b fixed. For each history, solve over 128 unconstrained state entries and one margin:

$$\max_{S,\gamma}\gamma,\quad
(W_y-W_c)S^Tq_i+b_y-b_c\geq\gamma\ (c\ne y_i),\quad\gamma\leq1.$$

There are 32 times five = 160 class constraints, plus the cap. Free state entries must have `(None,None)` bounds; SciPy's default nonnegative bounds would change the question. Maximizing the capped margin preserves positive-margin existence, not the uncapped maximum. No state-norm constraint is imposed. All correct labels supply the oracle; the write recurrence is bypassed.

CPU float64, torch one thread, NumPy 2.5.3, SciPy 1.18.1. HiGHS dual simplex (`highs-ds`), presolve on, primal/dual feasibility tolerances 1e-9, maxiter 10000, time_limit five seconds per history. One call per history, no retries, solver changes or tolerance changes. Stop after all declared histories; retain every solver failure.

## Certificate rules

Positive: the returned finite state has minimum class margin greater than 1e-7 in the actual Reader and an independent 60-digit decimal read. The decimal check subtracts a conservative 1e-50 times absolute-term scale guard. Scalar/native logits agree within 1e-9 times `(1+max absolute logit)`. The auditor uses 80 digits. This certifies a numerical positive-margin witness for the stored normalized q,W,b, without claiming the solver found an optimum or the state is reachable.

Nonpositive: use the nonnegative dual candidate, including the cap row, only if **exact rational arithmetic on the stored binary floats** verifies zero state coefficients, margin coefficient one and an upper bound <=0. Rounded constraint construction and approximate residuals alone cannot certify an upper bound for unrestricted S. No rational reconstruction or additional solve is allowed. Failing this rule is unresolved. Exact argmax ties are outside strict-positive correctness; report their presence separately.

Record every native/oracle answer, CE, complete-map status, minimum margin, state Frobenius norm, native-to-oracle norm ratio, solver status/iterations/time, dual residual and certificate outcome. Oracle scores are descriptive label-informed bounds, not model quality. Aggregate separately by seed and native complete/incomplete maps. Missing repeat groups are null/count zero. Report counts as evaluations, not independent replications.

## Preflight, protection and stop

Before the formal lock, exercise the actual evaluator and audit entry on an initialized saved checkpoint with independently labeled histories. Test supplied tied logits/missing groups, a known feasible fixture and a conflicting-identical-query fixture. Alter labels, states, certificates, groups and summaries deliberately; the audit must reject them. Derive counts independently from history/query/class dimensions.

Bind executed code, this protocol, solver pins/versions, preflight record, source data/checkpoints and the relevant historical report/audit pairs by hashes. Refuse existing output paths. Preserve all historical artifacts. No full historical study is rerun. Neither finding nor unresolved cases authorize another training recipe.

Sources inspected: [Yang et al. section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2), write/read equations; [Boyd and Vandenberghe slides 5.33-5.35](https://stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf), alternatives/certificate principle; [SciPy HiGHS documentation](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html), bounds, options and marginal signs. The margin reduction and exact certificate specialization are local derivations. No paper predicts the local outcomes.
