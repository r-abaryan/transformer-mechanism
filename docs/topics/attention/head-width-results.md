# Head width: what can one attention head represent?

**Q/K width constrains score relationships, but does not directly bound the rank of the attention probabilities.** Four controlled fixture families pass, with an independent decimal/rational audit. No model was trained or modified.

## Three matrices, three meanings

Rank counts independent rows or columns of a matrix. Head width d is the number of coordinates in each query/key vector.

| Matrix | Definition | What width constrains |
|---|---|---|
| Scores S | `QK^T/sqrt(d)` | `rank(S) <= d`. This follows from matrix multiplication. |
| Probabilities A | Row-wise `softmax(S)` | Can have rank greater than d. Softmax is nonlinear. |
| Log odds B | `B_ij=log(A_ij/A_i,n)`, j<n | `rank(B) <= d` for positive unmasked dot-product attention. |

For log odds, the softmax denominator cancels:

`B_ij = S_ij-S_i,n = q_i dot (k_j-k_n)/sqrt(d)`.

Thus B also factors through d coordinates. Unlike raw logits, it removes the irrelevant freedom to add a constant to each row. For freely chosen Q/K, `rank(B)<=d` is also sufficient: factor B, append a zero reference key and absorb sqrt(d) into Q. A trained head with `Q=XW_Q` and `K=XW_K` can face additional constraints from X and shared weights.

## Measured examples

CPU float64, four tokens; numerical rank threshold 1e-10. Full singular values and fixtures are saved. [Frozen protocol](../../head-width-protocol.md).

| Fixture | Result | Meaning |
|---|---|---|
| Width 1, Q=K=[0,1,2,3]^T | Score rank 1; probability rank 4 | Refutes "attention probability rank cannot exceed head width." |
| Desired A=softmax(I_4) | Log-odds rank 3; singular values 2,1,1 | Widths 1 and 2 cannot represent this exact positive unmasked pattern. |
| Best rank-1 / rank-2 approximation of its B | Frobenius errors 1.414214 / 1.000000 | Irreducible errors in this log-odds metric, not bounds on prediction error. |
| Explicit width-3 Q/K construction | Maximum probability error 0 | This particular target pattern is representable at width 3. |
| Causally masked zero scores | Probability rank 4; determinant 1/24 | A causal mask alone can yield full rank; masked zeros invalidate the global log-ratio formula. |

The first probability matrix has determinant .000434833844 and smallest singular value .00658796. Its full rank is not a near-threshold artifact. Independently, `exp(i*j)=(exp(i))^j` is a Vandermonde matrix with distinct bases, so its determinant is nonzero. Positive row normalization preserves that rank.

## Why the width-three example is conclusive

For A=softmax(I_4), B is:

```text
 1  0  0
 0  1  0
 0  0  1
-1 -1 -1
```

Its Gram matrix `B^T B=I_3+ones(3,3)` has eigenvalues 4,1,1, hence B has rank 3. The discarded singular values give the stated optimal low-rank errors. Set `Q=sqrt(3)*B`, `K=[I_3;0,0,0]` to reproduce A. This construction is an algebraic witness, not a learned solution or a recommended width.

## Architectural implication and limits

When a fixed model width is split equally among more heads, each head usually becomes narrower. That restricts each head's possible log-odds patterns. More heads also supply multiple independently mixed value streams, so this does not prove that fewer heads improve the whole model.

This analysis assumes pure dot-product scores without a separate additive score-bias matrix. It concerns simultaneous patterns across query rows on one input, not a universal recall-capacity bound. Causal masks, approximation tolerance, values, depth and residual paths require separate reasoning. In particular, matching A exactly is stronger than matching AV or the final prediction. It does not explain our earlier six-pair failures or justify a head-count training sweep.

## Foundational and updated research

| Primary source | Checked scope | Connection |
|---|---|---|
| [Bhojanapalli et al., ICML 2020](https://proceedings.mlr.press/v119/bhojanapalli20a/bhojanapalli20a.pdf) | Section 2.1, Theorem 1 and construction inspected. | An established head-width expressivity bottleneck. Their theorem includes conditions on the input embedding matrix; our free-Q/K construction is a narrower algebraic exercise, not their benchmark reproduction. |
| [Sui, Zhang and Tang, arXiv v2, September 22, 2026](https://arxiv.org/abs/2608.28150v2) | Abstract and version history checked; preprint, no peer-review status established here. | Studies approximation rank of normalized attention under an output-error criterion. Its authors distinguish theoretical complexity bounds from empirical explanations their measurements do not support. We have not audited the proofs or reproduced results. |

Search checked September 30, 2026. Exact representability, numerical rank and useful approximate rank are different questions. The recent source does not supersede the older theorem or establish that larger heads improve our task.

## Verification and stopping decision

The independent audit uses 50-digit decimal softmax/determinants and an exact integer Gram matrix. It checks 86 scalar results; maximum disagreement is 2.23e-16 (rounded upward). An initial exact decimal comparison of 1/24 failed because 1/3 has a repeating expansion; exact rational arithmetic now checks that identity, without changing the experiment or tolerance.

Close this representability probe. Retain the precise log-odds constraint and reject the simpler probability-rank claim. No novel architecture or trained-model improvement is established.

[Runner](../../../experiments/head_width.py), [raw matrices and spectra](../../../results/head_width.json), [independent auditor](../../../experiments/audit_head_width.py), [audit result](../../../results/head_width_audit.json). Run with `.venv/Scripts/python.exe`; the runner protects completed evidence.

Follow-up completed: [when pattern limits do and do not constrain output](output-capacity-results.md). Two value fixtures separate these cases without retraining.

Causal extension completed: [jointly visible keys retain width constraints](causal-width-results.md), with one exact counterexample and two value cases.
