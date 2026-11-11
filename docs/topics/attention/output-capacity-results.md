# Attention-pattern capacity versus output capacity

**A head may fail to reproduce an attention pattern yet reproduce its output. Whether that is possible depends on the values.** Two new fixtures pass an independent audit. The older non-uniqueness example and head-width target were reused, not retrained.

## What does equal output require?

For a fixed value matrix V, two normalized attention rows a and b produce the same output precisely when:

`(a-b)V=0` and `(a-b)1=0`.

The second equation preserves the sum of weights. Combine both as `(a-b)[V,1]=0`, where `[V,1]` appends a column of ones.

| Condition | Consequence among probability rows |
|---|---|
| `[V,1]` has full row rank n | Equal output implies a=b. The values are affinely independent. |
| `[V,1]` has rank less than n and a is strictly positive | There is a nonzero direction that changes weights without changing output or their sum. Small enough changes preserve positivity. |
| Values lie on the boundary of this analysis | A probability row with zeros requires checking allowed perturbations. A fixed Q/K family may not realize every probability row. |

The number of locally invisible directions for an interior row is `n-rank([V,1])`. Distinct values are not enough for uniqueness: four distinct scalars have augmented rank at most two. Conversely, rank(V)=n is stronger than necessary; the sum-to-one constraint can supply the missing dimension.

## Missing checks, now run

Reuse the positive four-token target `A=softmax(I_4)`. Its log-odds rank is three, so a free-Q/K head needs width at least three to match A exactly. [Previous derivation](head-width-results.md).

| Value fixture | Result | Meaning |
|---|---|---|
| V=[0,1,2,3]^T | A width-one head matches AV within 4.45e-16, while attention weights differ by up to .212218 | Exact attention-pattern capacity is stricter than output capacity on this fixture. |
| V=[I_3;0,0,0] | Recover A from AV within 5.56e-17; augmented rank is four | Matching output now requires matching A. Width one or two cannot match it exactly. |
| Apply the first fixture's width-one weights to the second V | Output error .212218 | Equality for one value matrix does not imply equality for another. |

Errors are maximum absolute coordinate differences, rounded upward where indicated. These are fixed-input constructions, not task accuracy measurements.

## Why the construction works

For scalar values, choose width-one keys `k_j=v_j=j`. One scalar query q produces `f(q)=sum_j softmax(q*j)_j*j`. Its derivative is the positive variance of j under those weights. Thus f increases continuously from 0 to 3 as q ranges from negative to positive infinity. Every target output here lies strictly inside that interval, so a matching q exists.

We bracketed each of four target outputs on [-10,10] and used 80 bisection iterations. This constructs Q for this input; it does not learn one projection that generalizes across inputs. The archived Q/K-width restriction on the complete pattern remains valid.

For the second fixture, `AV` contains the first three columns of A. The missing fourth weight in each row is `1-sum(first three)`. Therefore the output uniquely determines A. This is an exact algebraic result, not inferred from the numerical rank threshold alone.

## Existing evidence and sources

Archived [S05](../foundations/test-clarifications.md) already used weights [.2,.6,.2] and [.4,.2,.4] with scalar values [0,1,2]; both give output 1. Its exact rational identity is checked in the new audit. We add no duplicate training experiment.

| Source | Reading depth and contribution |
|---|---|
| [Brunner et al., ICLR 2020](https://tik-old.ee.ethz.ch/file/84248b631689290b665ce39984611681/on_identifiability_in_transformers.pdf) | Sections 3.1-3.3 inspected. Nullspace analysis includes the augmented matrix `[T,1]` under probability constraints. Their T includes the head output projection. This is the main source for the identifiability distinction. |
| [Bhardwaj et al., ACL 2021](https://arxiv.org/abs/2106.01269) | Abstract checked. Highlights key-vector restrictions in attention identifiability. Supports keeping unrestricted probability-row ambiguity separate from realizability by a particular head. No benchmark reproduced. |
| [Recent approximation-rank source](head-width-results.md#foundational-and-updated-research) | Previously screened September 2026 preprint concerns approximation under an output-error criterion. It is context, not a replacement for the older, directly relevant argument. |

Source priority follows explicit assumptions, proof quality, relevance and independent evidence. Publication year alone establishes neither accuracy nor superiority. The theorem conditions and local algebra support this conclusion; no claim is made that all conclusions in either older paper were audited.

## Verification, limitations and closure

[Frozen protocol](../../output-capacity-protocol.md). CPU float64 construction; separate 50-digit decimal recomputation and exact rational S05 check. All 72 scalar comparisons pass, with maximum disagreement 2.23e-16. Archived inputs/results and new source/protocol hashes are retained.

Q/K width and value width are separate quantities here. These fixtures allow freely chosen Q/K and fixed V; shared learned projections, causal masks and performance across inputs are not tested. A nullspace alternative need not be representable by the same head. A later output projection can discard further information; equality of final class predictions is weaker still than equality of AV.

**Close the connection:** a head-width limitation on A becomes a limitation on AV only when the values make that distinction necessary. We have not established that increasing head width improves our trained task. No new architecture or training sweep follows.

[Runner](../../../experiments/output_capacity.py), [raw witnesses](../../../results/output_capacity.json), [independent auditor](../../../experiments/audit_output_capacity.py), [audit result](../../../results/output_capacity_audit.json). Run with `.venv/Scripts/python.exe`; the runner protects completed results.
