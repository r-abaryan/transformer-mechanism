# Causal masking: which width constraints survive?

**Causal masking does not remove every Q/K-width limitation.** One four-token pattern provably needs width at least two, and our width-two construction reproduces it. Whether that limits the output still depends on V.

## The surviving constraint

A causal query at position i sees keys j<=i. Masked probabilities are zero, so taking log ratios across the entire matrix is invalid.

Instead, select query rows I and keys J visible to every selected query. Choose a reference key r in J. For the other keys:

`B_ij = log(A_ij/A_ir) = q_i dot (k_j-k_r)/sqrt(d)`.

The denominator cancels even when different rows have different additional visible keys. Therefore `rank(B)<=d` on this jointly visible rectangle. This is a necessary condition. We have not proved that checking every such rectangle suffices to construct shared Q/K across all prefixes.

## A concrete causal obstruction

Positions below are numbered 1 through 4. The target's visible scores are:

```text
query 1: 0  masked  masked  masked
query 2: 0     0    masked  masked
query 3: 1     0       0   masked
query 4: 0     1       0      0
```

Queries 3 and 4 both see keys 1,2,3. Relative to key 3, the log-odds matrix over keys 1 and 2 is `[[1,0],[0,1]]`. Its determinant is 1 and rank is 2. A width-one factorization cannot produce it.

Width two works: choose keys `[(1,0),(0,1),(0,0),(0,0)]` and queries `sqrt(2)*[(0,0),(0,0),(1,0),(0,1)]`. Scaled dot products give the scores above before masking. Thus the minimum free-Q/K width for this exact pattern is two.

## Live checks and output consequences

[Protocol frozen before execution](../../causal-width-protocol.md). CPU float64; one fixed causal pattern, two value fixtures, no training. Equality tolerance 1e-10.

| Check | Result | What it establishes |
|---|---|---|
| Jointly visible log odds | Singular values 1,1 | Exact width-one obstruction. |
| Causal probabilities | Future weights zero; visible weights positive; rows sum to one | Mask and normalization work. |
| Full probability matrix | Rank 4 | Positive causal diagonal makes it nonsingular; this does not imply Q/K width 4. |
| Scalar V=[0,1,2,3]^T | Width-one output error 5.56e-17; attention difference .212218 | Width-one output matching remains possible despite the pattern restriction. |
| V=[I_3;0,0,0] | Attention recovered from output within 5.56e-17 | Output uniquely identifies the pattern, so width one cannot match it exactly. |
| Scalar-case weights with the identifying V | Output error .212218 | A negative control: changing V exposes the hidden difference. |

Scalar queries use 80 bisection steps over each visible prefix, as in the [earlier output-capacity construction](output-capacity-results.md). The first query has only one visible key and needs no solving. The identifying value matrix makes AV equal to A's first three columns; row normalization recovers the fourth.

## Sources, limits and closure

[Vaswani et al., sections 3.1-3.2](https://arxiv.org/html/1706.03762v7) supplies the scaled-dot-product and causal-mask definitions; the relevant full-text sections were checked. The [head-width sources](head-width-results.md#foundational-and-updated-research) and [identifiability sources](output-capacity-results.md#existing-evidence-and-sources) remain the relevant theoretical context. The rectangle argument here is explicitly derived, not attributed as a reproduced theorem from those papers. Newer publication dates do not override stronger, directly applicable evidence.

The independent 50-digit decimal audit passes all 108 scalar checks, with maximum disagreement 2.23e-16. It reconstructs expected visible scores and checks the rank certificate and value outputs without PyTorch softmax/SVD. Errors above are rounded upward where needed.

Close this bounded question: shared visible keys retain a width constraint; causal probability rank alone is uninformative about required width; values determine whether a pattern restriction matters to the output. This does not characterize every causal pattern, prove an across-input learned solution, or justify wider heads in our trained models. Separate score biases and additional heads/layers are outside scope.

[Runner](../../../experiments/causal_width.py), [raw matrices and outputs](../../../results/causal_width.json), [independent auditor](../../../experiments/audit_causal_width.py), [audit result](../../../results/causal_width_audit.json). Run with `.venv/Scripts/python.exe`; completed experiment artifacts are protected.
