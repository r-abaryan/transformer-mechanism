# LayerNorm versus RMSNorm: what centering changes

**They differ in function and gradient, even when their outputs match at one point.** All68 cases and the independent audit pass. This establishes mathematical properties, not which normalization trains our recall model better.

## How they work

For one token's d features, let `mu=mean(x)` and `c=x-mu`.

| Method | Bias-free formula | What the denominator measures |
|---|---|---|
| LayerNorm | `gamma*c/sqrt(mean(c^2)+epsilon)` | Variation around the mean |
| RMSNorm | `gamma*x/sqrt(mean(x^2)+epsilon)` | Magnitude relative to zero |

`mean(x^2)=mean(c^2)+mu^2`. RMSNorm's denominator therefore includes the mean's magnitude. LayerNorm subtracts the mean first. Both normalize features within a token; neither requires statistics from other examples.

Gamma is a learned feature-wise gain. We set LayerNorm bias to zero to isolate centering; native RMSNorm has no bias. With nonuniform gamma, LayerNorm's final output need not have zero mean. Neither method makes features Gaussian or proves stable training.

```mermaid
flowchart LR
    X[One token's features] --> C[Subtract feature mean]
    C --> L[Divide by centered RMS]
    X --> R[Divide by raw RMS]
    L --> G1[Apply gain: LayerNorm]
    R --> G2[Apply gain: RMSNorm]
```

## Methods and measured findings

CPU float64, PyTorch2.14, one thread. Dimensions4/16; ordinary, centered, constant, near-zero and zero vectors; epsilon0/1e-5; unit/nonuniform gains. Undefined zero-denominator cases are excluded. Explicit formulas are compared with native operations, autograd and finite differences. [Frozen corrected protocol](../../rmsnorm-math-protocol.md).

Examples below use d=4, gamma=1 and epsilon1e-5.

| Question / input | LayerNorm result | RMSNorm result | Legitimate conclusion |
|---|---|---|---|
| Add3 to every feature of `[1,2,3,4]` | Output unchanged | Largest output change0.34755 | Common-shift invariance differs |
| Constant input `[2,2,2,2]` | Output all0 | Output all0.99999875 | Centering discards the constant component in this branch |
| Centered input `[-3,-1,1,3]` | Same output as RMSNorm | Same output as LayerNorm | Equal values at this point; not equal functions |
| Derivative along all-ones direction at that centered input | Approximately0 | Each component0.447213 | Local gradients differ despite equal outputs |
| Double near-zero `[1,-2,3,-4]*1e-6` | Largest change0.00110679 | Largest change0.00126491 | Positive epsilon breaks exact scaling symmetry |

With epsilon0, positive scaling2 preserves every defined tested output. Negative scaling-1 reverses the bias-free output. For fixed positive epsilon, exact scale invariance does not hold generally: `N(a*x;epsilon)=N(x;epsilon/a^2)` for a>0, with the same gain. Ordinary large-magnitude inputs can still show small differences.

## The gradient mathematics

A Jacobian J describes how a small input change alters output. Let `P=I-11^T/d`, where 1 is the all-ones vector.

`J_LN=diag(gamma)[P/r - cc^T/(d*r^3)]`, with `r=sqrt(mean(c^2)+epsilon)`.

`J_RMS=diag(gamma)[I/r - xx^T/(d*r^3)]`, with `r=sqrt(mean(x^2)+epsilon)`.

LayerNorm satisfies `J_LN*1=0`: equal shifts of every input feature have no local effect. RMSNorm generally responds to this direction. At a zero-mean point with unit gains, `J_RMS-J_LN=11^T/(d*r)`, explaining the table's different derivatives.

For **unit gains**, the derivative magnitudes along characteristic directions are:

| Direction | LayerNorm | RMSNorm |
|---|---|---|
| Common offset |0|Not generally an eigenvector; depends on x|
| Radial, when nonzero: c for LN, x for RMS |`epsilon/r^3`|`epsilon/r^3`|
| Orthogonal to offset and c for LN; orthogonal to x for RMS |`1/r`|`1/r`|

At epsilon0, the radial derivative vanishes where defined. At positive epsilon, it generally remains nonzero. At x=0 and epsilon1e-5, RMSNorm's Jacobian is `316.227766*I`; LayerNorm's is `316.227766*P`. Zero output therefore does **not** imply zero gradient. Nonuniform gains change this eigenvalue description; their Jacobians were checked separately.

## The residual path still matters

For the illustrative branch `y=x+LN(x)`, adding3 to every feature changes y by3 in every feature. Normalization removes the shift from its branch, while the skip path carries it. In the corresponding RMSNorm example, changes are `[3.34755,3.16057,2.97360,2.78663]`.

This is a local residual example. It does not establish that a whole decoder preserves offsets, nor that RMSNorm improves gradient flow during training. The earlier [gradient-flow study](gradient-flow-results.md) remains a separate trained observation.

## Verification, sources and closure

| Check | Maximum discrepancy |
|---|---:|
| Explicit versus native output |4.44e-16 absolute|
| Analytical versus autograd Jacobian |3.50e-16 normalized|
| Float64 finite-difference Jacobian |6.84e-9 normalized|
| Independent70-digit Decimal Jacobian |2.63e-16 normalized|

Normalized errors divide by `max(1,max|J|)`. The auditor checks30,464 scalar comparisons and the shift/scale counterexamples. The initial finite-difference step failed four constant-LayerNorm checks. Capping it by the normalization radius resolves the verification failure without relaxing the threshold. [Archived failed sources and manifest](../../../results/rmsnorm_setup_failure/manifest.json).

[Ba et al.2016, section5](https://arxiv.org/html/1607.06450v1) analyzes LayerNorm invariances. [Zhang and Sennrich2019, sections4–4.2](https://arxiv.org/html/1910.07467v1) defines RMSNorm and analyzes idealized scaling/gradients without epsilon in those formulas. Our epsilon-dependent results qualify that idealization; they do not refute the paper. Its reported training/speed benefits are untested here. [PyTorch2.14 RMSNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html) supplies the implemented epsilon/gain convention; epsilon is explicit in every test.

Evidence: [results](../../../results/rmsnorm_math.json), [independent audit](../../../results/rmsnorm_math_audit.json), [runner](../../../experiments/rmsnorm_math.py), [auditor](../../../experiments/audit_rmsnorm_math.py).

**Mathematical study closed.** Centering changes information and derivatives. Whether removing it improves accuracy, convergence or runtime requires a separate controlled comparison; no automatic training or timing sweep follows.
