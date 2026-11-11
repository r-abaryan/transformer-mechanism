# FFN branches: tested contribution and math

Concrete answer for these five models: block 1's FFN has the larger effect on ordinary task accuracy. Block 2's FFN improves prediction quality and is sensitive to mismatched outputs, even though removing it preserves most correct answers. Neither finding establishes universal FFN necessity or semantic fact storage.

## Fixed method

Five frozen varied models, 7,200 already-opened three-pair examples, canonical +4. Zero each entire FFN branch or both; separately permute each branch's outputs across examples, preserving token positions. Downstream computation stays live. The same saved random permutation is used for all models; it has two fixed points and 15.54% answer overlap. Permutation preserves the branch-output sample distribution but not compatibility with the recipient context. [Protocol](../../ffn-branch-protocol.md).

## Results

| Condition | Mean accuracy | Loss, nats | Mean correct-answer margin |
|---|---:|---:|---:|
| Ordinary / unchanged-output sham | 99.978% | .0018 | 8.10 |
| Zero block 1 FFN | 85.36% | .5669 | 3.88 |
| Zero block 2 FFN | 99.09% | .0421 | 6.12 |
| Zero both | 80.78% | .6803 | 2.87 |
| Permute block 1 FFN | 82.69% | .7606 | 3.88 |
| Permute block 2 FFN | 91.52% | .2364 | 4.44 |

| Seed | Zero block 1 | Zero block 2 | Zero both | Permute block 1 | Permute block 2 |
|---|---:|---:|---:|---:|---:|
| 461 | 88.43% | 99.97% | 88.13% | 91.00% | 99.43% |
| 487 | 85.11% | 99.85% | 86.63% | 81.08% | 93.97% |
| 509 | 87.88% | 99.92% | 80.31% | 86.57% | 84.78% |
| 541 | 80.81% | 96.92% | 71.29% | 79.08% | 88.92% |
| 571 | 84.57% | 98.81% | 77.56% | 75.72% | 90.49% |

Block 2 is not irrelevant: removing it worsens loss in every seed, and shuffled outputs can damage accuracy substantially more. Donor-answer agreement is 15.83% for block-1 permutation and 23.46% for block 2, versus 15.54% label overlap. This is not proof of wholesale answer copying or a particular stored fact.

Effects are not additive. For seed 487, removing both gives higher accuracy than removing block 1 alone. Zeroing is also outside normal activation distributions. These are intervention sensitivities in the trained network, not performance estimates for a separately trained FFN-free architecture.

## Math challenged

For normalized input z, f(z)=W2 ReLU(W1 z+b1)+b2. W1 has shape 32x16; W2 is 16x32. Equivalently f=sum_j h_j v_j+b2, where v_j is column j of W2.

| Check | Measured outcome | What it establishes |
|---|---|---|
| Sum of activated output directions versus FFN output, all tokens | Maximum error 2.67e-15 | Numerical agreement with the algebraic identity |
| J=W2 diag(W1 z+b1>0) W1 versus central directional differences | Maximum error 3.03e-11 over ten layer/seed checks | Local derivative at tested points with unchanged ReLU gates; excludes LayerNorm derivative and kinks |
| Claim that logit change equals W_out f | Mean absolute discrepancies .296 to .629 across seeds; maximum 7.19 | Refutes that exact shortcut here; final normalization changes the effect |

The actual final logit change is W_out[LN(u+f)-LN(u)], not generally W_out f. The next [normalization test](ffn-normalization-results.md) was executed to separate these terms exactly.

ReLU active fractions average 52.16% and 54.51% in blocks 1/2. Mean output norms are 2.28/3.27; incoming post-attention residual norms 5.42/7.22. These are descriptive: the larger block-2 output does not have the larger zero-ablation accuracy effect. Magnitude alone is not importance.

## Interpretation, sources and evidence

The [FFN learning guide](ffn-investigation.md) links the Geva et al. primary papers and distinguishes their language-model memory interpretations from our algebra and toy-task evidence. The weighted-direction identity supports a representation of the computation; it does not identify one neuron with one fact. No paper benchmark or full mechanism was reproduced.

[Script](../../../experiments/ffn_branches.py), [report](../../../results/ffn_branches.json), [full logits](../../../results/ffn_branches_scores.json), [permutation](../../../results/ffn_branches_permutation.json), [audit](../../../results/ffn_branches_audit.json). Library baseline, shams, finite outputs and immutable hashes passed. All 35 condition summaries reproduce exactly from saved logits. The separate audit checks recorded math bounds, but does not independently rerun the math checks. Same PyTorch implementation; exploratory opened data, not confirmation.

Close this branch-level question without neuron searches. The next step, [final normalization](ffn-normalization-results.md), explains why raw FFN-to-logit projections are insufficient.
