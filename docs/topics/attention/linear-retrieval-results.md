# Learned retrieval: softmax wins this comparison, reliability still fails

Executed2026-10-03 and closed. Six completed runs, no restarts or tuning. [Educational guide](linear-retrieval-guide.md) | [Frozen protocol](../../protocols/linear-retrieval.md).

**Six-pair balanced mean:95.29% softmax versus33.16% positive-kernel attention.** All three paired gains exceed the frozen10-point margin. This is a result for this recipe and budget, not a universal architecture ranking.

## What was held fixed?

Both models have4,544 parameters, two width16 pre-norm blocks, two query heads sharing one KV head, RoPE and ReLU FFNs. Complete initial weights and minibatch streams match within each seed. Only the attention similarity changes: scaled-dot softmax versus normalized ELU+1 feature products.

Each run receives6,000 Adam updates, batch64, lr0.001:384,000 examples/final-query targets and3,456,000 input tokens. CPUfloat64, one thread. A complete optimizer step was profiled before freezing the choices. GPU unused, allocated/reserved GPU bytes0; CPU peak memory was not measured. Both training implementations use dense pairwise arithmetic, so elapsed times do not benchmark an efficient recurrent kernel.

Data reuse the earlier MQA family splits, expanded to every query key. Train/validation contain3/4 pairs; tests contain3/4/6. These previously examined data are not fresh confirmation. The target is the latest value assigned to the query key. Full training, validation and test pools are evaluated.

## Did the models learn the shorter task?

R/U means repeated-key accuracy / unique-key accuracy. All entries are percentages. Four-pair results expose the principal fitting failures; [complete per-seed accuracy and loss](../../../results/linear_retrieval_metrics.csv) includes every3/4/6-pair pool and both groups.

| Seed | Rule | Train4 R/U | Validation4 R/U | Test6 R/U |
|---|---|---:|---:|---:|
| 2591 | Softmax | 99.97 /100.00 | 99.92 /100.00 | 99.69 /98.70 |
| 2591 | Linear | 80.51 /79.97 | 80.61 /79.88 | 53.65 /57.78 |
| 2609 | Softmax | 100.00 /99.98 | 100.00 /99.99 | 99.81 /95.28 |
| 2609 | Linear | 42.09 /25.68 | 41.17 /25.28 | 26.76 /16.43 |
| 2621 | Softmax | 98.78 /99.77 | 98.69 /99.75 | 92.00 /86.26 |
| 2621 | Linear | 43.93 /25.93 | 44.03 /25.42 | 27.61 /16.70 |

All linear seeds fail the99% short-fit gate. Softmax seed2621 also fails it on repeated queries. Therefore a pure length-transfer comparison is ineligible. Longer-input errors coexist with incomplete short-task learning.

## Paired longer-input comparison

The balanced score gives equal weight to R and U, despite their different numbers of examples.

| Seed | Softmax test6 | Linear test6 | Softmax gain |
|---|---:|---:|---:|
| 2591 | 99.20% | 55.72% | 43.48 points |
| 2609 | 97.55% | 21.60% | 75.95 points |
| 2621 | 89.13% | 22.16% | 66.97 points |
| Mean | 95.29% | 33.16% | 62.13 points |

Both all-seed reliability rules fail. Softmax is substantially better here, but its weakest unique-query test6 group scores86.26%, below the90% requirement. Its mean hides that miss.

Baseline test6 R/U: constant value16.67%/16.67%; first match0%/100%; final context value34.83%/16.29%; exact last-match oracle100%/100%. First-match success on unique queries cannot establish successful updating.

## Verification

| Check | Result |
|---|---|
| Initial whole-model outputs and all parameter gradients | Six cases pass1e-10; maximum gradient error1.67e-16 |
| Causal prefix and batch isolation | Six cases pass |
| Intentional missing denominator | Detected in each linear seed |
| Independent labels/families and shifted-label control | Pass |
| Final independent evaluation | 42 pools,1,023,840 predictions; zero argmax changes |
| Maximum final logit / metric discrepancy | 1.83e-13 /8.89e-16 |
| Pairing and chronology | Initial hashes, sampled streams,6,000 optimizer steps and pretest checkpoint hashes verified |
| Earlier evidence | All258 prior JSON artifacts unchanged |

The independent model uses explicit normalization/projection equations and recurrent S/z for linear attention, while training uses pairwise weights. This supports implementation correctness within the checked scope; a million predictions are not a million independent training runs.

## What remains unexplained?

The [original paper](https://proceedings.mlr.press/v119/katharopoulos20a.html), sections3.2-3.4, derives the feature-kernel recurrence. Our earlier math audit and this prediction audit agree with that identity. These experiments do not reproduce its trained benchmarks or contradict the identity.

The cause of poor fitting remains open. This objective, initialization, budget and positional construction are candidate factors. In particular, applying phi after RoPE changes the usual relative-position dot-product identity. The scalar state-collision witness does not prove that collisions caused these trained errors. Nor does this result cover gated memories, other feature maps or other linear-attention designs.

**Decision:** retain softmax for this retrieval recipe. Close the comparison at its declared budget, preserving both incomplete fitting and failed reliability. No automatic tuning, replacement seeds or extension.

Evidence: [training source](../../../experiments/linear_retrieval.py), [independent audit source](../../../experiments/audit_linear_retrieval.py), [raw report](../../../results/linear_retrieval.json), [audit report](../../../results/linear_retrieval_audit.json), [step profile](../../../results/linear_retrieval_profile.json).
