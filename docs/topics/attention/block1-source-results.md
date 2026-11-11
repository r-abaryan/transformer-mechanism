# Does block 1 read the paired key?

**Paired-key source interventions strongly affect the controlled downstream task.** Forcing the tested value positions to read their paired keys rescues every seed. Self and other-key forcing do not.

## Short explanation

The crop is `[k1,v1,k2,v2,k3,v3,marker,query]`. A value token's paired key is immediately before it. We ask whether reading that key helps build the representation used later for retrieval.

We inspect attention, then change its source weights. These are different kinds of evidence. A large weight is an observation. A change in predictions after replacing the weights is an intervention result.

We intervene at value positions 3 and 5, in both heads. Position 1 stays unchanged because it has no earlier nonpaired key for a fair causal control. All candidate sources remain at or before the reading position.

## Method and math

The [fixed protocol](../../block1-source-protocol.md) uses five frozen models and 5,120 inspected crops per model. No training. Reconstruct block-1 attention as A=softmax(QK^T/sqrt(8)+M), then compute Concat(AV)W_O^T+b_O. For forcing, replace the selected row of A with a one-hot row. For removal, exclude one source and recompute softmax over the rest.

Paired-key forcing reads row-1. Self reads row. Other-key forcing reads row-3, an earlier nonpaired key. Keep the chosen compact/shifted state's V and output projection unchanged. The prior downstream procedure stays fixed: build candidate y from shifted residual + intervened attention + saved shifted FFN; project block-2 K at value slots only. Final Q remains compact; other recipient paths stay shifted.

## Attention observations

Means over seeds, crops and the two tested value rows. Entries are probabilities. Head labels need not represent identical roles across seeds. Other tokens account for the remaining probability.

| State / head | Paired key | Self | Earlier nonpaired keys, summed |
|---|---:|---:|---:|
| Compact / 0 | 0.694 | 0.095 | 0.143 |
| Compact / 1 | 0.458 | 0.218 | 0.170 |
| Shifted / 0 | 0.193 | 0.049 | 0.685 |
| Shifted / 1 | 0.168 | 0.144 | 0.496 |

Under the position shift, average weight moves away from the paired key toward earlier nonpaired keys. This describes a pattern. The intervention results below test its behavioral relevance.

## Intervention results

Accuracy / cross-entropy means across five seeds. “Compact” and “shifted” identify block-1 attention's source state, not the entire downstream model.

| Attention-row intervention | Compact source | Shifted source |
|---|---:|---:|
| Unmodified | 94.75% / 0.2408 | 63.57% / 2.0033 |
| Force paired key | 96.61% / 0.1527 | 97.68% / 0.0915 |
| Force self | 61.88% / 2.6408 | 63.83% / 2.4492 |
| Force earlier nonpaired key | 41.71% / 4.1732 | 46.75% / 3.7118 |
| Remove paired key | 53.95% / 3.3154 | 47.57% / 3.1592 |
| Remove self | 94.96% / 0.2297 | 63.53% / 2.0227 |

Paired-key forcing passed every-seed 90% for both states. Self and other-key forcing failed. Compact paired-key removal reduced accuracy by >=10 points in every seed; compact self removal did not. Shifted paired-key removal failed the every-seed 10-point harm criterion, despite a large average decline, because seed 223 fell by only 8.96 points.

| Seed | Shifted unmodified | Shifted force paired key | Compact remove paired key |
|---|---:|---:|---:|
| 137 | 70.23% | 99.67% | 60.90% |
| 163 | 59.32% | 98.34% | 53.42% |
| 191 | 65.20% | 99.12% | 52.03% |
| 223 | 55.25% | 95.90% | 48.73% |
| 257 | 67.83% | 95.37% | 54.67% |

## What this supports

The paired-key source contributes useful information to the tested value-position representations. The observation and intervention results are consistent with a local key/value association step in block 1. They do not yet prove what is encoded or that this is the only mechanism.

Forcing changes the entire attention row; removal redistributes probability. Controls are not matched for perturbation magnitude. Only two value rows were changed, both heads together. Some nonpaired keys can share the correct answer value. Final Q is still supplied from the compact run, and the downstream key-construction mixture is artificial. Therefore 97.68% is not an unmodified-model accuracy or an end-to-end repair. Data are reused and crops dependent. No architecture change or paper benchmark is claimed; the [attention definitions](qkv-results.md#learn-the-operation) remain applicable.

## Verification and next question

[Script](../../../experiments/block1_sources.py) and [raw report](../../../results/block1_sources.json) preserve all 12 conditions per seed and per-position/head observations. Explicit block-1 attention matched the library within 2.45e-15. Unmodified constructions reproduced prior metrics within 1e-10. Causal zeros, row sums, labels, finite outputs and checkpoint integrity passed. Source hashes and paired-count arithmetic were checked separately.

Follow-up completed: [direct forcing without compact donors](../retrieval/direct-pair-results.md) raises mean shifted accuracy to 93.48%, but seed 223 fails the every-seed criterion. The controlled rescue therefore does not establish a reliable full-model repair.
