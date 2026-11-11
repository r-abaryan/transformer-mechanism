# Why seed 47 fails: frozen-model evidence

We inspected all five saved models on the original two test sets, then changed internal computations without training. The [protocol](../../intervention-protocol.md), [code](../../../experiments/inspect_first_token.py) and [raw results](../../../results/first_token_interventions.json) preserve methods and provenance. This is exploratory reuse of known test data.

## Method and reconstruction

Each head contributes y_h=sum_j a_hj v_hj. Its contribution from position j after output projection is c_hj=W_O,h(a_hj v_hj), using column-vector notation. We measured attention weights, entropy -sum_j a_hj log(a_hj), and norms of these projected contributions. A norm measures magnitude, not support for the correct answer.

Only the final query is reconstructed: in this one-block model it attends to normalized input representations, so earlier output rows are unnecessary. Explicit logits matched the library model within 1e-10 for every seed/split; seed 47's longer-set maximum error was 2.665e-15. Checkpoint and split hashes matched the training artifacts.

## Observation: high attention is insufficient

Inspection compares natural correct and incorrect predictions; this is correlation, not an intervention.

Seed 47 made 92 errors among 512 longer examples. Head indices are zero-based:

| Group | Count | Head 0 mean attention to first symbol | Head 1 mean attention to first symbol | Head 1 mean first-symbol contribution norm |
|---|---:|---:|---:|---:|
| Correct | 420 | 0.2170 | 0.9032 | 3.7373 |
| Incorrect | 92 | 0.1563 | 0.9214 | 2.3242 |

Head 1 attended strongly to the correct location even on failures. Attention weights alone therefore do not separate success from failure here. The lower contribution norm is an observation, not an established cause; these groups contain different input examples.

## Interventions: changing one computation

Forcing a head replaces its query attention row with a one-hot distribution. Removing a value contribution sets a_hj v_hj to zero without renormalizing. All other computations then run normally.

| Intervention, seed 47 at length ten | Accuracy | Errors corrected | Correct predictions broken |
|---|---:|---:|---:|
| Original | 82.03% | 0 | 0 |
| Zero head 0 | 73.63% | 0 | 43 |
| Zero head 1 | 21.29% | 0 | 311 |
| Force head 0 to first symbol | 100% | 92 | 0 |
| Force head 0 to last symbol, control | 84.38% | 33 | 21 |
| Uniform head 0 attention, control | 85.16% | 19 | 3 |
| Force head 1 to first symbol | 83.79% | 10 | 1 |
| Remove head 1 first-symbol contribution | 21.29% | 0 | 311 |
| Remove head 1 last-symbol contribution, control | 82.03% | 0 | 0 |

Head 1's first-symbol contribution matters to this model's correct predictions. Providing head 0 with direct access to the first symbol rescues the observed failures more effectively than the listed controls. This intervention supplies a known task-relevant location; it is not a learned solution or evidence that the natural model always follows that rule. Controls match operation type, not perturbation norm. All head variants and both lengths are retained in JSON, including results not displayed here.

## Query position diagnostic

The training query occurs at position six; the longer query occurs at position ten. We changed only its positional vector: x_query=E[query]+PE(10) became E[query]+PE(6), keeping every symbol and its position unchanged.

This restored **100% longer-set accuracy for every seed**. Seed 47's loss fell from 0.357077 to 0.001945, correcting all 92 errors without breaking a correct prediction.

This establishes sensitivity to the query's position-dependent representation. It does not isolate attention routing: the edit also changes the query residual, normalization, and its key/value representations. The evidence does not show whether extra distractors, score changes, residual changes or interactions explain the full original gap. Different position controls and pathway-specific patches remain untested.

## Relation to papers and next boundary

Attention and output projection follow the [original definitions](https://arxiv.org/html/1706.03762v7). Our observations illustrate why the [attention-explanation debate](../foundations/interpretability.md) requires more than a heatmap, but do not reproduce either paper's experiments.

The supported answer is narrow: **seed 47 depends strongly on head 1's first-symbol value contribution, while its longer-length errors can be rescued through head 0 routing or query-position replacement.** The full mechanism remains unresolved. Next isolate positional effects on the query residual versus Q/K/V, include other positional controls, and confirm predictions on fresh examples before architecture changes.
