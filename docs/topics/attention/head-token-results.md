# Where does the routing rescue act?

**Final-query Q plus attention K at value-token positions is sufficient for strong rescue in this test.** Both heads are patched together. Neither head alone passes the every-seed criterion.

## Learn the indexing

Our eight-token crop is `[k1,v1,k2,v2,k3,v3,marker,query]`. Positions are 0..7. Symbolic key tokens occupy 0,2,4. Value tokens occupy 1,3,5.

Every token produces a learned attention Q, K and V. Therefore even a value token has an attention K. These names describe different things: the task's key/value symbols versus the attention operation's vectors.

For the final query, a source score is q_7 dot k_j / sqrt(8). Replacing Q at position 7 changes the reader. Replacing K at position j changes that source's score. We leave attention V shifted throughout. This is not copying literal answer values.

In block 2, only Q row 7 can affect the final attention row. The remaining FFN, normalization and output projection are position-wise. There is no third attention block to mix other output rows back in. Thus replacing other Q rows cannot affect the final prediction when final Q, K and V stay fixed. We verified this expected property; it is not evidence of a newly discovered circuit.

## Method

Follow the [fixed protocol](../../head-token-protocol.md). Five frozen models, all 5,120 inspected six-derived crops each, compact matching donors into globally shifted +4 recipients. Each crop still contains only three pairs. No training. Head slices are dimensions 0..7 and 8..15 before concatenation. For source-group tests, patch final Q in both heads and K only at the listed slots. Use the verified explicit attention operation from [Q/K/V testing](qkv-results.md).

## Head results

Mean accuracy/loss. Head conditions replace final Q and all K within that head; the other head stays shifted. Baseline accuracy is 60.60%.

| Replacement | Accuracy | Cross-entropy | Every seed >=90% |
|---|---:|---:|---|
| Head 0 only | 90.19% | 0.4378 | No |
| Head 1 only | 82.03% | 0.8557 | No |
| Both heads | 99.15% | 0.0330 | Yes |

| Seed | Head 0 | Head 1 | Both |
|---|---:|---:|---:|
| 137 | 99.45% | 76.09% | 100% |
| 163 | 88.34% | 86.50% | 99.45% |
| 191 | 91.80% | 79.20% | 99.36% |
| 223 | 84.98% | 83.81% | 99.73% |
| 257 | 86.37% | 84.53% | 97.19% |

Head labels do not guarantee identical learned roles across seeds. A mean above 90% does not mean every seed passed. These sufficiency tests do not establish that both heads are necessary for every example.

## Source-position results

All rows below replace final Q in both heads, plus the specified K vectors. All attention V stays shifted.

| K sources replaced | Mean accuracy | Cross-entropy | Every seed >=90% |
|---|---:|---:|---|
| None, Q-only control | 63.57% | 2.0033 | No |
| Symbolic key slots 0,2,4 | 60.86% | 2.2589 | No |
| Value slots 1,3,5 | 98.66% | 0.0502 | Yes |
| Marker/query slots 6,7 | 60.20% | 2.2873 | No |
| Queried pair's key slot only | 60.53% | 2.2542 | No |
| Queried pair's value slot only | 89.96% | 0.4503 | No |
| All six pair slots | 99.22% | 0.0326 | Yes |
| All eight slots | 99.15% | 0.0330 | Yes |

Value-slot rescue ranges from 96.74% to 99.92% across seeds. Target-value-only rescue ranges from 84.63% to 93.01%; two seeds miss 90%. Improving one source is not equivalent to restoring the full competition among sources. Softmax normalizes all scores together. This offers a plausible explanation for the difference, not a separately verified mechanism.

K-only at every source, without Q replacement, remains 92.74% and fails one seed. Replacing non-final Q plus all K gives exactly the same final logits as K-only. Replacing final Q plus all K gives exactly the same final logits as replacing every Q plus every K. Both structural controls had zero error for every seed.

## Interpretation and limits

Attention K at value-token positions is a strong intervention site. By block 2, those vectors can contain earlier context, including information about preceding keys. We have not yet measured what they encode or shown how block 1 constructs it. A successful patch is not a complete explanation.

All source-group results are conditional on compact final Q and both-head replacement. They do not prove the K subset alone suffices. Query-aware source selection, unnatural mixed activations, reused data and dependent crops remain limitations. No architecture improvement or new paper benchmark is established; see the [existing mathematical source comparison](qkv-results.md#learn-the-operation).

## Evidence and next question

[Script](../../../experiments/head_token_paths.py), [raw results](../../../results/head_token_paths.json) and [input artifact](../../../results/subset_confirmation_splits.json) preserve all 13 conditions per seed. Prior matched Q/K/QK metrics reproduced within 1e-10. Structural equalities, label checks, finite logits and unchanged checkpoints passed. Source hashes and paired rescue/harm arithmetic were checked separately.

Follow-up completed: [target versus other value-position K controls](target-distractor-results.md). Both groups contribute under fixed compact Q; target-only restoration helps more, but only restoring all three meets the every-seed rescue criterion. Block-1 construction remains unresolved.
