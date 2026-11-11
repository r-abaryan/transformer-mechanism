# Final length check for the current recipe

Canonical positions retain near-perfect three-pair accuracy, but do not establish reliable length transfer. Four-pair mean accuracy is 92.20%; one seed fails 90%. Six-pair mean accuracy is 65.38%; every seed fails 90%. As agreed, this closes the length check without more patches or retraining.

## Method and what changed

Five frozen varied-position models, canonical start +4, one ordinary forward, no donor. Three/four/six pairs give 8/10/14 tokens. Increasing pairs adds distractors and moves the marker/query; it also changes which token roles occupy positions. Canonicalization fixes the starting offset, not those differences.

The [protocol](../../final-length-protocol.md) was saved before generation. Each length has 420 unique whole mappings with distinct keys and values, six sampled orders and every query per order. Sequence counts are 7,560/10,080/15,120. Labels and queried ranks balance exactly. The three-pair set excludes all 5,600 prior expanded-task mappings. Longer whole mappings are new in this task vocabulary, but can contain familiar submappings. Only six orders per longer mapping are sampled, not all permutations.

## Findings

| Pairs / tokens | Mean accuracy | Mean loss, nats | All sampled variants correct | Predeclared all-seed criterion |
|---|---:|---:|---:|---|
| 3 / 8 | 99.98% | .0017 | 99.57% | >=99%: passed |
| 4 / 10 | 92.20% | .4057 | 23.00% | >=90%: failed |
| 6 / 14 | 65.38% | 2.1783 | 0.00% | >=90%: failed |

Mapping correctness requires every sampled order/query variant to succeed. There are 18/24/36 variants per mapping; these are dependent and the number grows with length, so this stricter metric is not directly comparable across lengths.

| Seed | Three pairs | Four pairs | Six pairs |
|---|---:|---:|---:|
| 461 | 99.97% | 95.03% | 70.10% |
| 487 | 99.91% | 95.69% | 70.15% |
| 509 | 100% | 93.25% | 69.54% |
| 541 | 100% | 85.84% | 55.92% |
| 571 | 100% | 91.18% | 61.19% |

The queried rank is the location of the requested pair in the input, counted from one. Errors concentrate later:

| Pairs | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 | Rank 6 |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 99.98% | 99.98% | 99.97% | n/a | n/a | n/a |
| 4 | 98.93% | 98.82% | 94.65% | 76.40% | n/a | n/a |
| 6 | 98.47% | 97.21% | 90.11% | 59.83% | 27.59% | 19.07% |

This pattern is descriptive. It does not isolate attention, FFN, interference or position encoding as the cause.

## Relation to earlier evidence

[Earlier depth-confirmation models](depth-confirmation-results.md) already failed length transfer: means 80.43% on four pairs and 64.24% on six. The present means are 92.20% and 65.38%, but these are not controlled improvement estimates. Vocabulary, repeated-versus-distinct values, mapping holdout, seeds, position training and evaluation positions differ. The new finding concerns the current recipe; it is not a first discovery of length failure.

Conclusion: resetting an irrelevant global offset is useful but insufficient for longer inputs. Both earlier and current bounded tests fail their all-seed transfer criterion. Stop this branch here and begin the [FFN investigation](../ffn/ffn-investigation.md) as a separate question, not a presumed explanation for these failures.

## Evidence

[Script](../../../experiments/final_length.py), [reserved inputs](../../../results/final_length_inputs.json), [results](../../../results/final_length.json), [full logits](../../../results/final_length_scores.json), [audit](../../../results/final_length_audit.json). Mapping isolation, labels, sampled orders, query coverage, balance and unchanged hashes passed. Explicit/library logits agree within 1.57e-13. All 15 condition summaries, losses, margins and queried-rank metrics reproduce exactly from saved logits. Checks share PyTorch; no independent implementation or paper benchmark reproduction is claimed.
