# Query-position effects on attention versus the residual path

Live execution completed on 2026-09-29. **For seed 47, a residual-only position patch rescued all fresh-set errors; an attention-only patch did not.** Several donor positions rescued it, so the effect is not specific to the exact training position.

## Method: separate the two paths

The query position affects both attention and the vector added around attention. Separating them tests which intervention can rescue predictions.

\[
x=E[\mathrm{query}]+PE(10),\quad x'=E[\mathrm{query}]+PE(r),
\qquad u=x+\operatorname{Attention}(\operatorname{LN}(X))_{query}.
\]

| Patch | What changes | What stays fixed |
|---|---|---|
| Q only | Query vector projected from LN(x') | Keys, values and query residual |
| K/V only | Query token's key and value projected from LN(x') | Q, symbol keys/values and residual |
| Attention only | Query token's Q/K/V use LN(x') | All symbol rows and query residual x |
| Residual only | Add x' instead of x | Every attention weight and value |
| Both | Attention and residual patches together | Symbol representations and all weights |

After each patch, FFN, final normalization and vocabulary projection run normally. The residual patch therefore tests the residual input **and its downstream consequences**, not a direct linear effect on logits.

## Evidence design

Fresh data checks whether the earlier rescue survives new examples. We generated 1,024 unique length-ten sequences with seed 20261001, excluding the 512 previously inspected sequences. All five frozen checkpoints were tested in CPU float64. No training, model selection or architecture changes occurred.

The [protocol](../../query-position-protocol.md) fixed donor positions 0, 5, 6, 7 and 10 before execution. The combined position-six rescue was a confirmation hypothesis; pathway decomposition was exploratory. The success threshold was at least 95% for every seed. These examples share the same synthetic task distribution, not a new task.

## Results: donor position six

Accuracy measures correct first-symbol predictions. Values below are percentages on the same fresh examples for every intervention.

| Seed | Original | Q only | K/V only | Attention only | Residual only | Both |
|---|---:|---:|---:|---:|---:|---:|
| 11 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 29 | 97.27 | 99.41 | 91.60 | 98.83 | 100.00 | 100.00 |
| 47 | 79.49 | 81.25 | 76.37 | 78.22 | 100.00 | 100.00 |
| 73 | 99.71 | 100.00 | 97.75 | 100.00 | 100.00 | 100.00 |
| 101 | 96.78 | 100.00 | 97.56 | 100.00 | 98.63 | 100.00 |

Combined rescue confirmed: 100% for all five seeds. Residual-only passed the 95% criterion for all seeds, but was not perfect for seed 101: it corrected 22 errors and broke 3 formerly correct predictions. Attention-only, Q-only and K/V-only each failed the all-seed criterion.

## Seed 47: paired outcomes and loss

Corrected and broken counts distinguish a true net rescue from merely exchanging which examples fail. Cross-entropy measures probability quality as well as the winning prediction.

| Condition | Cross-entropy (nats) | Original errors corrected | Original correct predictions broken |
|---|---:|---:|---:|
| Original | 0.353092 | 0 | 0 |
| Q only | 0.284767 | 22 | 4 |
| K/V only | 0.435445 | 0 | 32 |
| Attention only | 0.335473 | 6 | 19 |
| Residual only | 0.002031 | 210 | 0 |
| Both | 0.001913 | 210 | 0 |

Seed 47's residual-only rescue preserves its original attention exactly. Thus changing attention routing is not necessary to rescue these observed errors through this intervention.

## Positional controls: is six special?

Changing the donor position tests whether recovery uniquely depends on restoring the training position. These controls are not matched for perturbation magnitude.

| Donor position, seed 47 | Attention-only accuracy | Residual-only accuracy | Combined accuracy |
|---|---:|---:|---:|
| 0 | 73.73% | 100% | 100% |
| 5 | 78.91% | 100% | 100% |
| 6 (training query) | 78.22% | 100% | 100% |
| 7 | 78.71% | 100% | 100% |
| 10 (no change) | 79.49% | 79.49% | 79.49% |

The exact training position is not uniquely required for this rescue. We have not established which positional components or downstream features cause the sensitivity, or whether all alternative positions would help.

## Verification and limits

No-change patches matched original logits within 6.440e-15. Combined position-six patches matched the previous explicit implementation within 5.829e-15. Checkpoint hashes were verified before loading; hooks were removed after every intervention. These checks validate patch equivalence at the tested inputs, not universal correctness.

The earlier [paper comparison](../../assessments/evidence-summary.md) still applies: these are our synthetic-task interventions, not a reproduction of published language-model results. The effects demonstrate causal sensitivity to specified internal replacements. They do not prove that positional encoding is generally defective, that the residual is unnecessary, or that we have recovered the entire learned mechanism.

## Reproduce and next question

Run `.venv/Scripts/python.exe experiments/query_position_paths.py`. [Code](../../../experiments/query_position_paths.py) and [raw results](../../../results/query_position_paths.json) record all 125 intervention settings, paired metrics, source/protocol/data hashes and checkpoint hashes.

Next question: which downstream computation converts the changed residual into wrong logits? The current evidence justifies tracing the FFN and final normalization paths, not changing the architecture. Additional investigations must distinguish exploratory analysis from new confirmation data.
