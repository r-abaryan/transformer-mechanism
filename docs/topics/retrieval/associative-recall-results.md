# Associative recall: results and diagnostic limits

**The unchanged one-block baseline did not learn reliable associative recall under the fixed recipe.** All five seeds failed the 90% criterion. An untouched episode check confirmed performance below a simple value-frequency baseline. No tuning or architectural changes followed.

## Task and method

Associative recall requires retrieving the value of a queried key whose position varies. Example: `[A,3,B,1,C,2,query,B] -> 1`. Keys are unique; values may repeat. Only the final query-key output is supervised:

\[
L=-\frac1B\sum_b\log p_\theta(v(q_b)\mid k_1,v_1,\ldots,k_m,v_m,\mathrm{query},q_b).
\]

One block, width 16, two heads, FFN width 32, pre-norm, sinusoidal positions, no dropout. Only vocabulary size changed from 5 to 13, giving **2,672 parameters**. Each seed used 3,000 SGD updates at learning rate 0.1, batch 64: 192,000 sampled examples and 1,536,000 input tokens. CPU float64; final checkpoints only. This budget differs from the earlier task, so cross-task scores are not a controlled architecture comparison.

Three-pair splits: 8,192 train, 1,024 validation, 1,024 test. Separate four-/six-pair tests contain 1,024 examples each. Canonical mapping/query identities are disjoint, including reordered duplicates. Individual mappings with different queries and individual key/value associations can occur across splits. See [fixed training protocol](../../associative-recall-protocol.md).

## Training and test results

Training accuracy measures whether even the training episodes were mastered. It prevents mislabeling this result solely as a generalization failure.

| Seed | Train, 3 pairs | Test, 3 pairs | Test, 4 pairs | Test, 6 pairs |
|---|---:|---:|---:|---:|
| 11 | 56.03% | 54.10% | 47.27% | 44.14% |
| 29 | 55.92% | 54.30% | 45.70% | 44.53% |
| 47 | 54.11% | 52.64% | 48.83% | 44.24% |
| 73 | 56.03% | 55.08% | 44.04% | 45.21% |
| 101 | 55.96% | 54.69% | 50.78% | 45.12% |
| Mean | 55.61% | 54.16% | 47.32% | 44.65% |
| Sample SD, percentage points | 0.84 | 0.93 | 2.63 | 0.50 |

Mean test losses: 0.811738, 1.090545 and 1.385153 nats for three/four/six pairs. Both the same-size learning criterion and longer-context criterion failed. No final evaluation produced an invalid key/marker prediction; low accuracy is not explained by predicting outside the value vocabulary.

## Why above-chance accuracy is insufficient

Repeated values allow shortcuts that ignore the queried key. Choosing the first stored value has expected accuracy 1/3+(2/3)(1/4)=50% with three independently sampled values. Uniform guessing is only 25%.

The most-frequent-stored-value baseline was added **post hoc**, before diagnostic execution. Ties choose the smallest value ID. For three uniform four-class values, probabilities of all-same, exactly-two-same, and all-distinct are 1/16, 9/16 and 6/16. Its expected accuracy is:

\[
\frac1{16}\cdot1+\frac9{16}\cdot\frac23+\frac6{16}\cdot\frac13=56.25\%.
\]

| Method | Three-pair test accuracy |
|---|---:|
| Uniform guessing, expected | 25.00% |
| Training-majority value | 22.95% |
| First stored value | 50.00% |
| Last stored value | 49.02% |
| Most frequent stored value, post hoc | 55.57% |
| Neural model, five-seed mean | 54.16% |
| Exact dictionary lookup | 100% |

Similar scores do not prove the model implements the frequency rule. They show that this accuracy level does not establish query-conditioned retrieval.

## Query and binding interventions

Changing the query tests whether predictions follow which key is requested. Rotating stored values tests whether predictions follow changed key/value bindings while the multiset of values stays fixed. See [diagnostic protocol](../../associative-diagnostic-protocol.md).

| Seed | Prediction changes when query changes the answer | Accuracy on those changed-answer queries | Prediction changes when value rotation changes the answer |
|---|---:|---:|---:|
| 11 | 18.64% | 40.32% | 11.01% |
| 29 | 20.13% | 41.17% | 8.03% |
| 47 | 18.77% | 41.04% | 13.47% |
| 73 | 15.73% | 41.68% | 6.09% |
| 101 | 22.85% | 41.36% | 16.71% |

Denominators: 1,545 changed-answer query cases and 772 changed-answer rotations. Low prediction-change rates support limited binding sensitivity, not complete query independence. A changing answer does not require an initially incorrect model to change its prediction to become correct, so change rate is a sensitivity measure, not accuracy.

These are exploratory derivatives of existing episodes. **1,557/3,072 all-query probes and 736/1,024 rotated probes overlap training identities.** They are not held-out generalization evidence. No claim of a formal one-layer limitation follows.

## Untouched episode confirmation

The finite three-pair universe has C(8,3)*4^3*3=10,752 canonical mapping/query identities. After the original splits, 512 were unused. We evaluated this entire complement in one predetermined pair order per identity, under a [separate protocol](../../associative-confirmation-protocol.md), without modifying models.

| Seed | Accuracy on untouched complement | Difference from frequency baseline |
|---|---:|---:|
| 11 | 57.62% | -2.34 points |
| 29 | 55.47% | -4.49 points |
| 47 | 56.05% | -3.91 points |
| 73 | 58.01% | -1.95 points |
| 101 | 56.25% | -3.71 points |

Frequency baseline: 59.96%. All seeds remained below 90% and below the stronger baseline, meeting the predicted negative-confirmation criteria. The complement is a finite unseen-episode set, not unseen mappings, all pair orders, or an independent task distribution.

## Interpretation and paper comparison

The failure occurs already on training episodes, so we cannot attribute it solely to longer context or the earlier positional-residual issue. Optimization, capacity and learning key/value binding remain unresolved alternatives.

In this one-block implementation, a value token's initial K/V vectors are computed from its own embedding and position; they have not yet incorporated its neighboring key through an earlier attention layer. This is an implementation observation that motivates investigating binding, not a proof that one block cannot solve the task. The [original transformer](https://arxiv.org/html/1706.03762v7) stacks layers; our experiment does not reproduce its architecture or translation results. Published retrieval comparisons in the [source register](../../reference/sources.md) use other tasks and scales, so their scores are not directly comparable.

## Artifacts and next decision

Training, diagnostics and confirmation completed with exit code 0. Source/protocol/split/checkpoint hashes and all curves are stored in [training results](../../../results/associative_recall_training.json), [diagnostics](../../../results/associative_recall_diagnostics.json), and [confirmation](../../../results/associative_recall_confirmation.json). Each of five final checkpoints is saved. Reproduce with:

```powershell
.venv/Scripts/python.exe experiments/train_associative_recall.py
.venv/Scripts/python.exe experiments/diagnose_associative_recall.py
.venv/Scripts/python.exe experiments/confirm_associative_recall.py
```

Rerunning training replaces generated checkpoints. The environment is pinned in `requirements-cpu.txt`; optional NumPy warnings did not prevent execution. No GPU benchmark, memory profile, optimizer comparison, architecture comparison or formal impossibility proof was performed.

Stop this fixed-recipe experiment here. Before a new architecture proposal, define a positive-control learning experiment that distinguishes a pipeline/optimization limitation from a binding limitation, with the stronger frequency baseline included from the outset. The existing negative result must remain part of that comparison.
