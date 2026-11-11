# Skip correction: confirmation across lengths

**Both predefined criteria failed.** Seed 47's rescue extended to all tested lengths, but the correction did not reliably rescue every model. No training or tuning followed these results.

## Method

This confirmation tests the earlier finding beyond one length and dataset. Five frozen checkpoints were evaluated on 512 new examples at each of lengths 8, 10, 12 and 16. Previously inspected length-ten examples were excluded. The [protocol](../../skip-confirmation-protocol.md) was fixed before execution.

With original attention held fixed, u is the post-attention residual, f(u) the normalized FFN, and delta=PE(6)-PE(L):

| Condition | Vector entering final normalization |
|---|---|
| Original | u+f(u) |
| Skip correction | u+delta+f(u) |
| Full residual correction | u+delta+f(u+delta) |
| Direction control | u+c+f(u), where norm(c)=norm(delta) and c is orthogonal to delta |

One predetermined control direction was used per length, shared across seeds. This matches magnitude but does not estimate effects over many random directions. Final normalization and vocabulary projection remain unchanged.

## Accuracy for every seed and length

Percentages below are final prediction accuracy. Per-condition loss and paired corrected/broken counts are in the [raw report](../../../results/skip_confirmation.json).

| Length | Seed | Original | Skip | Full residual | Control |
|---:|---:|---:|---:|---:|---:|
| 8 | 11 | 100.00 | 100.00 | 100.00 | 100.00 |
| 8 | 29 | 92.38 | 100.00 | 100.00 | 97.27 |
| 8 | 47 | 81.25 | 100.00 | 100.00 | 78.12 |
| 8 | 73 | 100.00 | 99.41 | 100.00 | 100.00 |
| 8 | 101 | 97.66 | 99.41 | 99.02 | 99.41 |
| 10 | 11 | 100.00 | 100.00 | 100.00 | 100.00 |
| 10 | 29 | 97.46 | 100.00 | 100.00 | 76.76 |
| 10 | 47 | 79.88 | 100.00 | 100.00 | 78.32 |
| 10 | 73 | 99.80 | 100.00 | 100.00 | 98.05 |
| 10 | 101 | 97.46 | 99.02 | 98.63 | 99.02 |
| 12 | 11 | 74.80 | 100.00 | 100.00 | 78.32 |
| 12 | 29 | 33.79 | 48.24 | 48.24 | 48.24 |
| 12 | 47 | 100.00 | 100.00 | 100.00 | 100.00 |
| 12 | 73 | 98.44 | 100.00 | 100.00 | 96.68 |
| 12 | 101 | 73.44 | 75.39 | 78.71 | 88.28 |
| 16 | 11 | 69.14 | 100.00 | 100.00 | 72.27 |
| 16 | 29 | 50.39 | 67.19 | 77.73 | 49.02 |
| 16 | 47 | 74.41 | 100.00 | 100.00 | 72.85 |
| 16 | 73 | 44.73 | 98.63 | 99.22 | 54.30 |
| 16 | 101 | 41.41 | 57.23 | 54.10 | 41.21 |

## Criteria and findings

The thresholds were fixed beforehand so a favorable average cannot replace a failed requirement.

| Criterion | Result |
|---|---|
| Skip accuracy at least 95% for every seed and length | Failed: seeds 29 and 101 at lengths 12 and 16 |
| Mean skip-minus-control gain at least 5 percentage points at every length | Failed: 4.80 points at length 8; 2.42 at length 12 |
| Other specificity measurements | 9.38 points at length 10; 26.68 at length 16 |

Seed 47's skip correction reached 100% at all tested lengths. However, its original model already scored 100% at length 12: length-dependent failure is not monotonic. Seed 101 at length 12 benefited more from the control than from the positional correction. These observations rule out presenting the intervention as a universally effective repair under our own tested conditions.

## Evidence boundary and stopping decision

The confirmed narrow finding is that this fixed correction rescues seed 47 on these new sampled lengths where errors occurred. The proposed extension to all models failed. This does not identify the remaining models' failure mechanisms, establish behavior on all sequences, or justify an architectural change. Comparisons with papers remain [scope comparisons](../../assessments/evidence-summary.md), not reproduced benchmarks.

Stop extending this correction as a general explanation. Preserve the negative result and carry the confirmed seed-specific finding forward. A next independent task would test whether the research methods and any mechanism findings extend beyond first-token retrieval; no task or model should be selected merely to make this correction succeed.

## Reproduce

Run `.venv/Scripts/python.exe experiments/confirm_skip.py`. [Code](../../../experiments/confirm_skip.py) records source/protocol/data/checkpoint hashes, all 80 condition results, control vectors and reconstruction errors. CPU float64; five seeds; 2,048 new sequences shared across models. Baseline reconstruction, equal-norm and orthogonality assertions passed. Timing, GPU performance and population uncertainty were not assessed.
