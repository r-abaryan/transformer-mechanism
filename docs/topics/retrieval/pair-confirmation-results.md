# Fixed paired-key intervention across shifts

**The intervention helps, but does not provide reliable rescue across shifts.** Every seed passes 90% at +1/+2. The criterion fails at +4/+6/+8. No tuning followed.

## Short explanation

We kept the same rule: two later value positions read their paired keys in block 1. The first value position stays unchanged. Every later calculation runs normally. We tested the rule on crops from new parent examples and moved all position numbers together by different amounts.

Moving every position preserves distances. The test measures sensitivity to absolute positional layout, not an increase in token count. Every model input still has three pairs and eight tokens.

## Method and data limits

The [protocol](../../pair-confirmation-protocol.md) fixed five checkpoints, generator seed 20261201, 512 six-pair parents and shifts 0/1/2/4/6/8 before execution. Enumerate ten query-preserving crops per parent. No training, compact donors, new intervention sites or parameter selection. Self and earlier-nonpaired-key forcing use the same two rows and both heads.

All new parent mapping/query identities exclude the 2,560 earlier long identities. However, removing pairs creates smaller inputs that can recur. Of 5,120 crop instances, 1,956 have canonical identities present in earlier diagnostic crops, and 3,890 have identities in these models' training set. There are 4,056 distinct crop identities. These are canonical overlaps, not necessarily identical token order. This is fresh-parent testing with partial crop reuse, not globally untouched input confirmation or independent replication.

## Results

Mean accuracy across five seeds. “All crops correct” is the fraction of parent episodes whose ten crops all succeed, averaged across seeds. It is stricter than mean crop accuracy.

| Shift | Original | Paired key | Self control | Other-key control | Paired: all crops correct | Paired mean loss, nats |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 100% | 99.98% | 61.18% | 43.25% | 99.88% | 0.0008 |
| +1 | 91.36% | 99.66% | 61.40% | 44.20% | 97.54% | 0.0099 |
| +2 | 77.16% | 97.67% | 62.27% | 48.41% | 89.61% | 0.0973 |
| +4 | 60.45% | 92.48% | 62.43% | 46.34% | 76.37% | 0.3694 |
| +6 | 85.72% | 91.00% | 58.45% | 45.84% | 75.12% | 0.4862 |
| +8 | 68.55% | 85.97% | 56.65% | 42.81% | 63.44% | 0.7019 |

| Seed | Paired +1 | +2 | +4 | +6 | +8 |
|---|---:|---:|---:|---:|---:|
| 137 | 100% | 99.84% | 97.48% | 96.39% | 93.85% |
| 163 | 100% | 99.86% | 99.57% | 99.08% | 95.55% |
| 191 | 99.47% | 99.39% | 94.16% | 95.76% | 94.34% |
| 223 | 99.41% | 93.77% | 78.71% | 76.21% | 70.57% |
| 257 | 99.43% | 95.49% | 92.46% | 87.58% | 75.55% |

| Predeclared criterion | Outcome |
|---|---|
| Every seed >=90%, separately at each positive shift | Passed +1/+2; failed +4/+6/+8 |
| Every seed >=90% across all positive shifts | Failed |
| Mean advantage >=10 points over each source control, at each positive shift | Passed at all five shifts |
| Compact loss <=1 point in every seed | Passed |

Each seed's mean accuracy improves over its original baseline at every tested positive shift. That does not mean every prediction improves. Paired rescue/harm counts are retained in the raw report. The low parent-level every-crop figures also prevent a near-perfect mean from being mistaken for robust behavior on every subset.

## What we learned

The paired-key rule has a repeatable benefit under this task's layout. The unresolved failures also persist: seed 223 misses 90% from +4 onward, and seed 257 fails at +6/+8. A successful local attention intervention is therefore not sufficient for broad shift robustness.

We should retain this bounded result rather than repeatedly add forcing sites until a threshold passes. These experiments have not established performance on uncropped long sequences, unseen tasks or language. They do not justify a new architecture. The [attention equations and source comparison](../attention/qkv-results.md#learn-the-operation) remain applicable.

## Verification and next decision

[Script](../../../experiments/confirm_pair_shifts.py), [exact parent inputs](../../../results/pair_shift_confirmation_parents.json) and [raw results](../../../results/pair_shift_confirmation.json) preserve all 24 conditions per seed. Explicit unmodified attention/forward checks against the unchanged model passed within 9.86e-14. Parent exclusions, labels, causal zeros, row sums, finite outputs and checkpoint integrity passed. Source hashes and paired-count arithmetic were checked separately.

Follow-up completed: [query-rank and value-pattern stratification](error-strata-results.md). Errors span all ranks; repeated targets are associated with better accuracy but may hide incorrect retrieval. A matched value-counterfactual test is proposed before causal interpretation.
