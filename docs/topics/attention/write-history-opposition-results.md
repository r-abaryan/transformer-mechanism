# Within-history versus between-history opposition

[Short guide](write-history-opposition-guide.md) | [Protocol](../../protocols/write-history-opposition.md) | [Source](../../../experiments/write_history_opposition.py) | [Report](../../../results/write_history_opposition/report.json) | [Every history](../../../results/write_history_opposition/histories.csv) | [Independent audit](../../../results/write_history_opposition/audit.json)

**Opposition occurs inside individual histories. Across-history pooling does not explain it away. It also occurs for correct-answer feedback, so it is not sufficient evidence of the failure's cause.**

## Method

Reuse audited event gradients from seeds 3461/3463, step 12000, all 768 train-32 histories each. Correct/error/total cohorts and the original masked CE / (4 H m) weighting remain unchanged. Map each event's eight-coordinate gradient into its shared value-embedding row before aggregation. Formal reduction uses saved tensors only: zero model evaluations, backward passes, optimizer calls or training updates.

CPU float64, one thread. Six cohort summaries and 4608 per-history rows. Reduction/CSV writing takes 0.15 seconds, excluding startup, initial input/hash checks, preflight and audit. This is provenance, not a performance benchmark. Histories include cyclic value renamings; their counts are not independent seeds.

## Signed decomposition

All terms below are divided by event-gradient energy E0. Negative means opposition; positive means alignment. W is within-history/different-position; B is between-history/different-position; P is between-history/same-position. Their sum plus one equals J/E0. These are algebraic ratios, not percentages of lost learning or accuracy.

| Seed | Cohort | W/E0 | B/E0 | P/E0 |
|---|---|---:|---:|---:|
| 3461 | Correct | -0.485249 | -0.046095 | +0.292977 |
| 3461 | Error | -0.581855 | +0.120462 | -0.134741 |
| 3461 | Total | -0.525957 | -0.367141 | +0.312705 |
| 3463 | Correct | -0.444111 | -0.564669 | +0.655585 |
| 3463 | Error | -0.554029 | -0.147901 | +0.095923 |
| 3463 | Total | -0.489926 | -0.497529 | +0.404361 |

W is negative in every cohort. For error feedback, between-history temporal alignment partly offsets W in 3461; in 3463 it adds opposition. Same-position terms have different signs too. Joint shared-value gradients remain nonzero, as in the primary diagnostic.

## Individual histories

Count only histories with at least one query supervised by that cohort. A history can contribute to both correct and error cohorts. Sign guard: absolute 1e-14; unresolved does not mean mathematically zero.

| Seed | Cohort | Supervised histories | Negative W | Positive W | Unresolved |
|---|---|---:|---:|---:|---:|
| 3461 | Correct | 768 | 706 | 52 | 10 |
| 3461 | Error | 207 | 203 | 4 | 0 |
| 3461 | Total | 768 | 713 | 45 | 10 |
| 3463 | Correct | 768 | 699 | 69 | 0 |
| 3463 | Error | 258 | 237 | 21 | 0 |
| 3463 | Total | 768 | 706 | 62 | 0 |

The error cohort excludes 561/510 histories with no wrong query, matching the earlier complete-map counts. Original 246/353 training errors remain unchanged; no repair ran. Within-history totals already combine query losses and shared-value writes. This does not establish that one isolated wrong query has opposing temporal gradients.

## Verification and stopping boundary

Two preflight methods pass known within-only, between-only, same-position, aligned, orthogonal, different-value and zero controls. The actual reduction/audit entry runs on an initialized saved primary fixture; deliberate summary, missing-cohort, CSV and input defects are rejected. Its synthetic primary forwards/backwards are separate from the formal saved-tensor reduction; neither uses an optimizer.

Independent chronological-prefix accumulation checks W, B and P directly, reproduces the earlier temporal C and joint J, and passes all six summaries/4608 rows. Maximum summary discrepancy: 1.554312e-15. All 42 scoped protected files remain unchanged. One formal reduction and audit; no recovery or threshold changes.

Compared with [Yang et al., section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2), the saved reader follows the delta recurrence in our transposed-state convention. This gradient decomposition is local evidence, not a reproduced paper result or historical causal explanation.

Stop this partition. The remaining useful distinction is **query mixing versus opposition for one individual query**. A separately bounded query-wise check could resolve it; more parameter variants or training are not justified by negative W alone. RL stays parked.
