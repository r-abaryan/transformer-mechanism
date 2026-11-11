# Single-query write opposition: final decomposition

[Short guide](single-query-write-opposition-guide.md) | [Protocol](../../protocols/single-query-write-opposition.md) | [Source](../../../experiments/single_query_write_opposition.py) | [Report](../../../results/single_query_write_opposition/report.json) | [Every query](../../../results/single_query_write_opposition/queries.csv) | [Independent audit](../../../results/single_query_write_opposition/audit.json)

**Opposition persists for individual wrong queries. Mixing query losses offsets part of it in both seeds. This resolves the pooling ambiguity, not the historical cause of failed fitting.**

## Method

Seeds 3461/3463, exhaustive step 12000; all 768 train-32 histories and 32 queries each. Fixed saved weights, native logits/correctness and CE / (4 H m) weighting. Reconstruct effective write coefficients and individual event gradients analytically; match saved logits and all three original cohort gradients. Formal run has zero recurrent prefills, autograd/optimizer calls or training updates. No held-out evaluation or repair.

CPU float64, one thread. Two coefficient-based logit reconstructions, 49,152 query rows and 1,572,864 query-event sensitivities. These are correlated measurements from two reused seeds. Execution/save takes 3.43 seconds, excluding startup, initial hashes, preflight and audit; this is not a speed benchmark.

## Individual-query results

Weighted W uses the original objective scale and a 1e-14 sign guard. Geometry is the guide's coefficient-only bracket, computed directly without dividing by tiny or zero gradients, with a separate 1e-12 guard. All geometry signs resolve; weighted unresolved cases are not declared zero.

| Seed | Cohort | Queries | Weighted negative / positive / unresolved | Geometry negative / positive |
|---|---|---:|---|---|
| 3461 | Correct | 24,330 | 3,538 / 392 / 20,400 | 14,154 / 10,176 |
| 3461 | Error | 246 | 234 / 12 / 0 | 234 / 12 |
| 3461 | Total | 24,576 | 3,772 / 404 / 20,400 | 14,388 / 10,188 |
| 3463 | Correct | 24,223 | 3,745 / 436 / 20,042 | 14,244 / 9,979 |
| 3463 | Error | 353 | 336 / 17 / 0 | 336 / 17 |
| 3463 | Total | 24,576 | 4,081 / 453 / 20,042 | 14,580 / 9,996 |

Negative W occurs in 234/246 and 336/353 wrong queries, about 95% in each seed. Every query, including correct queries, has some opposite-sign coefficients within a shared value row. Some nevertheless have positive net W. Opposing pairs are therefore neither sufficient for net opposition nor sufficient for an incorrect prediction.

## Query mixing

Within each history/cohort, W_mixed = sum W_single + W_cross_query. Values are signed squared-gradient cross terms, not loss or accuracy changes.

| Seed | Cohort | Sum W_single | W_cross_query | W_mixed |
|---|---|---:|---:|---:|
| 3461 | Correct | -7.398446e-08 | +1.715164e-08 | -5.683282e-08 |
| 3461 | Error | -6.722587e-08 | +7.616967e-10 | -6.646417e-08 |
| 3461 | Total | -1.412103e-07 | +2.719570e-08 | -1.140146e-07 |
| 3463 | Correct | -7.888178e-08 | +2.217817e-08 | -5.670361e-08 |
| 3463 | Error | -9.050249e-08 | +4.349651e-10 | -9.006753e-08 |
| 3463 | Total | -1.693843e-07 | +3.687649e-08 | -1.325078e-07 |

Cross-query terms are positive in all six cohorts, partly offsetting individual-query opposition. W_mixed reproduces the earlier history diagnostic. This does not establish that multi-query training improves accuracy or that all query pairs align.

## Concrete opposing writes

One deterministic first negative wrong-query example per seed; illustrative, not representative selection. Each pair writes the same value embedding.

| Seed | History/query | True / predicted class | Write steps/value | Coefficients | Gradient dot product |
|---|---|---|---|---|---:|
| 3461 | 0 / 29 | 4 / 5 | 1, 4 / value 3 | -0.036028, +0.003300 | -8.692217e-13 |
| 3463 | 32 / 3 | 2 / 3 | 1, 21 / value 5 | -0.008092, +0.001314 | -7.275909e-14 |

## Verification and closure

Two preflight methods pass individual-loss autograd, ties/absent cohorts, different-value and zero controls, actual initialized saved reload/evaluator/auditor, and deliberate coefficient/label/mask/summary/CSV defects. Synthetic fixture forwards/autograd are separate from the formal analytic run; zero optimizer calls throughout.

Independent forward basis propagation checks coefficients; reverse matrix state gradients check individual event derivatives; chronological prefix products check query cross terms. All six cohorts/49,152 query rows pass. Maximum factor discrepancy: 1.110223e-16; 44 scoped protected files unchanged. The auditor's generic summary comparator uses 1e-9; a separate reduction of audited CSV rows/example factors additionally enforces the frozen 1e-12 summary guard. Original executed sources remain preserved.

[Strict summary/document checks](../../../results/single_query_write_opposition/document_verification.json) pass 34 continuous fields and all 14 displayed numeric rows; an intentional 1e-10 summary error is rejected.

Compared with [Yang et al., sections 3.1/3.2](https://arxiv.org/html/2406.06484v2#S3.SS2), this confirms a signed effective-read mechanism in our saved toy reader, not a reproduced paper score. The CE-gradient and query-mixing decompositions are local.

**Close the gradient-decomposition chain.** Feedback survives early transport; opposing contributions occur for individual queries; query mixing can offset them. Original 246/353 errors remain. No gradient blockage, architectural impossibility, unique learning cause or repair benefit is established. The remaining research task is a separately justified training-history or causal comparison, not further automatic partitions. RL remains parked.
