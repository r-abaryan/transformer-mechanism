# Broader query exposure: results

**Large gains without an architecture change, but the fixed reliability rule fails.** All-key training improves unique-query accuracy in every seed at four and six pairs. Seed1327 reaches only87.77% on six-pair unique queries, below90%. Preserve the gains and failed criterion together. No additional tuning is selected.

## What changed, and why it matters

The [learning guide](query-exposure-guide.md) explains the input examples, loss and training tradeoff. Previously the model was always asked about the only repeated key. Here the broader arm is also asked about keys occurring once. We test coverage of questions, not more model capacity.

Both arms use identical RoPE/ReLU/FFN32 architecture and learned initialization,5,088 parameters, Adam.001, batch64 and6,000 updates. Alternating three/four-pair batches share identical contexts and sequence lengths. Only the query token and answer label can differ. Final checkpoints were evaluated after all ten training attempts completed. No failed or replacement runs.

## Data and scope

Four-pair families are fresh:800 train/100 validation/200 test. Six-pair families are fresh:200 test only. The exclusion inventory contains5,220 previously used families. Smaller fragments and other relabelings can remain familiar.

Three-pair splits are reused unchanged from earlier studies:160/20/40 families. Their results are exposed diagnostics. Each family supplies36 context variants; evaluation expands each context to all n-1 distinct queries. Thus four-pair testing has7,200 repeated and14,400 unique-query answers per model; six-pair testing has7,200 repeated and28,800 unique-query answers. These are dependent observations within shared families.

The [frozen protocol](../../query-exposure-protocol.md) sets transfer and preservation criteria separately. Historical studies use different seeds and datasets; their scores are not a matched baseline here.

## Mean results across five seeds

R means a query for the repeated key; U means a query for a key occurring once. Both training conditions are evaluated on both types.

| Pairs | Repeated-only training: R | All-key training: R | Repeated-only training: U | All-key training: U |
|---|---:|---:|---:|---:|
| 3 (reused) | 99.99% | 99.96% | 80.96% | 99.94% |
| 4 | 99.52% | 99.80% | 74.32% | 99.89% |
| 6 | 85.73% | 95.93% | 59.53% | 94.83% |

Six-pair repeated-query accuracy also improves in every seed, despite fewer repeated-query training examples. This is an observed benefit of the full coverage recipe, not proof of a particular internal learning mechanism.

## Each paired seed

Cells show repeated-only -> all-key training accuracy.

| Seed | Four-pair unique | Six-pair unique | Six-pair repeated |
|---|---:|---:|---:|
| 1301 | 48.44% -> 100.00% | 39.11% -> 98.22% | 86.11% -> 99.68% |
| 1321 | 77.20% -> 99.99% | 62.17% -> 96.20% | 85.47% -> 99.64% |
| 1327 | 70.36% -> 99.49% | 59.03% -> 87.77% | 72.47% -> 81.15% |
| 1361 | 88.19% -> 99.99% | 67.23% -> 95.59% | 90.81% -> 99.62% |
| 1373 | 87.42% -> 100.00% | 70.09% -> 96.36% | 93.76% -> 99.54% |

Seed1327 also reaches only81.15% six-pair repeated accuracy. The other broader-exposure seeds exceed99.5% there. Strong means do not erase this weak run.

## Exposure and cost

Each condition sees192,000 three-pair and192,000 four-pair examples (3,456,000 tokens total). Repeated-only always queries the duplicated key. Broader exposure divides the same budget among present keys.

| Seed | Broader3-pair: repeated / unique counts | Broader4-pair: repeated / unique counts |
|---|---:|---:|
| 1301 | 96,008 / 95,992 | 64,138 / 127,862 |
| 1321 | 96,026 / 95,974 | 64,050 / 127,950 |
| 1327 | 96,353 / 95,647 | 63,995 / 128,005 |
| 1361 | 96,062 / 95,938 | 63,842 / 128,158 |
| 1373 | 95,997 / 96,003 | 64,027 / 127,973 |

Training forward/backward architecture and batch shapes match. CPU float64, one thread, PyTorch 2.14.0+cpu. Recorded loop totals are 282.58s versus 291.25s. They include periodic validation, exclude checkpoint writes/final tests, and contain different data-rewrite overhead. They are not controlled speed measurements.

## Detailed test metrics

Cross-entropy (CE) penalizes low probability on the correct answer; lower is better. Context correctness requires all n-1 queries for that context to be correct. Overall accuracy weights query types by their frequency, so it should not replace the separate scores above.

| Seed / training / pairs | R accuracy | U accuracy | R CE | U CE | Overall accuracy | All-query context correctness |
|---|---:|---:|---:|---:|---:|---:|
| 1301 / repeated / 3 | 100.00% | 42.78% | 0.00090 | 3.94525 | 71.39% | 42.78% |
| 1301 / repeated / 4 | 99.06% | 48.44% | 0.03619 | 3.63298 | 65.31% | 18.24% |
| 1301 / repeated / 6 | 86.11% | 39.11% | 0.68313 | 4.57574 | 48.51% | 0.40% |
| 1301 / all / 3 | 100.00% | 100.00% | 0.00040 | 0.00013 | 100.00% | 100.00% |
| 1301 / all / 4 | 100.00% | 100.00% | 0.00064 | 0.00013 | 100.00% | 100.00% |
| 1301 / all / 6 | 99.68% | 98.22% | 0.01068 | 0.11585 | 98.51% | 92.58% |
| 1321 / repeated / 3 | 100.00% | 81.67% | 0.00013 | 1.46756 | 90.83% | 81.67% |
| 1321 / repeated / 4 | 99.89% | 77.20% | 0.00510 | 1.87216 | 84.76% | 56.62% |
| 1321 / repeated / 6 | 85.47% | 62.17% | 1.01905 | 3.17560 | 66.83% | 9.26% |
| 1321 / all / 3 | 100.00% | 100.00% | 0.00027 | 0.00009 | 100.00% | 100.00% |
| 1321 / all / 4 | 100.00% | 99.99% | 0.00039 | 0.00051 | 99.99% | 99.97% |
| 1321 / all / 6 | 99.64% | 96.20% | 0.01455 | 0.21821 | 96.89% | 84.57% |
| 1327 / repeated / 3 | 99.93% | 87.78% | 0.00261 | 0.73447 | 93.85% | 87.71% |
| 1327 / repeated / 4 | 99.22% | 70.36% | 0.02384 | 2.14717 | 79.98% | 45.35% |
| 1327 / repeated / 6 | 72.47% | 59.03% | 2.02886 | 2.94470 | 61.72% | 3.51% |
| 1327 / all / 3 | 99.79% | 99.72% | 0.01086 | 0.01003 | 99.76% | 99.51% |
| 1327 / all / 4 | 99.01% | 99.49% | 0.03008 | 0.01969 | 99.33% | 98.01% |
| 1327 / all / 6 | 81.15% | 87.77% | 1.10081 | 0.76053 | 86.45% | 47.76% |
| 1361 / repeated / 3 | 100.00% | 97.92% | 0.00016 | 0.09689 | 98.96% | 97.92% |
| 1361 / repeated / 4 | 99.96% | 88.19% | 0.00074 | 0.87764 | 92.11% | 76.50% |
| 1361 / repeated / 6 | 90.81% | 67.23% | 0.68774 | 2.83299 | 71.94% | 10.79% |
| 1361 / all / 3 | 100.00% | 100.00% | 0.00037 | 0.00007 | 100.00% | 100.00% |
| 1361 / all / 4 | 100.00% | 99.99% | 0.00035 | 0.00017 | 100.00% | 99.99% |
| 1361 / all / 6 | 99.62% | 95.59% | 0.01444 | 0.29561 | 96.39% | 82.35% |
| 1373 / repeated / 3 | 100.00% | 94.65% | 0.00094 | 0.30667 | 97.33% | 94.65% |
| 1373 / repeated / 4 | 99.47% | 87.42% | 0.02022 | 0.79507 | 91.44% | 74.97% |
| 1373 / repeated / 6 | 93.76% | 70.09% | 0.32634 | 2.32223 | 74.82% | 14.61% |
| 1373 / all / 3 | 100.00% | 100.00% | 0.00043 | 0.00009 | 100.00% | 100.00% |
| 1373 / all / 4 | 100.00% | 100.00% | 0.00095 | 0.00015 | 100.00% | 100.00% |
| 1373 / all / 6 | 99.54% | 96.36% | 0.01352 | 0.16444 | 96.99% | 85.21% |

## Fixed decision

| Requirement | Outcome |
|---|---|
| All broader seeds >=90% unique accuracy at4/6 pairs | Four pairs passes; six pairs fails at seed1327 (87.77%). |
| Mean unique gain >=5pp at4/6; no seed worsens | Pass at both lengths. |
| Every broader seed >=99% repeated accuracy at3/4 | Pass; three-pair evidence is reused diagnostic data. |
| Mean repeated loss <=.5pp at3/4 and <=2pp at6 | Pass. |

The overall adoption rule fails on one requirement. This rejects the claim of reliable transfer under the fixed thresholds, not the measured coverage benefit. The same architecture learned substantially better behavior with broader questions; the earlier weakness was therefore not evidence of an absolute representational inability. We have not proven the complete cause of all earlier errors.

Five initialization seeds share one split. There is only one duplicated key and all values differ. Multiple duplicated keys, equal values, other tasks and longer lengths remain untested. No significance claim, internal-circuit claim or universal training rule follows.

## Verification and evidence

All 10 runs finished. Independent attention evaluation reproduced all 30 checkpoint/length evaluations. Maximum absolute logit difference was `1.73e-13`. The audit also checked labels, split exclusions, paired initial weights, context streams, query counts, update counts and artifact hashes.

Evidence: [frozen protocol](../../query-exposure-protocol.md), [educational guide](query-exposure-guide.md), [raw results](../../../results/query_exposure.json), [data](../../../results/query_exposure_data.json), [independent audit](../../../results/query_exposure_audit.json). Implementation: [training](../../../experiments/query_exposure.py), [auditor](../../../experiments/audit_query_exposure.py).

Recheck saved checkpoints without training:

```powershell
.venv/Scripts/python.exe experiments/audit_query_exposure.py
```

This is a local controlled study, not a paper benchmark reproduction. Comparison closed. Retain the coverage benefit and failed reliability criterion; do not tune on this opened test.
