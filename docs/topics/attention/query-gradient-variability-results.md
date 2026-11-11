# Query sampling and beta-gradient variation

Query choice dominates the measured alpha-gradient variance at load 32. Late in both resampled runs, one-query batch-64 variation exceeds the expected derivative. This identifies a candidate learning difficulty, not a proved cause of the retention errors. No training ran.

## Method

All 28 saved replay checkpoints: seeds 3461/3463, fixed/resampled arms, steps 0/10/100/300/1000/2000/3000. Original training histories only, 768 per load 2/8/16/32. Enumerate every present key and its derivative with respect to alpha, where beta = sigmoid(alpha). CPU float64, one thread. Zero optimizer updates, no new test or data selection.

112 cells, 1247232 per-target derivatives. Assigned-query values are subsets of the same arrays. Exact finite-pool population moments replace Monte Carlo estimates. Compute the implied standard errors for one-query and exhaustive-query feedback on 64 independently sampled histories with replacement at fixed weights and one load. They share the expected all-query objective. The exhaustive variant uses m times as many answer targets; it is not a compute/label-matched comparison. [Protocol](../../protocols/query-gradient-variability.md) | [Short educational guide](query-gradient-variability-guide.md).

## Main finding

Across all 28 load-32 snapshots, within-history query choice contributes **96.47% to 98.66%** of raw alpha-gradient variance. This is a variance decomposition, not a percentage of errors caused by sampling. The remaining term comes from differences between histories.

At step 3000 in resampled models, one-query signal/noise is **0.480/0.392**. Reading every key at the same history count gives **3.234/2.831**, with standard error **85.17%/86.16% lower**. Those values concern a hypothetical fixed-weight estimator; no exhaustive-feedback training or quality improvement was tested.

Signal/noise = absolute mean derivative divided by the batch-gradient standard deviation. Values below one do not give a sign-error probability. The expected load-32 derivative is still positive, favoring lower beta under plain gradient descent; the noisy estimator remains unbiased.

## How it changes during resampled training

| Seed | Step | Mean alpha derivative | Query share of variance | One-query signal/noise | All-query signal/noise |
|---|---|---|---|---|---|
| 3461 | 0 | 0.028131 | 96.47% | 3.216 | 17.112 |
| 3461 | 10 | 0.018960 | 96.51% | 2.614 | 13.984 |
| 3461 | 100 | 0.018162 | 97.34% | 1.382 | 8.472 |
| 3461 | 300 | 0.120043 | 97.18% | 3.534 | 21.048 |
| 3461 | 1000 | 0.140593 | 97.89% | 2.073 | 14.271 |
| 3461 | 2000 | 0.062472 | 97.94% | 0.870 | 6.064 |
| 3461 | 3000 | 0.033941 | 97.80% | 0.480 | 3.234 |
| 3463 | 0 | 0.003490 | 97.17% | 0.707 | 4.204 |
| 3463 | 10 | 0.002244 | 97.54% | 0.415 | 2.650 |
| 3463 | 100 | 0.028476 | 97.68% | 2.000 | 13.129 |
| 3463 | 300 | 0.131193 | 97.91% | 3.691 | 25.526 |
| 3463 | 1000 | 0.153314 | 98.66% | 2.146 | 18.505 |
| 3463 | 2000 | 0.062321 | 98.37% | 0.875 | 6.853 |
| 3463 | 3000 | 0.026254 | 98.08% | 0.392 | 2.831 |

Both resampled seeds have one-query signal/noise below one at steps 2000 and 3000. Their step-1000 values exceed two. Seed 3463 is also below one at initialization and step 10. Noise relative to signal therefore does not worsen monotonically, and a small ratio alone does not identify a failed learning phase.

## All terminal loads and both checkpoint arms

All-query mean derivatives are evaluated even in fixed models, which were trained on the different assigned-query objective. Their signal/noise does not describe that assigned objective. Every load-2 mean is unresolved at the predeclared absolute 1e-7 cutoff, even when its relative signal/noise is large.

| Seed | Checkpoint arm | Load | Mean alpha derivative | Query share | One-query signal/noise | All-query signal/noise | Direction |
|---|---|---|---|---|---|---|---|
| 3461 | Fixed | 2 | -1.313e-10 | 30.99% | 1.048 | 1.261 | Unresolved |
| 3461 | Fixed | 8 | 1.107e-04 | 87.49% | 0.094 | 0.267 | Lower |
| 3461 | Fixed | 16 | 3.876e-02 | 92.95% | 2.313 | 8.709 | Lower |
| 3461 | Fixed | 32 | 2.104e-01 | 96.96% | 5.073 | 29.086 | Lower |
| 3461 | Resampled | 2 | -8.666e-09 | 10.98% | 6.481 | 6.869 | Unresolved |
| 3461 | Resampled | 8 | -1.184e-03 | 86.60% | 0.811 | 2.217 | Higher |
| 3461 | Resampled | 16 | -1.412e-02 | 94.50% | 0.701 | 2.989 | Higher |
| 3461 | Resampled | 32 | 3.394e-02 | 97.80% | 0.480 | 3.234 | Lower |
| 3463 | Fixed | 2 | -3.966e-10 | 24.88% | 1.114 | 1.286 | Unresolved |
| 3463 | Fixed | 8 | 3.295e-04 | 87.75% | 0.176 | 0.503 | Lower |
| 3463 | Fixed | 16 | 2.890e-02 | 93.13% | 2.274 | 8.677 | Lower |
| 3463 | Fixed | 32 | 1.586e-01 | 97.40% | 5.104 | 31.653 | Lower |
| 3463 | Resampled | 2 | -9.540e-09 | 40.49% | 1.020 | 1.322 | Unresolved |
| 3463 | Resampled | 8 | -8.230e-04 | 86.88% | 0.459 | 1.267 | Higher |
| 3463 | Resampled | 16 | -1.802e-02 | 92.87% | 1.645 | 6.162 | Higher |
| 3463 | Resampled | 32 | 2.625e-02 | 98.08% | 0.392 | 2.831 | Lower |

Resampled seed 3463 load 16 retains one-query signal/noise above one, unlike seed 3461. Fixed load-32 all-query signal/noise remains about five despite its worse all-query accuracy. These exceptions rule out a universal association between low scalar signal/noise and prediction failure.

## Verification and limits

Preflight: three methods pass known-variance fixtures, initialized checkpoint evaluation/audit at four loads, supplied logits, missing groups and deliberately wrong target derivatives, labels, metrics and variance fields. Formal audit passes all 112 cells and two finite-difference spacings for every target. Native autograd checks both objective means; independent scalar sums check all variance/statistic fields. Earlier snapshot logits and mean derivatives also match.

Maximum errors: logits 1.421e-14, per-target finite differences 1.358e-08, native all-query mean gradient 7.221e-17. The finite-difference tolerance is 1e-8 + 1e-5 times derivative magnitude. All 82 locked dependencies/artifacts remain unchanged. Producer evaluation took 14.71 seconds in one execution, excluding preflight/audit; this is not a speed benchmark.

Only the raw alpha coordinate is measured. This is not whole-vector covariance or noise in Adam-preconditioned updates. The original optimizer cycles loads and changes weights; fixed-weight iid moments do not reconstruct its stochastic path. Numerical checks for extremely small derivatives have limited relative resolution. The mean-direction cutoff and raw magnitudes stay visible. No convergence, optimal batch size, general architecture limit or reliable retention repair is established.

Original fitting failures remain unchanged: resampled load-32 all-query training accuracy is 89.75%/90.03%. The analysis neither invalidates earlier verified runs nor supplies a fresh confirmation.

- [Every snapshot/load/group statistic](../../../results/query_gradient_variability/metrics.csv). The derivative statistics belong to the complete load cell, not each subgroup separately. Missing groups have count zero and null loss/accuracy.
- [Formal report](../../../results/query_gradient_variability/report.json) and [independent audit](../../../results/query_gradient_variability/audit.json).

## What this justifies next

A controlled training comparison can now test whether reducing within-history query sampling changes fitting. Match histories, optimizer, updates and answer counts. For example, compare every present key against the same number of uniformly sampled queries with replacement. Both use more answer feedback than the original one-query recipe, so gains shared by both arms would not isolate variance reduction. Freeze the budget and groups before training; keep failed fitting and weak seeds visible. No such training comparison ran here. RL remains parked.

Sources inspected: [McCandlish et al. 2018, section 2.2](https://arxiv.org/html/1812.06162v1#S2.SS2), unbiased iid batch gradients and covariance/B; [Yang et al. 2024, section 2.2](https://arxiv.org/html/2406.06484v2), delta correction and input-dependent beta. Our one-coordinate diagnostic is narrower than their full gradient-noise scale and DeltaNet architecture. Within/between variance is a local application of total variance. Neither source predicts these scores or proves the sampling mechanism caused failure.
