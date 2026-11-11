# Fitting gains coexist with gains and harms on unseen maps

Both predeclared coefficient-margin signatures **fail**. Full L-BFGS improves training accuracy, but its mean coefficient unit margin is lower than negative gradient in every seed. Held-out margin contrasts are mixed. The result does not support either a uniformly broader target-ranking advantage or one confined to training.

The fixed cohorts add useful evidence: initially wrong test examples improve on average, while initially correct examples lose margin and many become wrong. The [guide](fitting-trajectory-guide.md) explains how accuracy can rise while average margins fall.

## Method

[Frozen protocol](../../protocols/fitting-trajectory.md): seeds 3473/3479/3481, both saved direction trajectories, all five checkpoints and all existing train/validation/test queries at loads 2/8/16/32. Stage bins contain different actual gradient counts, reported in the CSV. This follows the Adam warm-up; it does not reconstruct from-scratch learning.

Measure native accuracy/CE/maps, target-value coefficient rankings, raw/unit margins, and paired changes on cohorts fixed by shared-start native correctness. Unit margins remove positive scale and common offsets. No donor swaps, fitting or new data. These are opened pools, three initialization seeds and one shared dataset; the six value shifts and repeated answers are correlated.

## Primary final load-32 contrasts

Full minus negative-gradient mean coefficient unit margin:

| Seed | Training | Validation | Test |
|---|---:|---:|---:|
| 3473 | -0.003613 | -0.005519 | -0.000133 |
| 3479 | -0.002421 | +0.009154 | -0.000942 |
| 3481 | -0.002200 | -0.000358 | +0.009527 |

The numerical sign tolerance is 1e-10. Neither frozen signature passes; these signs are not independent statistical tests. Training coefficient-ranking accuracy nevertheless improves in all three pairs, by 0.6104/0.4150/0.1221 percentage points. A mean margin is not a substitute for correctness.

## Final scores

All rows are load 32. Full = native L-BFGS; N = negative-gradient direction. Coeff = fixed largest-coefficient answer.

| Seed | Arm | Train accuracy | Train CE | Train coeff accuracy | Validation accuracy / CE | Test accuracy / CE | Test coeff accuracy |
|---|---|---:|---:|---:|---|---|---:|
| 3473 | Full | 100.0000% | 0.001639 | 99.4629% | 86.6536% / 4.0680 | 87.4023% / 4.2212 | 87.5000% |
| 3473 | N | 99.5036% | 0.030709 | 98.8525% | 87.3047% / 1.0518 | 88.6719% / 1.0055 | 88.6719% |
| 3479 | Full | 100.0000% | 0.007988 | 99.4629% | 87.0280% / 2.9364 | 88.6637% / 2.6227 | 88.7207% |
| 3479 | N | 99.2839% | 0.038454 | 99.0479% | 87.6628% / 1.0015 | 88.6719% / 0.8838 | 89.0137% |
| 3481 | Full | 99.9756% | 0.013711 | 98.9990% | 86.2305% / 2.3911 | 87.5895% / 2.2299 | 87.6953% |
| 3481 | N | 99.1699% | 0.040849 | 98.8770% | 87.6953% / 0.9693 | 87.4268% / 0.9787 | 87.4023% |

Train CE here is load-32 CE, not the earlier equal-load objective. Full test complete maps remain 0/12/6 of 384. Lower-load native training is perfect in all final arms; held-out load 8 still has errors, and load-16 test accuracy ranges from 97.0703% to 97.7539%. All stages, loads, query groups, validation coefficient scores and maps remain in the [raw report](../../../results/fitting_trajectory/report.json) and [1,080-row CSV](../../../results/fitting_trajectory/trajectory.csv), including empty groups.

## Changes from the shared start

Native test-32 answers rescued or harmed at the final checkpoint, among 12,288 answers per seed:

| Seed | Full rescues / harms | N rescues / harms | Full unit-margin change: initially wrong | Full unit-margin change: initially correct |
|---|---:|---:|---:|---:|
| 3473 | 313 / 471 | 40 / 42 | +0.085780 | -0.012237 |
| 3479 | 418 / 423 | 47 / 51 | +0.083596 | -0.013520 |
| 3481 | 501 / 485 | 75 / 79 | +0.144385 | -0.010500 |

These cohorts were declared before scoring. They show nonuniform changes, not why training selected them. Full training rescues 177/246/224 load-32 answers with zero new harms relative to the shared start; seed 3481 retains six original errors. Intermediate snapshots are not monotonic in every measure.

## What coefficient ranking localizes

| Seed | Full training: native correct, coeff wrong | Full test: native wrong, coeff correct | Full test: both wrong | Both-wrong share of native test errors |
|---|---:|---:|---:|---:|
| 3473 | 132 | 50 | 1,498 | 96.77% |
| 3479 | 132 | 53 | 1,340 | 96.20% |
| 3481 | 240 | 31 | 1,494 | 97.97% |

The learned decoder repairs training mixtures that simple coefficient ranking misclassifies. Most test mistakes survive that fixed counterfactual too. This weakens a decoder-only explanation under the chosen rule; it does not prove information loss, a capacity ceiling or irrecoverability by another decoder. Counts are correlated answers, not independent seed replications.

## Verification and decision

Actual scoring took 28.13 seconds on native Windows CPU float64, one thread, inference batch 64. Zero optimizer calls or gradients. Thirty saved checkpoints give 360 cells and 2,338,560 repeated evaluated answers; audit recomputations are separate.

Two actual saved-runtime preflight methods passed in 4.619 seconds, including decoder counterexamples, ties/empty groups, scale invariance and deliberate coefficient/logit/label/count/decision/missing-cell/missing-group/CSV faults. The [independent audit](../../../results/fitting_trajectory/audit.json) checks all cells through matrix recurrences, scalar native metrics, separately implemented margins/quantiles/cohorts/primary signs and exact historical native controls. Maximum discrepancies: coefficients 1.11e-15, logits 4.55e-13. All 65 scoped files remain unchanged. CSV serialization and its expected row extraction are shared.

Stop this diagnostic at its frozen boundary. A useful next candidate is a **training-exposure intervention** asking whether fitting a repeatedly reused finite map pool causes these gains and harms, or whether they persist with broader map exposure. It needs matched examples/compute, adequate fitting checks and fresh confirmation; it is not yet a frozen or queued run. The current evidence does not select a new optimizer, decoder or architecture. RL remains parked.

Primary sources inspected: [Yang et al. (2024), section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2), delta writes/reads; [Guo et al. (2017), section 4.2](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), positive scaling preserves decisions. Local factorization, margin predictions and measured scores are ours; the papers neither predict these results nor establish our learning cause.
