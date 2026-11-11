# Fitting depends on compatible routing and decoding

Neither single-component swap reproduces native L-BFGS training errors in any seed. Both predeclared sufficiency rules fail. The decoder change improves training accuracy with L-BFGS routing and worsens it with negative-gradient routing. This identifies an endpoint interaction, not its historical learning cause.

## Method

We evaluate four routing/decoder combinations per seed using existing weights. **L** means the L-BFGS donor; **N** means the negative-gradient donor. Routing supplies keys, tied queries and beta. Decoder supplies the value embeddings, classifier and bias together. The [educational guide](fitting-representation-guide.md) derives $z=(WV^T)c+b$ and explains why this split avoids breaking the values' private coordinates. The [protocol](../../protocols/fitting-representation.md) was fixed before hybrid evaluation.

All three seeds, all loads 2/8/16/32 and training/validation/test pools are included: 12 hybrids, 144 cells. Pools and endpoints were already opened; this is diagnosis, not fresh confirmation. Six value shifts share each base history. Values are constant per key within a history, so changing-value overwrite behavior is outside scope.

## Training findings

Errors count all 44,544 training answers. The objective weights the four loads equally. Accuracy/maps shown are load 32. A complete map requires every query in a history to be correct.

| Seed | Routing / decoder | All-training errors | All-load CE | Train-32 accuracy | Complete maps |
|---|---|---:|---:|---:|---:|
| 3473 | L / L | 0 | 0.0004363 | 100.0000% | 768/768 |
| 3473 | L / N | 149 | 0.0114795 | 99.4751% | 639/768 |
| 3473 | N / L | 223 | 0.0075818 | 99.0926% | 575/768 |
| 3473 | N / N | 122 | 0.0080508 | 99.5036% | 652/768 |
| 3479 | L / L | 0 | 0.0020771 | 100.0000% | 768/768 |
| 3479 | L / N | 14 | 0.0113582 | 99.9430% | 754/768 |
| 3479 | N / L | 201 | 0.0094608 | 99.1821% | 585/768 |
| 3479 | N / N | 176 | 0.0100521 | 99.2839% | 605/768 |
| 3481 | L / L | 6 | 0.0035381 | 99.9756% | 762/768 |
| 3481 | L / N | 218 | 0.0123384 | 99.1130% | 576/768 |
| 3481 | N / L | 256 | 0.0105041 | 98.9583% | 564/768 |
| 3481 | N / N | 204 | 0.0104868 | 99.1699% | 588/768 |

Routing alone helps seed 3479 strongly, reducing 176 errors to 14, but harms the other two seeds. Seed 3473's L/N hybrid also introduces **20 load-16 errors**, in addition to 129 load-32 errors. All other lower-load training groups remain perfect. Decoder alone increases errors in every seed, although CE improves in the first two. Lower loss and fewer errors are different criteria.

Both hypotheses required no more errors than native L-BFGS in **every** seed. Routing-only errors are 149/14/218; decoder-only errors are 223/201/256; native L-BFGS errors are 0/0/6. Neither component is independently sufficient under this donor contract. This does not prove either component is universally necessary, or that another compatible decoder could not work.

## Interaction, explained

Changing N decoder to L decoder has opposite accuracy effects at the two fixed routing donors. Values below are percentage points at train load 32. Interaction is the difference between those conditional effects, not an additive causal percentage.

| Seed | Decoder effect at N routing | Decoder effect at L routing | Interaction |
|---|---:|---:|---:|
| 3473 | -0.4110 | +0.5249 | +0.9359 |
| 3479 | -0.1017 | +0.0570 | +0.1587 |
| 3481 | -0.2116 | +0.8626 | +1.0742 |

The successful pair cannot be understood by ranking component norms or assigning the gain to one donor alone. Hybrid failure can reflect co-adaptation; it does not identify what gradient history caused it.

## Saved parameter trajectories

The centered effective decoder is $B=[A\;b]$ with each column's class mean removed. Relative to the common warm start, we report the least-squares scalar $s=\langle B,B_0angle/\|B_0\|^2$ and residual $\|B-sB_0\|/\|B\|$. These are descriptive, not fitted prediction temperatures or interventions.

| Seed | Arm | Final beta | Effective decoder norm | Scalar versus warm start | Relative residual |
|---|---|---:|---:|---:|---:|
| 3473 | L | 0.1631 | 1765.76 | 4.463 | 4.95% |
| 3473 | N | 0.1767 | 408.41 | 1.028 | 0.57% |
| 3479 | L | 0.1609 | 1269.71 | 3.140 | 2.00% |
| 3479 | N | 0.1667 | 415.49 | 1.023 | 0.55% |
| 3481 | L | 0.1692 | 966.09 | 2.553 | 3.26% |
| 3481 | N | 0.1764 | 387.98 | 1.025 | 0.52% |

L-BFGS substantially enlarges the effective map, but its non-scalar change is nonzero. Mean off-diagonal key overlap changes little: warm values are 0.1829 to 0.1841; final L values are 0.1833 to 0.1843. This average does not rule out important pair-specific geometry changes. All 30 saved milestones remain in the [raw report](../../../results/fitting_representation/report.json).

## Held-out behavior

At either fixed routing donor, the L decoder increases test-32 CE in every seed. Accuracy effects are mixed; complete maps remain rare. This local decoder intervention carries a loss penalty on these opened test histories. It does not establish that scalar growth alone caused it, or measure calibration.

| Seed | Routing | Test CE with L decoder | Test CE with N decoder | Test accuracy L / N | Complete maps L / N, out of 384 |
|---|---|---:|---:|---:|---:|
| 3473 | L | 4.2212 | 0.9978 | 87.4023% / 87.5326% | 0 / 0 |
| 3473 | N | 4.2614 | 1.0055 | 88.4277% / 88.6719% | 18 / 18 |
| 3479 | L | 2.6227 | 0.8849 | 88.6637% / 88.6230% | 12 / 12 |
| 3479 | N | 2.6214 | 0.8838 | 88.7207% / 88.6719% | 12 / 12 |
| 3481 | L | 2.2299 | 0.9163 | 87.5895% / 87.7930% | 6 / 6 |
| 3481 | N | 2.3897 | 0.9787 | 87.3535% / 87.4268% | 0 / 6 |

The same conditional CE direction holds at validation load 32. All other loads, splits and empty/nonempty unique/repeated-key groups are retained in the [432-row CSV](../../../results/fitting_representation/evaluation.csv).

## Evidence and boundary

Actual execution: 12 saved hybrids and 935,424 evaluated answers, **zero optimizer, gradient or solver calls**. Producer time was 4.05 seconds; CPU float64, one thread. The task uses tiny checkpoints and vectorized evaluation, with no training. Audit and preflight time are separate.

Two runtime preflight methods passed in 3.504 seconds. Controls include initialized saved evaluator/audit, native versus value-basis arithmetic, private decoder-basis permutations, ties/absent groups and deliberate donor/logit/label/count/decision/missing-cell faults. The [audit](../../../results/fitting_representation/audit.json) independently checks matrix logits, labels, scalar metrics and sufficiency error counts for all 144 cells. Native self-donor logits exactly reproduce the saved endpoints. Maximum matrix-logit discrepancy is 4.55e-13; 63 protected files remain unchanged. Trajectory statistics and contrast reductions share reducer code, while the key-overlap calculation receives an additional independent check. Prior evidence is preserved; this is not a repository-wide audit.

Stop donor variants and training extensions. Positive scaling already cannot explain a change in accuracy at fixed routing: it preserves class ranking. That does not need another experiment. The next testable question is **how much of the held-out loss penalty comes from decoder scale versus its non-scalar change**. A separately declared scale/shape diagnostic can use these endpoints, with its projection fixed before evaluation and no temperature search. Why the optimizer learns the joint solution and why unseen maps remain weak are still unresolved. RL remains parked.

Source inspected: [Yang et al. (2024), section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2) for delta writes and reads. The exact value-basis factorization, donor predictions and interaction interpretation are local reasoning. The source does not predict these scores or establish our toy reader as full DeltaNet.
