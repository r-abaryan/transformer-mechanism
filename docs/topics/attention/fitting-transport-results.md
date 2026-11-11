# Curvature transport has a seed-dependent fitting effect

The predeclared all-seed fitting-benefit rule **fails**. Removing direct routing/decoder transport preserves zero training errors in seeds 3473/3479, but increases seed 3481's errors from 6 to 24. Full transport helps the weakest seed under this search policy and cap; it is not needed to reach perfect finite-pool fitting in the other two.

Training CE falls in all three removal branches. Held-out load-32 CE rises in every pair, with mixed accuracy/map effects. Lower training loss is not reliable retrieval on unseen histories.

## Method and prediction

The unchanged reader starts from the same saved Adam-12000 weights. Full L-BFGS uses d=-Hg. Block removal uses only H_RR and H_DD while retaining global secant history, search policy and the 128-gradient cap. Both branches still share the model objective; we remove direct cross-block action, not all joint learning. The [guide](fitting-transport-guide.md) explains the equations; the [protocol](../../protocols/fitting-transport-training.md) was frozen after preflight/profiling, before formal training.

The criterion required strictly more all-load training errors after removal in **every seed**. It passes only for 3481. CE is descriptive and does not replace that error criterion. This is a causal direction-rule contrast on reused starts/pools, not fresh confirmation or an architecture comparison.

## Final training

| Seed | Direction | All-load errors | Equal-load CE | Train-32 accuracy | Train-32 maps / 768 |
|---|---|---:|---:|---:|---:|
| 3473 | Full | 0 | 0.0004362638 | 100.0000% | 768 |
| 3473 | Block | 0 | 8.0268071e-06 | 100.0000% | 768 |
| 3479 | Full | 0 | 0.0020771284 | 100.0000% | 768 |
| 3479 | Block | 0 | 2.7250506e-08 | 100.0000% | 768 |
| 3481 | Full | 6 | 0.0035381038 | 99.9756% | 762 |
| 3481 | Block | 24 | 0.0016338123 | 99.9023% | 744 |

All six runs pass the existing 99% nonempty training-group gate. Both procedures leave errors in seed 3481; all-seed perfection remains unmet. Final loads 2/8/16, query groups and intermediate snapshots remain in the complete CSV. Its map fields measure whole histories and repeat across group rows; absent-group CE/accuracy are blank. Accuracy/map preservation here does not imply equal margins or equal learned representations.

## Held-out load 32

Validation:

| Seed | Direction | Accuracy | CE | Complete maps |
|---|---|---:|---:|---:|
| 3473 | Full | 86.6536% | 4.0680 | 6 / 192 |
| 3473 | Block | 86.3770% | 23.3295 | 0 / 192 |
| 3479 | Full | 87.0280% | 2.9364 | 6 / 192 |
| 3479 | Block | 87.5814% | 54.8221 | 8 / 192 |
| 3481 | Full | 86.2305% | 2.3911 | 0 / 192 |
| 3481 | Block | 87.2559% | 7.6249 | 12 / 192 |

Test:

| Seed | Direction | Accuracy | CE | Complete maps |
|---|---|---:|---:|---:|
| 3473 | Full | 87.4023% | 4.2212 | 0 / 384 |
| 3473 | Block | 88.2650% | 22.5923 | 11 / 384 |
| 3479 | Full | 88.6637% | 2.6227 | 12 / 384 |
| 3479 | Block | 88.3952% | 49.0772 | 19 / 384 |
| 3481 | Full | 87.5895% | 2.2299 | 6 / 384 |
| 3481 | Block | 86.8245% | 7.5634 | 0 / 384 |

Test complete-map reliability remains weak. The removal branch loses test accuracy in two seeds and gains it in one; complete maps increase in two and decrease in the weakest. CE is substantially worse in every validation/test pair. Held-out load-16 CE also increases in every seed/split pair, with mixed accuracy effects. We did not perform a scale decomposition or measure calibration for these new endpoints, so its loss increase cannot be assigned to one cause from the earlier scale study.

## Budget and actual execution

| Seed | Full / block gradients | Full / block optimizer calls |
|---|---:|---:|
| 3473 | 127 / 128 | 63 / 60 |
| 3479 | 128 / 127 | 63 / 62 |
| 3481 | 127 / 128 | 63 / 58 |

Formal execution: **765 gradients, 369 optimizer calls, 34,076,160 supervised answers**. Each gradient covers 3,072 histories, 44,544 query answers and 98,304 input events. Total history/event exposures are 2,350,080/75,202,560. Different trial/update counts follow from the direction intervention; equal caps do not imply equal updates or wall time.

Six serial native Windows runs took 454.96 seconds excluding imports, preflight, profiling and final audit. CPU float64, one thread. GPU peak allocated/reserved memory is inapplicable. Complete-step profiling included optimizer state; its tensor-payload counts are not process peak memory. No speed comparison is claimed.

## Verification

Two actual saved-runtime preflight methods passed in 9.651 seconds, including supplied logits/empty groups and deliberate gradient/direction/counter/loss/label/logit/missing-group/decision faults. The audit separately recomputes matrix gradients and source labels, replays every trial/state, checks removal directions through dense inverse-BFGS operators and checks every final/intermediate score cell through an independent forward/metric path. Final error counts are independently recounted from logits; some bookkeeping/CE reducers and the line-search replay are shared.

All native control traces and endpoints reproduce historical controls exactly. First steps agree; later directions differ in every pair. 369/369 moved steps satisfy measured strong-Wolfe conditions. All **168 score cells / 1,536,768 repeated evaluated answers** pass audit, with 49 protected files unchanged. Maximum discrepancies: gradients 1.35e-14, directions 6.74e-13, logits 5.46e-12. [Audit](../../../results/fitting_transport_training/audit.json), [raw report](../../../results/fitting_transport_training/report.json), [504-row CSV](../../../results/fitting_transport_training/evaluation.csv).

Three initialization seeds share one opened dataset; the six value shifts and repeated predictions are correlated. This 128-gradient phase follows an Adam warm-up, so it does not establish what from-scratch learning would do. No longer-input or independent-dataset test was added.

## What closes, what remains

Close cross-block direction variants at the declared limit. Presence and local cancellation did not imply an all-seed fitting benefit. We now have a measured conditional benefit in the weakest seed and two perfect witnesses without direct cross-block action, alongside poor held-out reliability.

The mechanisms of within-block preconditioning, representation learning and unseen-map failure remain unresolved. The next step is to consolidate the procedure and representation evidence into one causal account with explicit unknowns, before choosing another experiment. No new optimizer candidates, partitions or fitting extensions are queued. RL remains parked.

Sources inspected: [PyTorch 2.11 documentation](https://docs.pytorch.org/docs/2.11/generated/torch.optim.LBFGS.html) and [implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/lbfgs.py), secant updates, directions and strong-Wolfe search; [Liu/Nocedal (1989), author abstract](https://users.iems.northwestern.edu/~nocedal/Abstracts/limited-memory.html), limited-memory/convex-theory scope. The block intervention, fitting criterion and measured scores are local. These sources neither predict the outcome nor prove convergence for this reader.
