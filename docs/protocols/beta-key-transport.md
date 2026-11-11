# Beta versus learned key geometry

Frozen before evaluation, 2026-10-05. Question: does the observed beta change or learned key geometry account for improved target-write transport, and does either improve answers when value/head weights are fixed?

Use original seeds 3461/3463, exhaustive-feedback checkpoints at 3000 and 12000. Every original train-32 history, same order and all queries: 768 histories by 32 unique keys. No held-out evaluation, gradients, training, solver or extra donor values.

For each seed fix value embeddings and affine classifier to one anchor stage, then the other. At each anchor evaluate all four combinations of key-table stage (3000/12000) and beta stage (3000/12000), 2 seeds x 2 anchors x 2 key sources x 2 beta sources = 16 cells. Query keys and writing keys change together because they are tied. Exchange key.weight and write_logit only in private copies; preserve checkpoints, anchor/donor models and RNG. A joint rotation of tied normalized keys leaves zero-start reads unchanged, so key-table replacement probes geometry rather than key coordinates relative to the value/head space.

Reuse the frozen stage evaluator and independent matrix/write-basis auditor. Verify native diagonals against audited saved traces, all coefficient equations, discrete winners and source-weight scope. Exercise initialized saved evaluator/audit, identical-donor null controls, supplied ties/empty groups and deliberate source/label/logit/coefficient/count/contrast corruption before formal execution. Protect the executed dependencies and scoped source artifacts.

Keep step-12000 cohorts fixed: all, endpoint wrong, correct in incomplete histories, correct in complete histories. Report all groups and all 32 target positions, including empty groups. Outcome measures: final accuracy/CE/maps, post-write accuracy, target coefficient ratio, target/other score contributions, and rescued/harmed answers versus the anchor-native cell. Coefficient ratio is signed sensitivity, not retained information. Native results remain historical training-fit measurements; hybrid scores are counterfactual inference, not fitted models.

For a metric f let f00/f01/f10/f11 denote early keys/early beta, early keys/late beta, late keys/early beta, late keys/late beta. Report beta effect f01-f00 and f11-f10, key effect f10-f00 and f11-f01, and interaction f11-f10-f01+f00. Use target coefficient ratio, final accuracy and CE. No additive causal percentages or selection of a best recipe. Signed transition/change guard 1e-9, equation absolute/relative 1e-9 with matching discrete winners. Counts derive independently from factor lengths.

Stop after 16 evaluations and one audit. This isolates computation at fixed weights, not why training selected the weights or whether a swap is a generalizable repair. Two opened development seeds and outcome-selected cohorts are not fresh confirmation. Preserve original 246/353 errors, last-write exceptions and earlier all-load affine certificates. RL remains parked.

Source inspected: [Yang et al. (2024), sections 2.2 and 3.2](https://arxiv.org/html/2406.06484v2). Published recurrence/unrolling underlies target transport; factorial contrasts and rotation invariance are local derivations. Scalar beta/tied keys differ from the full paper; no source predicts local scores.
