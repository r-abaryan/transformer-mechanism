# Following representations through fitting

Question: does the full-direction fitting advantage improve target-value coefficient rankings on both training and unseen maps, or only on training maps? Existing endpoint swaps establish compatibility effects; they do not answer how these margins evolve on fixed examples.

Frozen before formal evaluation, 2026-10-05. This is an exploratory saved-trajectory diagnostic on opened pools. No new training, architecture changes, optimizer variants or model selection.

## Data and scope

- All saved direction-study checkpoints: seeds 3473/3479/3481, full L-BFGS and negative-gradient arms, five stages per arm including shared start and final. Actual gradient counts differ by stage/arm; stage numbers are archive bins, not identical training time or update counts.
- Every existing training/validation/test history, all queries, loads 2/8/16/32. Per load: 768/192/384 histories. Labels are reconstructed from events independently. Values are constant within each key's history.
- Cartesian count: 3 seeds × 2 arms × 5 stages × 3 splits × 4 loads = 360 cells; 1,080 all/unique/repeat CSV rows. Query counts per checkpoint: (768+192+384) × (2+8+16+32) = 77,952; total 2,338,560 repeated evaluated answers. Six cyclic shifts and shared histories are correlated. There are three initialization seeds and one dataset, not millions of independent experiments.
- Report native accuracy/CE/complete maps and coefficient rankings at every stage. Groups: all, unique and repeated queries, including empty groups. Track cohorts fixed by shared-start native correctness. No final-error selection.

## Measurements and predictions

Use the exact local factorization S=C V, c=C^T q, z=(W V^T)c+b. Coefficient index j names the original value ID j; coefficients can be signed. Identity decoding argmax(c) is a declared fixed counterfactual, not the trained decoder or a necessary condition for correct retrieval.

For scores x and target y, define margin m(x,y)=x_y-max(j≠y)x_j and unit margin u=m/||x-mean(x)||₂. Define u=0 for an exactly constant vector. Unit margins remove positive scaling and common offsets. Report raw/unit means, 5th/50th/95th percentiles, paired changes from the shared start, and native/coefficient correct-disagreement counts. Argmax ties select the lowest class ID.

Primary comparisons: at final load 32, full minus negative-gradient **mean coefficient unit margin**, separately for each seed/split. A tolerance of 1e-10 treats unresolved numerical signs as non-positive; it is not a statistical significance threshold.

| Operational prediction | Frozen criterion | Limit |
|---|---|---|
| Broad target-ranking advantage | Contrast >1e-10 in every seed and all three splits | May coexist with wrong coefficients and harmful decoding; does not establish the historical cause |
| Training-only target-ranking advantage | Training contrast >1e-10 in every seed; both held-out contrasts ≤1e-10 in every seed | Applies to this explicit coefficient ranking, not all information accessible to another decoder |
| Mixed/unresolved | Neither criterion holds | Retain every seed and sign; no replacement hypothesis or threshold after scores |

Native unit-margin contrasts and accuracy/disagreement counts are descriptive checks. Neither signature by itself proves a decoder bottleneck, memorization or an architectural limit. Coefficient-correct/native-wrong cases show failures after the supplied identity rule; coefficient-wrong/native-correct cases demonstrate its limitation.

## Execution and audit

Native Windows CPU float64, one thread, batch 64 for inference. Evaluate all 30 saved checkpoints once per declared pool, then stop. Audit separately reconstructs matrix writes, labels, coefficients, logits, distributions and primary signs. Native archive cells must reproduce exactly. No fitting, gradients, calibration search, donor grid or new held-out selection.

Preflight must exercise the actual saved producer and auditor with initialized checkpoints, supplied ties/empty groups, positive-scale invariance, decoder counterexamples, and deliberate coefficient/logit/label/count/decision/missing-cell/missing-group/CSV faults. Preserve old sources/artifacts through scoped hashes. Source/preflight hashes are bound before formal scoring. Reusable checks live in tests; their arithmetic is not a trained finding.

Stop after full audit and interpretation. Continue only if the observations justify a specific learning intervention with competing predictions; otherwise retain the limitation and reconsider the question. No new run is automatically queued.

Sources inspected: [Yang et al. (2024), section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2), delta memory writes/reads; [Guo et al. (2017), section 4.2](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), positive logit scaling preserves decisions. The value-basis factorization, unit-margin screen and predictions are local derivations. Our scalar-beta/tied-query reader differs from full DeltaNet, and these measurements do not reproduce a paper's experiment or measure calibration.
