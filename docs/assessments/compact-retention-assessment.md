# Compact retention: what changed our understanding

2026-10-04. This closes the current retention branch. Scope: a small delta reader, known keys and consistent six-class assignments. No general Transformer/DeltaNet ranking or new architecture is established.

## Evidence chain

| Question | Measured answer | Limit |
|---|---|---|
| Does more association load hurt? | [Load study](../topics/attention/association-load-results.md): delta test falls to 58.42% at 32 keys; compact softmax stays at 100% | Different reader memory; load also changes repetitions |
| Does more repetition repair it? | [Repeated histories](../topics/attention/history-repetition-results.md): helps at moderate loads, fails the high-load repair criteria | Extra writes can rescue or harm |
| When do wrong answers appear? | [Write-order trace](../topics/attention/query-write-order-results.md): all selected immediate reads correct; later writes leave 479 wrong | Query-last recovery changes input order using the requested query |
| Can separated keys retain everything? | [Geometry intervention](../topics/attention/key-geometry-retention-results.md): imposed orthogonal keys give 100% answers and maps | Supplied known-key slots double the state; not a learned original-width repair |
| Does a low-overlap penalty suffice? | [Penalty training](../topics/attention/overlap-regularization-results.md): fresh mean all-query test 62.19% to 65.79%, zero complete maps | Failed benefit/retention rules despite near-minimum average overlap |
| Does supervising more queries help? | [Query exposure](../topics/attention/delta-query-coverage-results.md): development validation mean 61.39% to 85.90% | All-query training remains about 90%; no fresh confirmation |
| Does write strength contribute? | [Beta swap](../topics/attention/beta-swap-results.md): lowering beta gains 18.46/17.76 points; raising it loses 17.84/19.08 | Frozen inference intervention; complete-map reliability still fails |

Scores across different studies are references, not paired controls. The reported swaps are paired within each saved checkpoint.

## What we learned

**Fitting one assigned query does not mean storing an entire map.** Fixed-query training reaches 100% on its assigned questions while other reads from the same state fail. Changing query exposure makes this distinction visible and improves many answers.

**A successful write does not guarantee a surviving answer.** Earlier traces and the beta intervention show correct first-write classifications becoming wrong later. This is classification damage, not proof that all relevant information has vanished.

**Write strength is a contributing factor.** Smaller beta improves fixed-model retention, and the reverse intervention harms resampled models. Keys and the decoder also matter: resampled models keep a 6.40-point mean advantage when both arms use the same lower beta. That remainder is not a uniquely attributed cause.

**Individual accuracy and complete-map reliability differ.** High average answer accuracy still leaves almost every 32-key validation history with at least one error. Neither broader exposure nor beta substitution meets the declared complete-map criterion.

## Missing exposure is not the whole explanation

A [post hoc check of saved training scores and exposure counts](../../results/beta_swap/exposure_background.json) finds exposed-query accuracy 89.94%/90.17%, versus 88.58%/89.21% for never-supervised history/key combinations. About 84.55%/84.57% of errors occur in exposed combinations, which also form most of the pool. These cohorts are observational, not randomized treatments. They reject an explanation consisting solely of unsupervised combinations; they do not measure the causal value of additional exposure.

## Mathematics, unknowns and stop

[Yang et al., DeltaNet 2024, section 2.2](https://arxiv.org/html/2406.06484v2) supplies the inspected correction/read rule. The local change `delta r(q)=beta*(q dot k)*(v-S^T*k)` connects overlap, write error and strength. It permits interference but does not predict our scores or prove a classification capacity ceiling. Our global scalar and standalone reader differ from the paper's input-dependent beta and full model.

Unresolved: why training selects these joint solutions; which remaining failures reflect finite optimization/exposure versus representation; and whether a learned compact repair can fit all-query training and pass fresh confirmation. The current evidence does not settle those questions.

The explanatory gain is concrete: supervision changed the learned retention behavior, and beta has a verified contribution under frozen weights. Close this branch without more beta, geometry or exposure grids. Future training requires a separately justified fit-development protocol; broader adoption requires fresh quality/resource evidence.
