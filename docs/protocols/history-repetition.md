# Whole-history repetition: frozen checkpoint diagnostic

2026-10-04. User authorized testing whether repeated exposure helps saved delta readers retrieve the same associations. No training, recipe development or architecture change. The closed association-load study remains the source of six unchanged confirmation checkpoints and already-opened test pools.

## Fixed intervention and count

Seeds 3323/3329/3331; softmax/delta; test loads 2/8/16/32; factors 1/2/4. Repeat the entire 32-event history as an ordered block, producing lengths 32/64/128. Preserve query, labels, distinct keys and every value assignment. No reordering, selective event duplication or error-selected histories. All 384 examples per pool are evaluated: 64 base contexts with six cyclic renamings. Cartesian count is 3*2*4*3=72 cells and 27,648 evaluated predictions; 48 paired comparisons against factor 1.

Repetition adds writes and processing length, but no new association information. Factors 2/4 are outside the training length. This diagnostic cannot independently distinguish every benefit/cost of extra writes from longer processing. Opened pools are not fresh confirmation for a learned repair; unchanged weights are the scope.

## Measures and interpretation

CPU float64, one thread. Use the existing `LoadReader` forward and full-pool metrics. Record every seed/load/factor: accuracy, CE, count, original-versus-repeated rescues/harms/both-correct/both-wrong, and maximum logit change. Original group tags describe original histories; report all-example metrics only.

Report each factor separately. Delta's descriptive criteria are: >=10-point gain at load 32 in every seed; >=95% load-32 accuracy in every seed; preservation within two points at loads 2/8 in every seed. These are separate findings, not an adoption gate. Do not choose a best repetition, revise thresholds or add factors after results.

Softmax has fixed zero recency and no other positional operation. For identical whole-history copies, both the unnormalized weighted-value numerator and denominator multiply by the repetition factor. Require max absolute logit change <1e-9 and zero class differences across tested factors. Mathematical invariance does not guarantee invariance for partial duplication, recency-biased or contextual readers.

Delta processes each repeated write again. A whole block is an affine transformation of the incoming matrix: `S -> A*S+B`, with ordered key-directed transitions. Starting at zero, two passes give `A*B+B`. The equations permit changed reads; they do not guarantee useful recovery or classify an empirical failure's cause.

## Preflight and audit

Freeze script/test/protocol and imported equation/reducer hashes before saved-reader evaluation. Preserve targeted earlier evidence, parent report/data/audit, all checkpoints and parent execution sources. The parent report, source hashes and independent audit must verify before running.

Four focused preflight methods exercise distinct-value softmax invariance, intentionally incomplete duplication, delta wrong `repeat_interleave` order, invalid factor, exact paired-count categories, incomplete Cartesian coverage, initialized saved reload and actual evaluator/audit entry, missing original groups, wrong labels/scalars and decision controls. Different values are explicit in routing controls so equal V cannot hide an error.

The post-run audit independently constructs repeated event lists, re-evaluates matrix-transition/explicit-exponential equations, reduces CE with Python math and reconstructs paired outcomes from per-example labels. Require equation error <1e-9, zero argmax disagreements, original metrics matching the archive and every protected file unchanged. Save raw score tensors plus JSON/CSV summaries. Shared weights/data/PyTorch do not constitute an independent software stack.

Stop after these 72 cells and audit. Preserve null or harmful results. No extra repetitions, order grid, retraining, wider state or novel proposal follows automatically.

Sources: local derivation from softmax normalization and the correction recurrence; [Vaswani et al. 2017, section 3.2.1](https://arxiv.org/abs/1706.03762), [DeltaNet 2024, section 2.2](https://arxiv.org/html/2406.06484v2). These papers define the operations, not the local intervention scores. [Parent task and findings](../topics/attention/association-load-results.md).
