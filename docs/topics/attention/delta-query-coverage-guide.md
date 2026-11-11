# What changes when the query changes during training?

Consider `A:2, B:5, C:1`. The memory should answer A with 2, B with 5 and C with 1.

| Training arm | Feedback each time this history appears | What perfect fit means |
|---|---|---|
| Fixed query | Always ask A | The assigned question is answered correctly |
| Resampled query | Draw A, B or C uniformly | Learn to answer every present key |

Both arms receive **one answer target per example**. We change which association is supervised. We keep histories, model size and the number of targets matched. Random sampling may leave some history/key combinations unobserved; the experiment counts them.

## Why this matters for delta memory

The published [DeltaNet correction rule, Yang et al. 2024, section 2.2](https://arxiv.org/html/2406.06484v2) motivates our update/read equations:

\[
S_t=S_{t-1}+\beta k_t(v_t-S_{t-1}^{\mathsf T}k_t)^{\mathsf T},
\qquad r(q)=S_T^{\mathsf T}q.
\]

Here S is 16x8, k and q have 16 coordinates, and v has eight. Our beta is one learned scalar; the paper's beta depends on the input. The classifier converts r(q) into six value-class logits.

Keys overlap. A later correction can change an earlier key's read. Supervising only one query per history can leave other answers wrong without increasing that history's supervised loss. Resampling exposes more of these errors to training. It does not guarantee that the available model or optimizer will solve them.

## The objective, briefly

For present keys Q in a history H, uniform query sampling gives the local identity:

\[
\mathbb E_q\,\mathrm{CE}(z(H,q),y(H,q))
=\frac{1}{|Q|}\sum_{q\in Q}\mathrm{CE}(z(H,q),y(H,q)).
\]

The sampled gradient is unbiased for this finite average. This is not a convergence guarantee. The preflight compares enumerated one-query losses and gradients with their all-query average.

**All-query accuracy** counts individual answers. **Complete-map accuracy** counts a history only when every present key is answered correctly from the same final state. A high first score can coexist with a low second score.

## Scope

The previous [overlap penalty](overlap-regularization-results.md) changed geometry while keeping fixed queries. This study changes query exposure with no penalty. Earlier [Transformer query exposure](../training/query-exposure-results.md) used a different model/task; it does not answer this compact-memory question.

The cited paper supplies update equations, not a prediction for local scores. [Measured results](delta-query-coverage-results.md) show improved development validation answers, but incomplete all-query training fit and no complete validation maps at 32 keys. The [frozen protocol](../../protocols/delta-query-coverage.md) therefore stops without fresh confirmation. More exposure helped; reliable retention remains unsolved in this setup.
