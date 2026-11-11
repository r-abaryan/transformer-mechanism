# Excluding added read positions gives partial recovery

2026-10-07. **The weak model recovers 17/128 fifth-token answers (13.28%). The other 111 still predict EOS.** The frozen 95% recovery criterion fails; all successful-control groups preserve accuracy.

[Protocol](../../protocols/copy-added-read-mask.md) · [Report and audit](../../../results/copy_added_read_mask/report.json) · [Every group](../../../results/copy_added_read_mask/metrics.csv)

## What changed, and why

The previous [paired-prefix test](copy-prefix-extension-results.md) changed the weak model from correct content to EOS on all 128 examples when the source grew from six to ten symbols. Here we keep the full ten-symbol encoder computation and hide only the four added content positions from decoder cross-attention.

Both heads retain source indices 0..5 and the actual source EOS at index 10. Indices 6..9 are masked. Encoder representations and the decoder state before cross-attention are exactly unchanged. These retained representations can still contain information about the hidden suffix.

For retained set R, the local intervention computes `a_j = exp(s_j) / sum(exp(s_k), k in R)` for j in R, and zero otherwise. The output remains `sum(a_j v_j)`. Removing positions also renormalizes the remaining weights; this does not isolate competitor count from changed content mixing.

## Measured results

Each model receives the same 128 opened long sources and correct prefix `BOS + first four symbols`. The target is the fifth source symbol. Margin means `EOS logit - correct-symbol logit`; positive values favor EOS over that symbol.

| Model | Native correct | Masked correct | Masked EOS | Rescues / harms | Mean margin, native → masked |
|---|---:|---:|---:|---:|---:|
| Cross-only 3767 | 0/128 | 17/128 | 111/128 | 17 / 0 | 6.700 → 3.769 |
| Cross-only 3779 | 128/128 | 128/128 | 0/128 | 0 / 0 | -11.442 → -11.442 |
| Cross-only 3793 | 128/128 | 128/128 | 0/128 | 0 / 0 | -11.091 → -11.091 |
| Combined 3767 | 128/128 | 128/128 | 0/128 | 0 / 0 | -11.342 → -11.342 |

The weak model recovers 1/10 distinct-prefix and 16/118 repeated-prefix examples. Each successful control preserves 10/10 and 118/118. Thus control preservation passes, but recovery and the joint support rule fail. The weak mean margin decreases by 2.931; that shift is insufficient for most answers.

## What we learned

Excluding added read positions changes some failed answers with the long encoder fixed. It is not sufficient for broad recovery. The result leaves retained encoder representations, marker position and downstream interactions unresolved. It does not establish that the extra positions are irrelevant, or identify why training learned this behavior.

This is one supplied-prefix decision, not full generated copying or a learned repair. Seven examples receive prefixes beyond their earlier native stopping point. The pool is reused and selected from a known failure; this is not fresh confirmation.

## Verification and sources

Four models × 128 examples × two modes = 1,024 selected predictions per implementation, with zero training updates. Saved initialized reload, empty-group, tie, rescue/harm and mask controls passed before scoring. A selected-key reference agrees within 5.951e-14, with zero argmax differences; native logits also match the previous saved evidence. All 54 protected files remain unchanged. The reference shares projection/attention helpers, so its independence is partial.

Preflight initially required both perturbation signs to expose a missing mask. A negative score can suppress an unmasked key, so that assertion was corrected before scoring: both signs must preserve masked outputs, and a missing mask must be detected by at least one sign. No empirical criterion changed.

[Newman et al. (2020), sections 1 and 3](https://arxiv.org/html/2010.07174v1#S3), inspected in the preceding review, motivates separating termination from content. The mask equation, intervention and scores above are local work, not predictions from that paper. Stop this diagnostic without another mask or training sweep.
