# Copy learning: rank failure, not a sharpening-only explanation

Saved-score review, 2026-10-07. **The correct source position is outranked in at least one head on 314 of 351 random-long failures (89.46%).** This argues against selecting a sharpening-only remedy. It does not prove a unique cause or predict the output of a sharpening intervention.

## What was checked

Use only native E1M1 tensors from the audited [visibility experiment](../topics/benchmarks/copy-source-visibility-results.md). No new forward passes, gradients or training. For each head, compute

\[
g_h=s_{h,0}-\max_{j>0}s_{h,j}.
\]

Negative g means another source position outranks position zero. Values within 1e-10 of zero are classified separately as near ties; none occur in the random-long cohorts. Source position is not a value label: repeated symbols and contextual encoder values make the distinction important.

| Seed | Length | Wrong outputs | Position zero first in both heads | Outranked in one head | Outranked in both |
|---|---:|---:|---:|---:|---:|
| 3163 | 8 | 57 | 1 | 23 | 33 |
| 3163 | 10 | 56 | 1 | 24 | 31 |
| 3167 | 8 | 79 | 20 | 35 | 24 |
| 3167 | 10 | 64 | 10 | 31 | 23 |
| 3169 | 8 | 41 | 4 | 12 | 25 |
| 3169 | 10 | 54 | 1 | 14 | 39 |
| Total | | 351 | 37 | 139 | 175 |

Successful random-long outputs provide the counterexample: of 417, 299 rank position zero first in both heads, 107 are outranked in one and 11 in both. All 1,344 train6/validation6/test6 seed-example evaluations rank position zero first in both heads. All uniform examples are correctly predicted even though some outrank position zero. Thus routing rank is informative but neither necessary nor sufficient for a correct output.

Positive score scaling preserves order: for c>0, argmax(cs)=argmax(s). It cannot correct these attention rankings. It could still change final predictions through mixtures and downstream layers; no such intervention was run. The earlier two-way suggestion, dilution versus incorrect ranking, was too simple to exhaust the causes.

## One candidate learning test

**Hypothesis:** narrow positional-displacement exposure during short training permits routing rankings that fail at larger displacements. This is plausible from the short/long contrast, not established by it. Extra competing content and other learned features remain alternatives.

Compare ordinary short training with short training using shared, ordered, randomly spaced position IDs for corresponding source/decoder positions. Keep token sequences, targets, architecture, updates and exposure matched. Do not add longer content sequences or oracle attention targets. This changes positional exposure and training activations; it is not a pure optimizer change.

The predicted result must include fewer outranked first-source positions on fresh longer inputs, preserved short fitting, and improved full-copy performance. Accuracy improvement without the ranking change would not support the proposed explanation. Better rankings without better copying would show that this change is insufficient. A null or harmful result would leave the cause open.

This is a candidate, not a frozen training protocol. Before execution, inspect the predecessor's full method, bound recipe development, reserve fresh confirmation, and freeze position sampling, budgets and all-group metrics. No run or parameter grid is queued.

## Sources and scope

[Ruoss et al. (2023), abstract inspected](https://arxiv.org/abs/2305.16843) already proposes ordered randomized positional exposure for length generalization, including relative encodings. Our candidate is a mechanism test using an existing idea, not a novel method or verified reproduction. [Chiang and Cholak (2022), sections 5.1-5.3](https://aclanthology.org/2022.acl-long.527.pdf), inspected in the preceding diagnostic, explains competing values and proposes attention scaling; its FIRST setting does not settle our routing question.

The [declared reduction plan](../../results/copy_score_review_20261007/plan.json) preceded tensor inspection. The [report](../../results/copy_score_review_20261007/report.json) preserves all eight pools, three seeds, original distinct/repeated groups, empty cohorts and correct/wrong outcomes: 2,184 seed-example evaluations, with overlapping summaries. Scalar and tensor gap counts agree; reconstructed QK scores and scalar softmax match saved tensors within 4.441e-16. Four input hashes remain unchanged. Known rank/tie fixtures and an inverted-ranking control were checked. This is a checked descriptive reduction, not a separate independent model audit or new confirmation.

[Reproducible reduction source](../../results/copy_score_review_20261007/inspect_scores.py). Stop score partitions here; move only to a bounded hypothesis-driven training design.
