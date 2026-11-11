# Is the remaining error reset handling or parity?

Executed 2026-10-03, saved independent-exposure checkpoints only. [Protocol](../../protocols/state-reset-response.md) | [Source](../../../experiments/state_reset_response.py) | [Raw grids](../../../results/state_reset_response/report.json) | [All pattern counts](../../../results/state_reset_response/patterns.csv).

**Seed 3019 mostly fails through mixed initial/reset responses, rather than a pure initial-only rule. It also responds less reliably to suffix toggles.** The stronger seeds have different failures at longer lengths. No new training or model selection.

## Why four inputs help

Fix one suffix and vary both initial bit i and reset value r. The answer should be `r XOR parity(suffix)`. A suffix family has all four `(i,r)` combinations. We require the whole grid to match, rather than accept a good average over partly correct cases.

Every binary response grid can be written as `f(i,r) = c XOR (a*i) XOR (b*r) XOR (d*i*r)`. Correct reset handling requires `a=0, b=1, d=0` and `c=parity(suffix)`. These coefficients summarize predicted classes; they are not network weights or an identified circuit.

## Which response pattern appears?

Each row has 128 held-out suffix families. Counts sum to 128. Correct/inverted reset ignores the initial bit and responds to r, with correct/incorrect suffix parity. Initial-only responds solely to i. Constant predicts the same class for all four inputs. Other means a mixed response. Constant class predictions do not imply identical logits or erased internal state.

| Seed | Operations | Correct reset | Inverted reset | Initial-only | Constant | Other |
|---|---:|---:|---:|---:|---:|---:|
| 3011 | 12 | 123 | 0 | 0 | 1 | 4 |
| 3019 | 12 | 66 | 1 | 1 | 2 | 58 |
| 3023 | 12 | 122 | 0 | 0 | 0 | 6 |
| 3011 | 24 | 9 | 8 | 4 | 24 | 83 |
| 3019 | 24 | 35 | 19 | 3 | 26 | 45 |
| 3023 | 24 | 3 | 13 | 0 | 102 | 10 |

The weak seed already has incomplete training behavior: only 159/256 twelve-operation training grids are fully correct, versus 250/256 and 248/256 in the other seeds. Initial dependence appears in 87/256 training and 59/128 test families. Of its 58 mixed test grids, 54 get three corners right and four get only one right. Aggregate accuracy hid these asymmetric failures.

An observed seed-3019 test suffix is `1,0,1,1,1,1,1,0,0,1,1`, with eight FLIPs. Its parity is even:

| Initial i | Reset r | Correct answer | Prediction |
|---|---|---:|---:|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 1 |
| 1 | 0 | 0 | **1** |
| 1 | 1 | 1 | 1 |

On this suffix, the predicted classes match OR(i,r), not RESET alone. This is a finite behavioral witness, not proof that the network implements a general OR algorithm.

## Controlled suffix changes

Toggle the first suffix bit, the last, or both. One toggle must change the answer; two must preserve it. Each test intervention has 512 inputs, balancing all initial/reset combinations. The table requires both original and intervened answers correct, preventing two consistent wrong answers from counting as success.

| Seed | First toggle | Last toggle | Two toggles |
|---|---:|---:|---:|
| 3011 | 97.07% | 96.09% | 95.70% |
| 3019 | 73.83% | 73.44% | 70.31% |
| 3023 | 96.09% | 95.31% | 95.90% |

Seed 3019 also misses the required response change/preservation: agreement is 81.05%/77.73%/75.98%. Thus remaining weakness includes both initial/reset interactions and incomplete suffix responses. Derived first/last/two-toggle suffixes overlap training families in 21/128, 18/128 and 13/128 cases. These are behavioral diagnostics from test originals, not new held-out scores. Flip-count strata and training interventions remain in [the run folder](../../../results/state_reset_response/); bins have unequal, sometimes tiny counts and do not isolate a cause.

## What is established and what remains open?

This rules out a single explanation that the weak model simply substitutes the initial bit for RESET everywhere. It localizes the behavioral failure to particular suffix/cue combinations. At length 24, seed 3023 instead gives a constant class on 102/128 grids, while seed 3011 mostly gives mixed responses. Similar weak accuracy does not establish a shared failure mechanism.

We have not identified which attention, residual or FFN path produces these interactions, or why training leaves them. Attention maps alone cannot answer that. A next mechanistic intervention would isolate the initial-token contribution within the weak checkpoint, using matched input pairs and verified activation substitutions. That is not run here; no tuning or architecture change is selected.

All 2,064 cached grids match independent truth-table classification; all sixteen binary functions are checked at both parity values. The wrong initial-instead-of-reset rule fails on 4,128 labels. All 18,432 original/intervened evaluations match independent model equations, zero argmax changes, maximum logit error 8.53e-14. Original logits match archives exactly; scalar response metrics agree within 4.62e-14. Input/source/report/checkpoint hashes are unchanged. These are previously opened inputs and three training seeds, not fresh confirmation or 18,432 independent experiments.
