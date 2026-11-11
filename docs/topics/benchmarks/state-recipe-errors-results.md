# What remains wrong after better state-tracking fit?

Executed 2026-10-03. Four saved checkpoints, two paired seeds, all 1,920 training examples. No training or held-out evaluation. [Protocol](../../protocols/state-recipe-errors.md) | [Source](../../../experiments/state_recipe_errors.py) | [Raw evidence](../../../results/state_recipe_errors/report.json).

**The schedule improves parity responses, but reset-aware behavior is still unreliable. High original accuracy hides sensitivity to an irrelevant initial bit.**

## The rule the model should follow

KEEP leaves the bit unchanged. FLIP toggles it. SET0/SET1 replaces it. Let b be the last SET value, or the initial bit if there is no SET. Let f count subsequent FLIPs. The answer is `b XOR (f mod 2)`.

This gives three controlled checks: one relevant toggle must change the answer; two relevant toggles must preserve it; changing the initial bit after a SET must preserve it. We recompute labels rather than assuming every edit changes the answer.

## Which errors did the schedule fix?

Same examples, starting weights and training streams. Counts below use 512 examples per cell. Fixed means constant wrong, scheduled correct; regressed means the reverse.

| Seed | Twelve-operation group | Constant errors | Scheduled errors | Fixed | Regressed |
|---|---|---:|---:|---:|---:|
| 2903 | No resets | 60 | 25 | 44 | 9 |
| 2903 | Long tail after reset | 48 | 31 | 22 | 5 |
| 2903 | Mixed resets | 7 | 6 | 5 | 4 |
| 2909 | No resets | 233 | 25 | 225 | 17 |
| 2909 | Long tail after reset | 241 | 25 | 227 | 11 |
| 2909 | Mixed resets | 17 | 2 | 16 | 1 |

Improvement is substantial, but not every previously correct example stays correct. All shorter cells are in [paired.csv](../../../results/state_recipe_errors/paired.csv).

## Better parity responses, incomplete consistency

For twelve-operation no-reset examples, each intervention has 512 cases. Response agreement means the predicted class changes exactly when the true label changes. It can count two wrong answers as consistent, so it is not accuracy.

| Intervention | 2903 constant | 2903 scheduled | 2909 constant | 2909 scheduled |
|---|---:|---:|---:|---:|
| Initial bit: must change | 77.34% | 92.58% | 25.39% | 90.23% |
| First eligible toggle: must change | 77.34% | 87.50% | 11.13% | 92.77% |
| Last eligible toggle: must change | 75.39% | 89.06% | 24.80% | 92.19% |
| Two suffix toggles: must preserve | 86.33% | 91.80% | 76.56% | 95.31% |

Both-correct rates for scheduled two-toggle pairs are 90.23%/93.55%. Thus improved consistency accompanies better answers, but exact parity behavior is not established. Interventions reuse examples and may change their original distribution. [All response and accuracy metrics](../../../results/state_recipe_errors/interventions.csv).

## The reset cue reveals a hidden weakness

Every original long-tail example starts with a SET whose value equals the initial bit: 128/128 examples at eight operations and 512/512 at twelve. This follows from the generator's complementary pairing and is checked directly in the saved snapshot. That correlation makes the two cues indistinguishable on the original examples.

Changing only the initial bit breaks the correlation while leaving the correct answer unchanged:

| Scheduled checkpoint | Twelve-operation group | Cases | Answers changed incorrectly | Both answers correct |
|---|---|---:|---:|---:|
| 2903 | Long tail | 512 | 83.01% | 11.52% |
| 2909 | Long tail | 512 | 71.48% | 24.02% |
| 2903 | Mixed resets | 512 | 4.88% | 94.34% |
| 2909 | Mixed resets | 512 | 3.71% | 96.09% |

This establishes behavioral dependence on an irrelevant initial cue under the intervention. It does not identify the internal circuit or prove that the correlation caused learning failure. Mixed groups also differ in reset placement and suffix length. Earlier scores remain valid on their original distributions; they are not evidence of robust reset handling.

## Remaining errors differ between seeds

Both scheduled checkpoints make 25 errors on the same 512 no-reset examples. Their error locations differ:

| Suffix FLIPs | Cases | Seed 2903 errors | Seed 2909 errors |
|---|---:|---:|---:|
| 3 | 46 | 0 | 23 |
| 8 | 78 | 6 | 0 |
| 9 | 26 | 15 | 0 |
| 10 | 4 | 2 | 2 |

Equal aggregate accuracy hides different weaknesses. Flip count alone is not a common explanation. Last-reset positions and last-active-operation distances are also recorded in [strata.csv](../../../results/state_recipe_errors/strata.csv); their observational bins do not isolate causal effects, and some contain only two examples.

## Verification and next decision

All 44,096 original/intervened example evaluations agree with independent model equations, zero argmax changes, maximum logit error 1.41e-13. Archived original logits match exactly. Scalar response metrics agree within 6.58e-14. The wrong rule that every single edit changes the label fails on 2,800 prepared cases. Targeted input/source/checkpoint hashes are unchanged. One initial launch stopped before evaluation or artifact creation because Windows path separators mismatched a lookup; corrected before the successful run.

No recipe is adopted and the previous 99% gate stays failed. The next useful question is whether removing the initial/reset correlation during training improves reset invariance. That would be a separately frozen exposure comparison with independent initial and reset values, paired budgets and a training-fit gate. It is not executed here. Avoid widening the optimizer grid before addressing this concrete cue ambiguity.
