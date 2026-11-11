# State tracking: what must the model learn?

Recall retrieves a stored association. This task applies a sequence of updates to one binary state. We train the existing architecture from scratch; earlier learned weights are not reused.

| Operation | New state | Meaning |
|---|---|---|
| KEEP | `s` | Preserve the bit |
| FLIP | `1-s` | Toggle it |
| SET0 | `0` | Forget the previous state |
| SET1 | `1` | Forget the previous state |

Example: initial0, SET1, FLIP, SET0, FLIP gives `0 -> 1 -> 0 -> 0 -> 1`. The final answer is1. The model sees the initial state and operations, then a QUERY token. It receives no intermediate answers.

## The math

Each operation is `s' = a*s XOR b`, with binary coefficients. KEEP, FLIP, SET0 and SET1 correspond to `(1,0), (1,1), (0,0), (0,1)`.

Applying `(a1,b1)` then `(a2,b2)` gives:

`s'' = (a2*a1)*s XOR ((a2*b1) XOR b2)`.

Order matters: SET0 then FLIP gives1; FLIP then SET0 gives0. Composition is associative, but generally not commutative. A correct answer alone cannot show whether a model uses sequential updates, parallel composition or another solution.

An independent exact solver uses a simpler rule: take the last SET value, then toggle it once for each later FLIP. Without a SET, start from the initial state. An even number of FLIPs cancels; an odd number changes the bit. This solver supplies ground truth, not learned-model evidence.

## Why three groups?

| Group | What it tests | Main limitation |
|---|---|---|
| Mixed resets | Locate relevant recent updates | Late resets erase most earlier history |
| No resets | Combine FLIPs across the entire sequence | A small binary task, not general reasoning |
| One early reset, long tail | Discard the old state, then combine later FLIPs | Reset location is fixed in this diagnostic |

Every example has a partner with the initial bit and SET values inverted. Its target flips too. Both stay in the same data split. This balances labels and prevents a constant answer from exceeding50%.

Training uses8/12 operations; evaluation also uses16/24. Each seed and group must pass separately. Longer inputs change both sequence length and required dependency span in the no-reset/long-tail groups; this experiment does not isolate those two effects.

## Reading the outcome

Success supports this finite test distribution, not arbitrary-length computation. Failure identifies a capability boundary for this recipe and budget. It does not prove transformers cannot track state, or identify an attention/FFN/position defect. We stop after the fixed three runs and audit.

[Protocol and research context](../../protocols/state-tracking.md). The2023 flip-flop and2025 state-tracking papers motivate the question; this task and its thresholds are local, not paper replications.

**Observed outcome:** [three completed runs](state-tracking-results.md) perform much better with mixed resets than without them. At eight operations, mean test accuracy is95.31% with mixed resets and46.88% without resets. Reliable state tracking fails already at training lengths. All13,824 final validation/test predictions pass independent verification. Study closed without tuning.
