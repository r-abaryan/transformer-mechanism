# Remaining state-tracking errors

Frozen before evaluation, 2026-10-03. Inspect final constant_001 and cosine_001 checkpoints for development seeds 2903/2909. No training, model selection, validation or test evaluation.

Use all six saved training pools. Compare paired fixes and regressions on identical examples. Stratify errors by suffix FLIP count, last SET position (-1 means none), and distance from the final operation to the last non-KEEP operation after the reset. Report exact stratum counts; these descriptive strata do not isolate causal factors.

Reuse the earlier initial-bit, first/middle/last eligible KEEP/FLIP toggle and two-suffix toggle interventions. An initial-bit change matters only without a SET. A toggle matters only after the last SET. Two suffix toggles preserve parity. Report label-changing and label-preserving cases separately, including both-original-and-intervened-correct accuracy. Intervention inputs can be outside the original group distribution and are not held-out confirmation.

Check every original/intervened prediction against the independent full-model equations, all labels against last-reset/parity and recurrent oracles, and summary metrics against Python scalar reductions. Detect the intentionally wrong rule that every single change flips the label. Original logits must match archived scores. Hash targeted inputs, checkpoints and source dependencies before/after. Stop after this diagnostic and one short educational report. No mechanism, convergence or generalization conclusion from response consistency alone.
