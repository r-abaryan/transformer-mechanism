# State-tracking sensitivity diagnostic

Frozen 2026-10-03 before checkpoint interventions. Question: do the saved models respond to logically relevant state changes, and does that response depend on position or FLIP count?

Use all three original checkpoints (2003/2011/2027), full train 8/12 and test 8/12/24 pools, all three groups. These are reused diagnostic inputs, not untouched confirmation. Keep weights fixed, CPU float64, batch 128, one thread.

For each original input: flip the initial-state bit; toggle KEEP/FLIP at the first, middle and last eligible operation (middle uses floor(count/2)); and toggle two operations after the last SET, the first and last, when at least two exist. The last control leaves final parity unchanged. Do not toggle SETs. Report skipped cases and actual position distributions. Derived inputs may overlap prior pools or each other; they are not independent samples.

Independent last-SET/parity labels must agree with sequential transitions for originals and every intervention. An initial-bit change flips the label exactly when no SET exists. One toggle changes the label exactly when after the last SET. Two chosen suffix toggles always preserve the label. Classify single-toggle results by label relevance so before-reset invariance is not confused with neglect. Include a deliberately wrong label rule, and archive its detected disagreements.

Measure original/intervened accuracy, both-correct fraction, predicted class changes, logical response agreement (predicted XOR matches label XOR), mean absolute change in class-one probability and in logit gap. Report by seed/pool/intervention/relevance. Class responses alone can agree while both answers are wrong. Also stratify original training/test no-reset and long-tail accuracy by suffix FLIP count; do not interpret aggregate correlations as a cause.

Compare every native output with the previously verified independent full-model forward: <=1e-10 logit error and zero argmax changes. Verify original scores against their archives, targeted source/data/checkpoint hashes, independent scalar probability/metric equations and labels. No optimizer, training or source changes to original studies.

Evaluate explicit hypotheses descriptively: literal input neglect requires zero observable response; weak response is not proof of internal information absence. Compare first/last relevant-toggle sensitivity within each seed and pool before describing recent-position dependence. Count-specific success does not by itself prove counting. Stop after these interventions and one concise report with educational interpretation. Training development requires a subsequent concrete decision, not a silently enlarged diagnostic.
