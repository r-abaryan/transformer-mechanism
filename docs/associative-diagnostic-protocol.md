# Associative recall: query sensitivity diagnostic

Defined after the first training seed showed roughly 54% test accuracy, before diagnostic execution. This is exploratory, not a new confirmation test. Do not tune or retrain.

On every three-pair test mapping, evaluate all three possible query keys using the frozen checkpoints. Record accuracy across queries, prediction-change rate relative to the original query, and accuracy restricted to examples whose alternative query has a different target. This separates predicting common values from responding to the query. Include deterministic most-frequent-stored-value baseline, tie broken by smallest value ID. This stronger baseline was not in the initial training protocol and must be labeled post hoc.

For each model, permute the values among the fixed key slots, leaving the query key unchanged. Rotate values by one slot and compute the new target by exact lookup. Record accuracy and prediction changes where the answer changes. These edits produce valid task episodes but are exploratory derivatives of the existing test set, not held-out confirmation data.

Observed query insensitivity or a strong frequency baseline does not prove an impossibility theorem for one-layer attention. A general claim would require a formal argument or different evidence.
