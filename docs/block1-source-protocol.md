# Block-1 source interventions

Predeclared diagnostic. Five frozen models, same 5,120 inspected crops/model. Keep previous downstream construction: candidate y=shifted x + intervened block-1 attention + saved shifted FFN; derive block-2 K at value slots only; compact final Q, shifted V and recipient execution.

Reconstruct block-1 attention for compact and shifted states. Report per-head attention probabilities at value positions 1/3/5: paired key, self, and sum over earlier nonpaired keys. Observation alone is not causal proof.

For each state, intervene on value rows 3 and 5 in both heads. Leave row 1 unchanged because it has no earlier nonpaired key. Conditions: unmodified; force paired key (row-1); force self (row); force immediately preceding nonpaired key (row-3); remove paired-key attention; remove self-attention. Forced rows are one-hot. Removed-source rows are renormalized softmax over remaining allowed sources. Recompute weighted V and unchanged output projection. Preserve original V for the chosen compact/shifted state. All interventions are causal by token order.

Compare >=90% accuracy for every seed per forcing condition. Descriptive harm criterion for paired-key removal: >=10-point loss in every seed relative to that state's unmodified donor construction; self removal is the comparator. Report all seeds, loss and paired rescue/harm. Row 1 retention prevents claims about all value positions. Forcing alters entire distributions and is not magnitude-matched; other-key controls can carry useful repeated answer values. Results are conditional and exploratory, not necessity or a full circuit proof.

Verify block-1 attention reconstruction against library (<1e-10), causal zeros/row sums, finite outputs, unchanged checkpoints, and unmodified compact/shifted construction against frozen_010/frozen_000 (<1e-10). Save source/protocol/input hashes; no training or automatic architecture change.
