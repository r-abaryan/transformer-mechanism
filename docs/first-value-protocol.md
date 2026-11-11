# First-value position diagnostic

Fixed before execution. Same frozen models, inspected 5,120 crops/model and shifts 0/+4. No training or compact donors. Full downstream execution. Force block-1 attention in both heads using known pair positions.

Conditions: original; rows 3/5 paired (prior control); row 1 paired only; row 1 self only; rows 1/3/5 paired; row 1 self with rows 3/5 paired. Paired source is row-1; self source is row. Row 1 has no earlier nonpaired key. This factorial tests its contribution with later rows unchanged or paired-forced.

Primary criterion: all-three paired forcing reaches >=90% at +4 for every seed. Preservation: compact loss <=1 percentage point for every seed. Report incremental effects of row-1 paired/self forcing on both backgrounds, every seed, loss and paired rescued/harmed counts. Also report accuracy by queried pair rank 0/1/2; these are descriptive subgroups, not independent trials. Do not tune after outcomes.

Verify prior original and rows-3/5 metrics, explicit attention sham (<1e-10), labels, finite outputs, causal rows/normalization, immutable checkpoint hashes and input/source/protocol hashes. Known layout and reused cropped examples limit generalization. Passing requires separately predeclared fresh-example/additional-shift confirmation; failure is retained.
