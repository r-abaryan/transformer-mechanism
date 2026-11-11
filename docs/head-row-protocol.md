# Localizing head-output effects by row

Fixed before execution. Reuse all five frozen pre-norm constant-rate 6,000-update models and the opened 400 mappings (7,200 variants) from head-ablation. No training or head selection. Compare each head and both heads in each block.

Row groups: value rows 1/3/5; final-query row 7; remaining rows 0/2/4/6; all rows as a reproduction control. These groups refer to output query rows, not attended source columns. Remove the selected projected head contributions only at these rows. Preserve shared output bias and recompute downstream operations. Twenty-four conditions per seed, 120 total, plus reused ordinary references.

Reuse the verified explicit head decomposition to cache ordinary contributions. Subtract them with an attention-output hook; a single intervened block has unchanged upstream inputs. Independently verify using library attention with selected output-projection columns zeroed, mixing its output into only selected rows. All-row results must match prior logits within 1e-10. Checkpoint hashes must remain unchanged.

Report accuracy, loss, mapping correctness, rank accuracy and paired prediction changes/harms/rescues. Compare error-set overlap between value-only and all-row removals descriptively; no equivalence or significance claims. Effects need not add across row groups.

Structural control: block 2 is the last mixing layer. Removing its non-final output rows cannot affect the final prediction; removing its final-query output row must match all-row removal. Verify logits within 1e-10. This is an architectural identity, not a newly discovered mechanism.

Stop after this localization. Value-row damage would localize a dependency, not prove preceding-key content was transmitted. These are exploratory zeroing interventions on reused data, not fresh confirmation or a model improvement.
