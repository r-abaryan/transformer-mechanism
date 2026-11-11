# Direct paired-key forcing without compact donors

Fixed before execution. Five frozen models, same 5,120 inspected six-derived crops/model. Test shifts 0 and +4. Intervene only on block-1 attention rows 3/5 in both heads: original, force paired key, force self, force immediately earlier nonpaired key. Row 1 unchanged. Sources are row-1, row and row-3 respectively; no future access.

Compute all Q/K/V from the current input. Replace selected probability rows with one-hot rows; project weighted V normally. Let block-1 residual/FFN and all block-2 Q/K/V/residual/FFN operations execute normally. No compact donor activations and no weights updated. Shift 0 is a separate harm control, never a donor.

Primary rescue criterion: every seed >=90% at +4 with paired-key forcing. Control advantage: mean paired-key accuracy exceeds each self/other-key condition by >=10 points at +4. Compact preservation: each seed loses <=1 percentage point relative to its own unmodified compact accuracy. Report each separately, every seed, loss and paired rescue/harm. Reused crops are dependent and this is exploratory.

Verify explicit original attention and final logits against unchanged forward (<1e-10), saved shift baselines within 1e-10, causal zeros, row sums, finite outputs and checkpoint integrity. Preserve protocol/source/input hashes. This uses known task layout and deterministic forcing; success is not a novel architecture, general repair or evidence on uncropped longer inputs.
