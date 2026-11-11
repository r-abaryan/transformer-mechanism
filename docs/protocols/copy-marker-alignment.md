# Source alignment in the marker-trained RoPE models

2026-10-04. Saved-checkpoint diagnostic; zero training updates. Does selecting the matching encoder position recover longer content or stopping? The original aligned-routing study used different, unmarked models. These models now fit training and pass short tests.

## Frozen scope

Use only the RoPE checkpoints with seeds 3163/3167/3169 in `results/copy_marker_training/report.json`. Preserve checkpoints, earlier code, data, protocols and scores. CPU float64, deterministic, one thread. Opened pools: train6, validation6, test6/8/10 and uniform6/8/10, with the original inputs and labels. Uniform probes are a challenge set, not independent confirmation; earlier overlap disclosures remain applicable. No new sampling or seed selection.

Encoder input is content+EOS. At decoder row j, the target is source position j, including source EOS at j=n. Conditions: native cross-attention; both heads forced to position j; both heads forced to (j+1) modulo (n+1), an indexing control. Beyond j=n, preserve native attention. Keep encoder, causal decoder self-attention, V/output projections, residuals, normalization, FFN and readout unchanged. No target token is inserted into logits. Alignment uses task knowledge and is not a deployable repair or a training comparison.

Correct-prefix evaluation uses BOS+content and measures content/EOS logits, CE, first incorrect row, per-position accuracy, correct-versus-best-other margin and native matching-position attention mass. Ordinary greedy generation uses its own prefix, all 16 token classes and cap n+3, stopping at EOS. Oracle content uses its own prefix for n steps, excluding EOS only. These are three distinct evaluation conditions.

Report all existing scalar metrics, per seed/group/position, including absent distinct groups at length10 and uniform sources. The descriptive recovery rule requires at least 10 percentage points of oracle-content exact gain in **every** random test8/10 seed/group, with no greater than 5-point ordinary full-copy loss in any train6/validation6/test6 seed/group. Report gains even if the combined rule fails. Full recovery means 100% ordinary exact copying in every long random seed/group. Teacher errors and attention maps alone do not establish the historical cause.

## Checks and stopping

Before scoring checkpoints, exercise initialized models at lengths6/8/10 for all three seeds and all three conditions: 27 numerical cases. Separate equations use explicit rotations/normalization and direct value indexing for forced conditions. Check outputs and all parameter gradients on small fixtures, native equivalence, EOS row selection, post-terminal preservation and a deliberately shifted selector. Exercise saved/reloaded initialization, final evaluator and independent audit entry, including absent groups and supplied known-answer logits.

Expected final design: 3 seeds x 8 pools x 3 conditions =72 cells. Derive counts from declared factors independently. Compare all teacher logits, native cross-attention weights, ordinary/oracle sequences and scalar/position/error-count metrics within 1e-9, with zero prediction differences. Native results must reproduce archived scores. Record relevant dependency/checkpoint hashes before and after. Stop after these cells, audit and a concise educational report; then consolidate and close the copy branch. No new training, head/token subdivisions or further position/LR search.

Interpretation: recovery demonstrates useful encoder states under supplied alignment. Failure means alignment alone is insufficient; it does not identify the readout as the sole cause. Shared equations/projection utilities limit audit independence. The Q/K rotation identity comes from [RoFormer, section 3](https://arxiv.org/html/2104.09864v5); this intervention and its outcomes are local evidence, not a paper reproduction.
