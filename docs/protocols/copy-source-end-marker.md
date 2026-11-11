# Source-end marker: saved-checkpoint diagnostic

2026-10-03. Question: does a distinct visible source-end token break the uniform-source logit collision found in the RoPE copy comparison? This checks available information and checkpoint sensitivity, not learned copying after marker training.

Use all six saved sinusoidal/RoPE checkpoints, seeds 3119/3121/3137. No training, optimizer changes, model selection or original-pool scoring. All eight content symbols, length pairs (3,4) and (6,8). Common decoder prefix is BOS followed by the shorter number of symbols. Required next labels are EOS for the shorter source and the content symbol for the longer source.

Four encoder formats: original content; content plus EOS ID2; content plus EOS masked in both encoder and decoder memory; content plus another copy of the content symbol as an intentionally indistinguishable marker. Prefix and target meanings remain unchanged: appended tokens represent a source boundary and are not copy targets. Source EOS was never a source training token in these checkpoints; this is an out-of-training-format intervention, not a deployed repair.

Record every next-token logit, argmax, EOS probability, EOS-versus-content margin and both-label correctness. Distinguish a logit collision (maximum absolute difference <=1e-9) from a detectable distinction (>1e-6). Intermediate differences satisfy neither category. A marker-distinction criterion requires >1e-6 for every RoPE seed/symbol/length pair. Do not interpret it as accurate copying or universal injectivity.

Check native outputs against explicit rotation/manual-attention and separate layer equations at 1e-9 with zero reference argmax differences. For RoPE, original, same-symbol marker and masked marker must retain their length collision. Both-mask exclusion must match original logits in both architectures. Verify checkpoint/dependency hashes before and after. Compare original RoPE outputs with the already saved witness. Projection/shape utilities are shared with the reference, as disclosed in the original comparison.

Stop after 192 paired cells, 384 source/prefix examples. No new training or wider marker/position grid is selected. Preserve original quality results and criteria. Write one short educational note with results, the value-averaging equation, controls and unknowns. This diagnostic does not increase the count of trained copy comparisons.
