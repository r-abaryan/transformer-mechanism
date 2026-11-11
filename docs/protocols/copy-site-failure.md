# Localize the cross-only copy failure

2026-10-07. The [site comparison](../topics/benchmarks/copy-position-sites-results.md) finds a length-10 failure in cross-only seed 3767 despite complete short-task fitting. First inspect saved outputs; do not train or select another recipe.

Four saved controls: cross-only seeds 3767, 3779, 3793 and combined seed 3767. Seven pools each: train6, validation6, test6, test8, test10, uniform8, uniform10. Total 4×7=28 cells. Keep all distinct/repeated groups, including absent groups. These pools are opened evidence, not new confirmation.

Record the first generated mismatch, its 1-based position, premature EOS versus wrong content versus missed terminal EOS, correct-prefix content errors and correct-prefix terminal EOS correctness. Before the first wrong answer, generated and supplied-correct prefixes coincide; require the first teacher-forced and generated error to agree. Check archived full-copy counts and parent audit binding. Preserve row-level predictions/margins and input hashes. Check known examples of each error type, absent groups, ties through the existing lowest-index argmax convention, scalar/tensor agreement and deliberate corrupted answers. No new model forwards or updates.

Only after this reduction, select at most one intervention tied to the observed failure location. Freeze its operator, controls, metrics and stopping rule separately before scoring it. A selected endpoint intervention tests a conditional prediction effect, not a learned repair or full training explanation. Do not add head, temperature, seed or optimizer sweeps.

[Newman et al. (2020), sections 1 and 3](https://arxiv.org/html/2010.07174v1#S3) were inspected for separating content production from EOS decisions and for the limits of oracle-length evaluation. The paper motivates distinguishing these error types; it does not predict our checkpoint's failure or establish its hidden-state mechanism.
