# Same-history writing interference during learning

Frozen before evaluation, 2026-10-05. Question: how do the queried write's retention and other-write score contributions change at saved stages, and do endpoint errors differ from successful controls?

Use seeds 3461/3463 at steps 0/3000/12000 from the completed [stage comparison](../topics/attention/readout-conflict-stages-results.md). Use every original train-32 history in its stored order: 768 histories, 32 unique keys and 32 writes each. Query every key. No validation/test evaluation, training, optimizer, gradient, solver, parameter intervention or extra stage.

Start from zero memory. Reuse the existing preservation replay for 33 native frames and reconstruct final effective write coefficients by reverse transport. Independently audit with forward matrix recurrence and transported write-basis vectors. Verify labels, uniqueness, frame winners, original final logits, coefficient decomposition and unchanged model/RNG. Exercise actual initialized saved-checkpoint evaluator and audit before the formal run; include supplied ties, empty cohorts and deliberate target/logit/coefficient/summary defects.

Freeze cohorts by the original step-12000 native answers: endpoint wrong, endpoint correct in incomplete histories, endpoint correct in complete histories, and all. These same identities apply at every stage. They are descriptive, selected using known training outcomes, not fresh confirmation. Report every seed/stage/cohort plus each target-write position, including empty groups. Positions are one-based.

For each query record immediate post-target-write and final correctness, true-versus-best-other margin, any later correct-to-wrong transition, and any later wrong-to-correct transition. Fix its competitor to the strongest non-target class at the final frame of that stage, first-index ties. Decompose the final target-minus-competitor margin into target write, earlier writes, later writes and bias. Compare target contribution at its own write with its transported final contribution. Report target coefficient divided by its immediate coefficient (beta times squared key norm); this is signed sensitivity, not percent information retained. Also report beta and native final CE/complete maps.

Threshold for signed descriptive counts: 1e-9; values within it are unresolved. Equation checks use absolute/relative 1e-9 and require identical discrete winners. No adaptive thresholds or subgroup selection. Counts derive independently from history/query/write dimensions. Repeated positions and cyclic value renamings are not independent seeds.

Stop after six checkpoint diagnostics and one independent audit. Jointly changing keys, values, beta and head do not isolate a training cause. Other writes can help by carrying the same value. Prefix correctness is diagnostic; the original objective supervises final reads. A harmful transition does not establish absent supervision or an optimization defect. Preserve original 246/353 errors and all prior evidence.

Source inspected: [Yang et al. (2024), sections 2.2 and 3.1/3.2](https://arxiv.org/html/2406.06484v2). Published delta and unrolling equations motivate the local decomposition. Our transposed, tied-key scalar-beta toy reader differs from the paper. Neither paper equations nor this protocol predict local scores.
