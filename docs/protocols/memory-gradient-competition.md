# Memory-gradient competition: frozen diagnostic

Question: at saved exhaustive-feedback checkpoints, do correct-answer training gradients oppose repair of remaining load-32 errors on keys, values or the write-strength logit?

The [clean-input diagnostic](../topics/attention/target-only-history-results.md) establishes decoding sufficiency, while original mixed-history errors remain. This question concerns their local training incentive, not another clean-input intervention or historical causality.

| Item | Frozen choice |
| --- | --- |
| Models | Existing exhaustive-feedback delta reader; seeds 3461/3463, steps 3000/12000 |
| Primary data | Full training pools: 768 histories at each of loads 2/8/16/32; every present key queried |
| Other splits | Existing validation/test scores remain contextual references; no new held-out gradients, training, model selection or accuracy criterion |
| Primary counts | 2 seeds × 2 endpoints × 4 loads = 16 cells; 178,176 answer targets; four pooled objectives |
| Parameters | Raw key embeddings, value embeddings, alpha=write_logit, joint memory; classifier fixed; CPU float64, one thread |
| Objective | Equal mean weight per load, then uniform histories/queries within each load |
| Partitions | Correct/error membership fixed by original argmax, first class wins ties; recompute neither mask during gradients nor numerical probes |
| Contributions | Sum correct contributions across loads; separate load-32 errors and other-load errors; include absent partitions as zeros, conditional CE null |
| Measures | Weighted gradient norms, correct/error cosine, derivatives of conditional load-32 error CE along correct-only and total memory descent, per block and jointly |
| Sign resolution | Cosines classified opposed/aligned only beyond ±1e-6 with both norms >1e-10; otherwise unresolved |
| Numerical probes | Unit correct-only/total descent, each block, central differences h=1e-5/5e-6 on private copies; original load-32 error mask fixed |
| Verification | Saved initialized evaluator/audit, independent labels/matrix gradients, supplied logits/ties/absent partitions, deliberate label/gradient/sign/count/weighting errors |
| Tolerances | Logits 1e-9; gradients 1e-9+1e-7 absolute reference; finite differences 1e-7+1e-4 absolute reference |
| Stop | Exactly these 16 cells/four objectives and bounded numerical checks; no extra snapshots, optimizer step, tuning, gradient surgery or new training |

With N_m answers at load m, each weighted partition gradient is the sum of its CE gradients divided by 4N_m. Error repair is assessed against the **conditional mean** CE of remaining load-32 errors. Gradients recombine to the original equal-load objective. An error-empty group has zero contribution, not a fabricated conditional gradient.

Finite-difference directions with norm ≤1e-10 are skipped and reported. If all directions are resolved, the checks use at most 3,145,728 additional answer evaluations, not additional independent data or training exposure. Changing parameter coordinates changes block norms/dot products. A negative cosine alone does not show net error harm or Adam behavior; no curvature or learning repair is measured.

Primary sources inspected: [Yu et al. (2020), section 2.2](https://arxiv.org/html/2001.06782v3#S2.SS2) for negative-cosine conflict and its insufficiency alone; [PyTorch 2.11 CrossEntropyLoss](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html) for CE/reduction. Correct/error partitions within one task are our local construction, not the paper's multitask experiment. No published local-score prediction is claimed.
