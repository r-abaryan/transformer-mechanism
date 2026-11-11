# Learned causal global relay

Frozen2026-10-02 after six complete CPU profiling steps and before task training/evaluation. Distinct question: can a middle global position learn a distant relay, and does removing its extra connections remove that behavior?

## Task and split

Sequences length17 for training/validation/test; length21 test only. Final query18, exactly one marked answer12..17. Distractors0..10; existing middle position n//2 is fixed marker11 in all three arms. Distances1/4/5/12/16 from marked value to query. Radius2 with two layers reaches1/4 locally. Middle-global paths additionally reach12/16. Distance5 is after the global and outside local reach at both lengths. The marker never contains the answer itself.

Generator2026100205,128 training families,32 each validation17/test17/test21. Every family expands to all distances and all six values. Whole base families and exact inputs must be disjoint across pools. Counts3840/960/960/960; familiar symbols/subsequences can overlap. Full pools are evaluated. Six-value variants share every input except the marked value; if no path exists, deterministic accuracy is at most1/6. Baselines: oracle, constant12, last context symbol, expected uniform six-answer choice.

## Models and budget

Three arms full/local/global. Reuse two pre-norm width16 MQA blocks, two Q heads/one K/V head, FFN32 ReLU, RoPE offset4,4544 parameters. Shared supplied projections for local/global connections. Full causal mask; local radius2; global adds middle query/key connections, then intersects the entire mask with causality. Marker and weights are identical across arms initially. Only connectivity changes. No pooling or cross-token normalization.

Seeds2521/2531/2539 with complete copied initial states. Rotate execution order across seeds. CPU float64, one thread, deterministic algorithms. [Profile](../../results/global_retrieval_profile.json) includes optimizer allocation at batch60 and lengths17/21. GPU unused; GPU allocated/reserved peaks are not applicable. Dense masking is retained; no sparse-speed claim.

Adam.001, betas.9/.999, eps1e-8, no weight decay/clipping/schedule/dropout.6000 updates, batch/effective batch60, no accumulation. Two sampled families per distance, each expanded to all six labels; replacement sampling seed+1000 and identical streams across arms.360,000 examples/final supervised targets,6,120,000 input tokens per run. All arms receive all groups, including their structurally unreachable ones. All19 classes stay in the softmax.

Minibatch loss every1000; full train/validation accuracy and loss at2000/4000/6000. Final checkpoint only, no selection. Finish and snapshot all nine attempts before any test scoring. Nonfinite attempts stop and are preserved, no replacement. Save scores/checkpoints. After training evaluate global models again with both extra global query and key connections removed in both layers, weights fixed, on all four pools. This ablation can alter other predictions too.

## Gates and audit

- Per-seed training gate>=99% at every eligible distance: full all five, local1/4, global1/4/12/16. Validation>=95% at every eligible distance. Short and longer test>=95% separately, conditional on both gates. Report every distance/seed regardless of gate.
- Global dependence: report native-minus-ablated accuracy per seed/distance12/16 in both tests. Require native>=95% and ablated<=1/6 at each, plus global training/validation gates, before calling distant retrieval dependent on these extra connections. No unique route or representation claim.
- All structurally unreachable groups must have six-answer-family logits invariant within1e-9 and accuracy<=1/6. Includes ablated global models at5/12/16.
- Independently scan labels/splits; shifted-label control must fail. Before training compare full logits and all parameter gradients against independent gathered-key/norm/rotation equations within1e-10. Full arm must equal the preserved full forward exactly. Check future-token perturbations and global distance5 invariance; omitted-window control effect>1e-6.
- After training independently recompute all native and ablated pools: logit/loss errors<=1e-9, zero argmax changes, metrics and decisions from counts. Verify complete optimizer steps, matching initial states/data streams and unchanged historical evidence. Record failures without weakened thresholds.

Longer inputs move the middle marker and add context while preserving value-to-query distance; both marked sources at12/16 remain before the marker. No claim of arbitrary-distance transfer. Stop after the fixed comparison, ablation and audit, regardless of result. No tuning or follow-up selected.
