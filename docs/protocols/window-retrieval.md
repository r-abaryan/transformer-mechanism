# Full versus local marked-value retrieval

Frozen2026-10-02 after the complete-step CPU profile and before task evaluation/training.

## Question and task

Can two causal radius2 layers learn to retrieve a value that needs more than one attention step? Compare full causal attention with local causal attention; every weight and all other operations match initially.

Input length9 for training/validation/test; length13 test only. Distractors are IDs0..11, exactly one marked value is ID12..17, final query is18. Predict the marked value among all19 output classes, using only final-position cross-entropy. Distance from marked value to final query is1/3/4/5/8. Distances1,3/4,5/8 mean directly reachable, reachable through two layers, and unreachable for this local model.

Generator2026100204 supplies128 training distractor families and32 each validation9/test9/test13. Each family expands to every distance and all six values. Keep whole families in one split; independently check exact inputs do not overlap. Counts3840/960/960/960. Symbols and shorter subsequences remain shared. For unreachable distances the six answers share exactly the same visible last five tokens. Any deterministic local predictor can correctly answer at most one of six in each family. Randomness is in family generation and minibatch selection; labels remain the actual marked value, never arbitrary noise.

Baselines: marked-value oracle, constant12, last context symbol, expected uniform choice among six values. Report all distances separately; aggregate scores cannot decide success.

## Fixed models and budget

Reuse existing two-block width16 MQA model: two query heads, one K/V head, head width8, pre-LayerNorm, FFN32 ReLU, final normalization, RoPE offset4,19-class output,4544 parameters. No dropout. Only the visibility mask changes. Radius2 includes self and two earlier positions. No cross-token operation outside attention. This implementation still computes dense scores; no sparse-kernel speed claim.

Seeds2441/2447/2459, copied complete initial states for each pair. CPU float64, one thread, deterministic algorithms. GPU unused, peak GPU allocated/reserved memory not applicable. The [complete-step profile](../../results/window_retrieval_profile.json) includes Adam-state allocation at lengths9/13 and batch60. Its first serialization attempt had a path-type defect, preserved in [issue record](../../results/window_retrieval_profile_issue.json); no task training had begun.

Adam lr.001, betas.9/.999, eps1e-8, no weight decay, schedule or clipping.6000 updates per arm. Effective batch60: two sampled training families per distance, each expanded to all six labels, sampled with replacement using seed+1000. Paired streams identical. Each run360000 examples and final supervised targets,3240000 input tokens. Both arms receive unreachable examples; these gradients/objective terms are part of the declared comparison.

Full training/validation accuracy and loss at2000/4000/6000 updates; minibatch loss every1000. Final6000 checkpoint only, no selection. Alternate arm execution order across seeds. Finish and snapshot all attempts before scoring tests. Save per-pool logits and checkpoints. Nonfinite loss/gradients stop and record that attempt, without replacement.

## Criteria and checks

- Local training gate: >=99% at each distance1/3/4 in every seed. Full gate: >=99% at all five distances in every seed.
- Validation reliability: >=95% at each eligible distance in every seed. Short test reliability and length13 transfer separately require >=95% per eligible distance and both training/validation gates.
- Local unreachable control: <=1/6 accuracy and identical final logits across each six-value family within1e-9, at both lengths. A violation blocks interpretation and triggers an implementation investigation.
- Before training, independent gathered-key equations check complete outputs and all parameter gradients within1e-10. Original full-model forward must agree. Omitted-window control must change unreachable-family logits by >1e-6. Independent label scan and shifted-label negative control required.
- After training, independently recompute every pool: logits/loss within1e-9, zero argmax differences, per-distance counts and decisions. Verify state/stream/protocol hashes and unchanged prior artifacts.

A successful distance3/4 test would establish usable multistep information transfer for this task under these local-only operations, without identifying a unique internal route. A failed training gate means this recipe did not master its eligible task. Stop after this comparison and audit regardless of result. No tuning, larger model, extra budget or new seed follows automatically.
