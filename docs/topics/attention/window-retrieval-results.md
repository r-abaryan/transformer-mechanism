# Full versus local retrieval: learned reach matches the boundary

Executed and closed2026-10-02. Three matched pairs complete the fixed budget. Every seed retrieves perfectly at eligible distances. The local unreachable controls remain exactly at one sixth.

[Educational guide](window-retrieval-guide.md) | [Frozen protocol](../../protocols/window-retrieval.md)

## Findings

Values below hold for **every seed**, in training, validation and both test lengths.

| Value-to-query distance | Full attention accuracy | Local attention accuracy | Meaning |
|---|---:|---:|---|
|1 |100% |100% |Direct access is learned |
|3 |100% |100% |Local information must cross layers |
|4 |100% |100% |Local boundary distance is learned |
|5 |100% |16.67% |Outside local two-layer reach |
|8 |100% |16.67% |Outside local two-layer reach |

Each training distance has768 examples; each validation/test distance has192. Families contain all six answers with matched distractors. The exact unreachable count is128/768 for training and32/192 elsewhere. These are correlated examples from128 or32 families; three paired seeds are the independent training replications.

## Methods and fair-comparison boundary

Both arms use4544 parameters: two width16 pre-norm blocks, MQA with two query heads/one K/V head, FFN32 ReLU and RoPE. Initial weights and sampled data streams match within each pair. The sole forward change is the attention visibility mask. The local radius is2, including self plus two earlier positions. All19 output classes remain available.

Training uses length9, Adam.001,6000 updates, effective batch60, CPU float64 and one thread. Each run sees360,000 sampled examples/final supervised targets and3,240,000 input tokens. No accumulation, schedule, clipping or dropout. Full training pools and held-out families are evaluated. Length13 adds earlier distractors while preserving the same distances. This tests robustness to added context, not increased relay range.

Both arms receive all five distance groups, including cases impossible for the local model. Their objectives and exposure match, but attainable losses differ. Dense masked attention is used for both, so these results do not measure sparse execution speed. GPU memory was not used or measured. The complete-step profile allocates Adam state before the task budget is frozen; its tensor storage is not process peak memory.

## Per-seed results and losses

Accuracy entries are eligible / unreachable percentages. Eligible means distances1/3/4; the second value covers5/8. The table above and raw report retain every distance separately.

| Seed | Arm | Train9 | Validation9 | Test9 | Test13 |
|---|---|---:|---:|---:|---:|
| 2441 | full | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 |
| 2441 | local | 100.00 / 16.67 | 100.00 / 16.67 | 100.00 / 16.67 | 100.00 / 16.67 |
| 2447 | local | 100.00 / 16.67 | 100.00 / 16.67 | 100.00 / 16.67 | 100.00 / 16.67 |
| 2447 | full | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 |
| 2459 | full | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 |
| 2459 | local | 100.00 / 16.67 | 100.00 / 16.67 | 100.00 / 16.67 | 100.00 / 16.67 |

| Seed | Arm | Train9 loss | Validation9 loss | Test9 loss | Test13 loss |
|---|---|---:|---:|---:|---:|
| 2441 | full | 0.000052 | 0.000052 | 0.000052 | 0.000053 |
| 2441 | local | 0.716767 | 0.716771 | 0.716772 | 0.716769 |
| 2447 | local | 0.716778 | 0.716838 | 0.716798 | 0.716787 |
| 2447 | full | 0.000060 | 0.000060 | 0.000060 | 0.000062 |
| 2459 | full | 0.000058 | 0.000059 | 0.000058 | 0.000059 |
| 2459 | local | 0.716788 | 0.716791 | 0.716800 | 0.716793 |

Local aggregate accuracy is66.67%, although all reachable answers are correct. Its aggregate loss approaches the structural lower bound `(2/5) ln(6) = 0.716704`:40% of examples have six indistinguishable answers. The [guide](window-retrieval-guide.md) derives this bound. This low aggregate accuracy is expected from the task design and must not be described as failure to learn reachable retrieval.

Full training, validation, short-test and longer-test criteria pass for both arms on their declared eligible distances. Local controls pass separately. These criteria do not establish local/full equivalence across all tasks or all distances.

## Learning curves and baselines

Full-pool validation losses at the scheduled checkpoints:

| Seed | Arm |2000 updates |4000 |6000 |
|---|---|---:|---:|---:|
| 2441 | full | 0.000998 | 0.000187 | 0.000052 |
| 2441 | local | 0.717930 | 0.716916 | 0.716771 |
| 2447 | local | 0.718202 | 0.717048 | 0.716838 |
| 2447 | full | 0.001172 | 0.000217 | 0.000060 |
| 2459 | full | 0.001118 | 0.000209 | 0.000059 |
| 2459 | local | 0.717875 | 0.716988 | 0.716791 |

All final validation intervals improve. Neither convergence nor an eventual ranking is inferred. The final checkpoint was selected by fixed update count.

The marked-value oracle scores100% in every group. Constant12 scores1/6; expected uniform choice among the six answers is1/6. Returning the final context symbol scores100% only at distance1 and0% elsewhere. Thus the distance3/4 success cannot be explained by that immediate-neighbor baseline. No specific learned internal code or unique relay path has been identified.

## Independent checks and recorded corrections

Before training, gathered-key attention, independent normalization/rotation/FFN equations and all parameter gradients agree within1e-10. Omitted-window controls change unreachable logits by at least0.1032, exceeding1e-6. Independent label scans check all6720 inputs across224 families; shifted labels are detected. Exact inputs and distractor families do not overlap across splits; vocabulary and subsequences do.

After training, all24 full pools and40,320 predictions pass the independent audit. Maximum logit error2.15e-14, metric error4.44e-16, zero argmax changes. Every local six-answer family at distances5/8 has equal logits within1e-9. Checkpoint/optimizer steps, paired initial values and streams, pretest snapshot and decisions are verified. All237 recorded prior JSON artifacts and175 frozen source/protocol files retain their hashes.

Two pretraining implementation issues are preserved:

- Profile serialization passed a string to a path-only hash helper. Corrected the argument and reran profiling; no task training had begun. [Record](../../../results/window_retrieval_profile_issue.json).
- Exact original-forward comparison found differences up to1.11e-15 from broadcast versus repeated K/V layout. Restored the original layout; the exact check and numerical thresholds stayed unchanged. Original data were reused byte-for-byte. The profile predates this layout correction; training uses the corrected source. [Record](../../../results/window_retrieval_preflight_issue.json).

All six task-training attempts then completed without replacement or extension. Training plus scheduled evaluation/checkpoint saving took40.31 to41.17 seconds per run. This is an execution record, not a controlled performance comparison.

## What this establishes

These local models learn usable transfer through multiple attention layers on this simple marked-value task. Information beyond their structural reach remains inaccessible to the final query under the declared operations. Full attention succeeds across all tested distances.

The bound follows from the computation graph and matched-answer construction. The learned success is empirical and limited to these seeds, data and budget. General key/value retrieval, language quality, other depths/windows, internal relay mechanisms and efficient GPU kernels remain open. No copying failure is explained by this result.

Study closed. No new tuning, extra layers, window sweep or automatic follow-up selected.

[Raw report](../../../results/window_retrieval.json) | [Independent audit](../../../results/window_retrieval_audit.json) | [Data](../../../results/window_retrieval_data.json) | [Profile](../../../results/window_retrieval_profile.json) | [Training code](../../../experiments/window_retrieval.py) | [Audit code](../../../experiments/audit_window_retrieval.py)
