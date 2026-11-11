# Gated delta-memory results

Executed2026-10-03, closed after a recorded fixture correction. No training. [Guide](gated-delta-memory-guide.md) | [Protocol](../../protocols/gated-delta-memory.md) | [Correction record](../../protocols/gated-delta-memory-correction.md).

**Decay and delta correction compose correctly. Preserving other associations still depends on the gate and key geometry.**

## Numerical evidence

18 cases: three seeds, lengths3/7/17, widths3/2 and5/4, batch2, heads2, CPUfloat64, one thread. Independent ordered transition expansion agrees with recurrence and two chunks. Gradients include raw-key normalization and both gate logits.

| Check | Maximum discrepancy |
|---|---:|
| Outputs | 8.89e-16 |
| Final state | 3.34e-16 |
| Q/raw-K/V/alpha-logit/beta-logit gradients | 4.17e-17 |
| Directional finite difference | 2.48e-11 |
| Future perturbation against reference | 8.89e-16 |
| alpha1 versus archived plain delta | 0 |
| beta0 from zero state | 0 |
| alpha0 versus current write only | 8.89e-16 |
| General read-change equation | 1.34e-15 |

All pass unchanged1e-10 arithmetic and1e-7 finite-difference limits. First alpha-logit gradients are zero because initial memory is zero.

| Intentional error | Detected difference |
|---|---:|
| Subtract undecayed old read | 0.25 |
| Decay the correction write too | 2.0625 |
| Omit decay | 0.25 |
| Detach carried state | Old-value gradient changes by0.125; outputs unchanged |
| Read before writing | 3.25 |

## Preservation, replacement and reset

The [guide's five boundary cases](gated-delta-memory-guide.md) verify final A/B reads and gradients to both old associations. alpha1,beta1 replaces A2 with8 and preserves orthogonal B5. alpha0,beta1 writes A8 while all old-state gradients vanish. Partial decay weakens orthogonal B too.

For an overlapping query(.6,.8), initial state(2,4.75) gives read5. Updating key(1,0) to8 with alpha.5,beta1 gives state(8,2.375), hence read6.7. The original protocol incorrectly expected6.8. All18 random cases passed before that assertion failed. Original source/report/protocol are preserved; the separate corrected source changes only that expected value and artifact/provenance paths. Identical cases and tolerances then passed. This was a fixture arithmetic defect, not a failed recurrence or a threshold adjustment.

## Source comparison and limits

[Gated Delta Networks v3, section3.1 equation10](https://arxiv.org/html/2412.06464v3) specifies the combined transition. Our tests support its local arithmetic in the declared convention. No learned gates, paper benchmarks, optimized chunkwise kernels, low-precision behavior or GPU speed were tested. Preserved earlier retrieval failures remain unexplained by this audit.

The four-rule mathematical branch is now [consolidated and closed](../../assessments/memory-update-assessment.md). No training or automatic extension selected.

Evidence: [verified source](../../../experiments/gated_delta_memory_verified.py), [passing report](../../../results/gated_delta_memory_verified.json), [original source](../../../experiments/gated_delta_memory.py), [failed fixture attempt](../../../results/gated_delta_memory.json).
