# Delta-memory results: replacement with a geometric condition

Executed2026-10-03 and closed. No training. [Educational guide](delta-memory-guide.md) | [Frozen protocol](../../protocols/delta-memory.md).

**The delta rule can replace a unit-key value exactly while preserving orthogonal reads. Overlapping keys can still interfere.** This is arithmetic evidence, not a trained retrieval result.

## Method and checks

Three seeds, lengths3/7/17, key/value widths3/2 and5/4:18 CPUfloat64 cases, batch2, heads2, one thread. Supplied random queries/values, L2-normalized raw keys and sigmoid write strengths. Gradients include key normalization.

Compare correction recurrence, an independent expansion of ordered matrix transitions, and two chunks carrying state. Initial state zero; read after current write. No positional encoding, masks, positive feature map, normalization accumulator, decay gate or learned projections.

| Check | Maximum discrepancy |
|---|---:|
| Outputs | 2.67e-15 |
| Final state | 7.78e-16 |
| Q/raw-K/V/write-logit gradients | 5.56e-17 |
| Joint directional finite difference | 3.73e-11 |
| Earlier outputs after future perturbation, against reference | 2.67e-15 |
| General interference equation on random states | 8.89e-16 |

All pass frozen1e-10 arithmetic and1e-7 finite-difference thresholds. Zero raw keys are rejected because they have no unit direction.

## Error controls

| Deliberately wrong operation | Detected difference |
|---|---:|
| Omit subtraction of old read | 2.45392 |
| Reverse correction sign | 24.45856 |
| Reverse matrix-transition order | 0.32928 |
| Detach carried state | Old-value gradient changes by0.45304; outputs unchanged |
| Read before current write | 4.0768 |

Transitions generally do not commute. Matching only final outputs also misses the broken gradient caused by detachment.

## What the witnesses show

| Question | Result | Assumption or limit |
|---|---|---|
| Replace old value2 with8? | beta0/.5/1 gives2/5/8; old-value derivatives1/.5/0 | Unit key; exact beta1 is a boundary fixture |
| Preserve another association? | A2,B5,A8 ends with A8,B5 | Orthogonal unit keys, beta1 |
| Preserve an overlapping key? | Updating A2 to8 changes B5 to8.6 when overlap is0.6 | Prior memory supplied to satisfy both old reads; confirms interference, not inevitable error for every representation |
| Ignore key normalization? | Scalar key2, target2 produces reads8,-16,56; unit key gives2,2,2 | Explicit overshoot example, not a statement about every unnormalized design |
| How does state sensitivity change? | At beta1, unit-key transition eigenvalues0/1; norm2 gives-3/1 | Local homogeneous-state transition only |

The interference formula is delta_read=beta*(q^T k)*(new_value-old_read). The frozen random-state checks and exact scalar witnesses agree with it.

## Source comparison and stopping point

[Schlag et al. (2021)](https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf), section4.2, equations20-25, supplies the error-correcting memory update. Its feature/sum normalization is not our L2 convention. [Yang et al., Gated Delta Networks v3 (2025)](https://arxiv.org/html/2412.06464v3), section2.2, gives the equivalent plain-delta matrix form; its full model adds mechanisms outside this audit.

Our results support the declared equations and expose the conditions behind selective updating. They do not show learned key orthogonality, improved language quality, efficient GPU execution or a cause of earlier retrieval failures. The change from the previous normalized positive-kernel memory includes the readout and key representation, so these witnesses are not a controlled trained architecture comparison.

This question is closed. No training or automatic extension selected.

Evidence: [executed source](../../../experiments/delta_memory.py), [raw cases and controls](../../../results/delta_memory.json).
