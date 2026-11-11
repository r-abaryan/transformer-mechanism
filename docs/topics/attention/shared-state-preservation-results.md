# Correct shared states are not preserved by these writes

Executed 2026-10-04. **The first replayed write makes every supplied correct map incomplete.** Subsequent writes recover most answers. Each write improves its own value-reconstruction error while other queries can lose correctness.

[Educational guide](shared-state-preservation-guide.md) | [Frozen protocol](../../protocols/shared-state-preservation.md) | [Source](../../../experiments/shared_state_preservation.py) | [Raw report](../../../results/shared_state_preservation/report.json) | [All frames](../../../results/shared_state_preservation/metrics.csv) | [Independent audit](../../../results/shared_state_preservation/audit.json).

## Method and budget

Two reused seeds, 3461/3463, step 12000; the same 768 train-32 histories per seed. Replay one unchanged 32-event history from its saved native final state and from its supplied correct oracle state. Keep learned parameters fixed. Read every key at steps 0 through 32, including native successes and failures. Starting oracle states use all labels and were not learned or established reachable. Both arms are warm starts; the native cold prefill is a fidelity control.

CPU float64, one torch thread. Four trajectories produce 132 metric cells and **3,244,032 answer outputs**. Replay processes 98,304 writes; cold fidelity adds 49,152, totaling 147,456. **Zero training or optimizer calls.** Formal replay/save takes 3.62 seconds excluding preflight/audit; this is execution provenance, not a speed benchmark. Two seeds share the 128 context families with six cyclic renamings. Frames and queries are correlated evaluations. No new validation/test evaluation or selection.

## Accuracy and complete maps

Each seed/arm grades 24,576 queries per frame. Complete maps require every query correct from that frame's shared state.

| Seed | Start | Accuracy: step 0 % | Step 1 % | Step 32 % | Complete maps: step 0 / 1 / 32 |
|---|---|---:|---:|---:|---|
| 3461 | Native | 98.9990 | 96.3257 | 98.7590 | 561 / 227 / 515 |
| 3461 | Oracle | 100.0000 | 70.6868 | 99.6663 | 768 / 0 / 687 |
| 3463 | Native | 98.5636 | 95.7397 | 98.3561 | 510 / 211 / 490 |
| 3463 | Oracle | 100.0000 | 69.4377 | 99.4425 | 768 / 0 / 657 |

All 1536 oracle maps break immediately. All 1071 initially complete native maps also become incomplete somewhere during replay. None stays fully correct at every frame. Final oracle states lose 82/137 answers from their initially perfect maps, although they exceed their matched native replay endpoints. This is not trained accuracy or a repair of the original fitting failure.

| Seed/start | Final rescues from start | Final harms from start | Start CE | Final CE |
|---|---:|---:|---:|---:|
| 3461 native | 30 | 89 | 0.040355 | 0.037892 |
| 3461 oracle | 0 | 82 | 0.686312 | 0.022401 |
| 3463 native | 39 | 90 | 0.049351 | 0.047613 |
| 3463 oracle | 0 | 137 | 0.680806 | 0.026198 |

Lower final CE coexists with accuracy loss relative to the starting state in every arm. Intermediate scores and complete-map subgroup counts remain in the CSV/traces. Many damaged answers recover later; a temporary error is not necessarily a final task failure.

## Local correction versus other-key damage

| Seed/start | Writes lowering local reconstruction error | Written-key harms | Other-key harms across all steps |
|---|---:|---:|---:|
| 3461 native | 24,576 / 24,576 | 0 | 9,422 |
| 3461 oracle | 24,576 / 24,576 | 0 | 28,008 |
| 3463 native | 24,576 / 24,576 | 0 | 9,664 |
| 3463 oracle | 24,576 / 24,576 | 0 | 28,503 |

All 98,304 replay writes lower their own squared reconstruction error beyond the frozen 1e-9 guard. No increases or unresolved changes occur. All correct-to-wrong transitions affect keys other than the current write. The harm counts include repeated break/recovery events, not that many distinct failed queries. Both classifiers correctly label all six learned value embeddings, supporting the guide's conditional interpolation argument.

One deterministic first-harm example per seed, both at step 1/history 0 while writing key 31:

| Seed | Harmed query | Query/write dot product | Class gap before | Gap after | Written-key error before / after |
|---|---:|---:|---:|---:|---:|
| 3461 | 0 | 0.135699 | +1.000000 | -2.665689 | 110.221797 / 76.657772 |
| 3463 | 5 | 0.206356 | +2.436794 | -3.234309 | 96.717230 / 69.802970 |

The gap compares the true class against the new wrong winner. Its measured change agrees with $\beta(q^Tk)(W_y-W_c)(v-S^Tk)$ to numerical tolerance. These examples illustrate the observed mechanism, not independent replications or proof that all overlapping writes harm answers.

## Verification and conclusion

Three preflight methods pass: actual initialized saved reload/evaluator/audit, supplied ties/absent groups, local error gradients, destructive nonorthogonal write, zero-beta and orthogonal controls, zero-start agreement, and intentional target/state/logit/residual/step/group/summary errors.

Independent audit passes every frame, transition, residual contraction, explanatory example and CSV entry using matrix affine transitions and independent label/metric reductions. Maximum equation discrepancy is 8.526513e-14. All 25 scoped protected files remain unchanged. Formal replay/audit completes once with no recovery or tolerance change.

**Supported:** these selected correct states are not preserved by the learned writer. Local value correction can change other queries' class rankings. **Still open:** existence of other stable states, reachability and why training chooses these weights. The original objective supervises the final state; transient map damage alone does not establish a training defect. Backpropagation still passes through the whole sequence, but no formal training-gradient measurement ran here.

Sources inspected: [Yang et al. section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2), delta equations; [Schlag et al. (2021)](https://proceedings.mlr.press/v139/schlag21a.html), fast-weight context. Our score-change/error-contraction derivations and all numbers are local, not published score predictions or full DeltaNet reproduction.

Stop this replay. Next distinguishing question: do error-correcting gradients weaken or cancel across positions in the actual cold-start training computation? Freeze that diagnostic separately; no extra replay pass, oracle variant or training objective change. RL stays parked.
