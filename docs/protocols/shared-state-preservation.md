# Frozen preservation test for supplied correct states

Question: does the learned delta write rule preserve the already correct shared states saved by the feasibility diagnostic when the same stable-value history is replayed once?

Use both saved exhaustive seeds 3461/3463 at step 12000 and all 768 original train-32 histories per seed. Keep order, labels, keys, values, beta and classifier fixed. Two starts per history: saved native final memory and the saved label-informed oracle memory. The latter is not learned or established reachable. This is distinct from earlier repetition beginning at zero.

Replay exactly one 32-event pass from each start. Read all 32 queries at steps 0 through 32. Native Reader zero-start prefill reconstructs the native starting state as a fidelity control; no-write invariance and orthogonal preservation are preflight controls. No second pass, state search, solver call, new parameter setting, held-out evaluation or training follows automatically.

## Equations and measurements

$$S'=S+\beta k(v-S^Tk)^T,\qquad
\Delta z(q)=\beta(q^Tk)W(v-S^Tk).$$

Record all logits, initial/final memories, accuracy, CE, complete maps, first loss of full-map correctness, every-step correct-to-wrong and wrong-to-correct counts. Separate the just-written key from other keys. Report native initially complete/incomplete groups too. Exact argmax ties use the native first-index rule; margins are measured rather than rounded to a fit gate.

For each write, record $E=\frac12\|v-S^Tk\|^2$ before/after and check $E'=(1-\beta\|k\|^2)^2E$. Classify changes with absolute 1e-9 guard: decrease, increase, unresolved. This local squared-error quantity is not the outer training CE objective. Also check the six value embeddings' direct classifier outputs, since correct endpoint reads interpolate with the value when querying the written key. Do not assume this control passes.

Select at most one explanatory oracle harm per seed: earliest step, then smallest history index, then smallest query key. Report key overlap and the signed true-versus-new-winner gap change. This is an example selection rule, not a new causal population or chosen training criterion.

## Execution and audit

CPU float64, torch one thread, same environment as the source study. Formal counts: two seeds times two arms times 33 frames = 132 metric cells; times 768 histories times 32 queries = 3,244,032 answer outputs. Replay writes: two times two times 768 times 32 = 98,304. Cold fidelity prefill adds 49,152 writes, totaling 147,456 processed writes. These repeated answers/events are not independent experiments. Zero optimizer calls or training updates. No validation/test scores are newly produced; this is a training-pool mechanism diagnostic.

Freeze source/protocol/preflight hashes plus the source witnesses, bound report/audit, checkpoints and data before replay. Reuse the actual Reader's reads; its prefill cannot accept an initial state, so a separate state-initialized evaluator uses the same native write expression. Verify zero-start equivalence before formal execution. Refuse existing output paths and preserve all source evidence.

Preflight must exercise the actual saved reload/evaluator/audit entry, tied scores and absent groups. Include a known destructive nonorthogonal write, zero-beta and orthogonal controls, a local squared-error gradient check and deliberate wrong state/label/logit/residual/step/group/summary defects. Independent audit reconstructs labels, matrix affine transitions, scalar/classifier equations, transitions, example gap identities and all CSV cells. Use atol/rtol 1e-9 for continuous quantities, exact discrete counts and coverage. Preserve numerical disagreements; no post-result tolerance change.

Sources inspected: [Yang et al. (2024), section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2), correction/read equations; [Schlag et al. (2021)](https://proceedings.mlr.press/v139/schlag21a.html), fast-weight delta-rule context. Our transposed-state score-change and squared-error relations are local derivations. The compact scalar-beta reader is not a full-paper reproduction, and no source predicts local scores. A damaged supplied state does not prove absence of other stable states or explain why training chose its parameters.
