# Objective beta pressure: local learning incentives conflict

Completed 2026-10-04. At identical saved weights, assigned-query and all-query objectives exert opposite pooled beta pressure in all four checkpoints. This establishes a local objective tension. Whether it shaped the historical beta divergence remains unresolved.

## Method

Four original 615-parameter delta readers, seeds 3461/3463, fixed/resampled endpoints at update 3000. Original training pools only: 768 histories per load 2/8/16/32. Each history has 32 events. Assigned loss grades its fixed query; all-query loss averages over every present key. Pool losses and gradient vectors equally over the four loads, matching original sampling. No validation/test evaluation, new targets, optimizer updates or training.

CPU float64, one thread, batch 64 histories. Actual evaluation took 5.55 seconds, excluding preflight and audit; this is not a speed benchmark. CPU peak memory was not measured. The [protocol](../../protocols/objective-beta-pressure.md) was frozen before evaluation; the [guide](objective-beta-pressure-guide.md) explains signs, objectives and the state tangent.

## Pooled gradient pressure

A positive alpha derivative means lower-beta pressure under plain gradient descent; negative means higher. These are local derivatives, not recorded Adam steps. Beta is sigmoid(alpha).

| Seed | Checkpoint | Beta | Assigned alpha gradient | All-query alpha gradient | Whole-vector cosine |
|---|---|---:|---:|---:|---:|
| 3461 | fixed | 0.94757 | -0.00007892 | +0.06232282 | -0.70377 |
| 3461 | resampled | 0.32236 | -0.01495288 | +0.00465813 | +0.04718 |
| 3463 | fixed | 0.96204 | -0.00006491 | +0.04696271 | -0.74191 |
| 3463 | resampled | 0.31870 | -0.01261739 | +0.00185357 | +0.17556 |

Fixed assigned-query mean CE is 0.000675/0.000512, while its all-query mean CE is 0.592662/0.620189. Perfect assigned accuracy therefore coexists with a nonzero gradient and poor unassigned answers. Fixed-model whole-vector gradients also conflict: negative cosines imply a sufficiently small plain-gradient step on assigned loss locally raises all-query loss. This last statement follows from the first-order dot-product formula; no such update was executed.

Resampled whole-vector cosines are positive despite opposite beta directions. Their remaining training errors are not explained by beta having a zero derivative at these endpoints. None of these measurements proves convergence, historical causation or a finite-step repair.

## All load-specific directions

"Unresolved" means absolute alpha derivative <=1e-7 under the frozen numerical rule. It does not establish mathematical zero or task irrelevance. Derivatives shown to eight decimal places can round very small values to zero; full precision is retained in raw records.

| Seed | Checkpoint | Load | Assigned alpha gradient | All-query alpha gradient | Pressure: assigned / all |
|---|---|---:|---:|---:|---|
| 3461 | fixed | 2 | -0.00000000 | -0.00000000 | unresolved / unresolved |
| 3461 | fixed | 8 | -0.00000658 | +0.00011074 | higher / lower |
| 3461 | fixed | 16 | +0.00002321 | +0.03875983 | lower / lower |
| 3461 | fixed | 32 | -0.00033230 | +0.21042069 | higher / lower |
| 3461 | resampled | 2 | -0.00000001 | -0.00000001 | unresolved / unresolved |
| 3461 | resampled | 8 | -0.00163038 | -0.00118446 | higher / higher |
| 3461 | resampled | 16 | -0.02559925 | -0.01412370 | higher / higher |
| 3461 | resampled | 32 | -0.03258188 | +0.03394068 | higher / lower |
| 3463 | fixed | 2 | -0.00000000 | -0.00000000 | unresolved / unresolved |
| 3463 | fixed | 8 | -0.00000981 | +0.00032953 | higher / lower |
| 3463 | fixed | 16 | -0.00001918 | +0.02889961 | higher / lower |
| 3463 | fixed | 32 | -0.00023065 | +0.15862170 | higher / lower |
| 3463 | resampled | 2 | -0.00000001 | -0.00000001 | unresolved / unresolved |
| 3463 | resampled | 8 | -0.00082535 | -0.00082298 | higher / higher |
| 3463 | resampled | 16 | -0.01254718 | -0.01801636 | higher / higher |
| 3463 | resampled | 32 | -0.03709703 | +0.02625360 | higher / lower |

There are 7 opposite-sign and 5 resolved same-sign comparisons; four load-2 comparisons are unresolved. Fixed seed 3461 load 16 favors lower beta under both objectives. Resampled loads 8/16 favor higher beta under both. Keep these exceptions: the pooled conflict is not a rule at every load.

Loads 2/8/16 contain repeat-query groups only; load 32 contains unique queries only. Missing groups are explicitly null. The [96-row metrics CSV](../../../results/objective_beta_pressure/metrics.csv) retains every cell/group CE, accuracy, counts and gradients. The [report](../../../results/objective_beta_pressure/report.json) additionally retains all block norms and pairwise gradient cosines.

## Verification

Four preflight methods pass, covering eight actual initialized checkpoint evaluator/audit cells, supplied logits, missing groups, independent labels, batch changes, tangent checks and deliberate label/gradient/metric errors. The independent matrix auditor checks all 32 formal cells, 190464 primary target scores and all 615 gradient coordinates per cell. An analytic beta tangent and two fixed finite-difference spacings agree with the producer. Finite-difference recomputations do not add independent data or training seeds.

Maximum logit error: 1.421e-14. Parameter-gradient error: 1.805e-16. Alpha finite-difference error: 1.696e-10. No argmax disagreements. Parent CE/accuracy matches. All 30 locked files remain unchanged, including original checkpoints and executed sources. [Preflight](../../../results/objective_beta_pressure_preflight.json), [audit](../../../results/objective_beta_pressure/audit.json), [lock](../../../results/objective_beta_pressure/lock.json). The audit shares weights/data/PyTorch and reuses summary policy functions; its forward equations, CE reductions and beta tangent are separate implementations.

## What this explains and what comes next

The measured objectives locally prefer different write strengths, even at the same representation and decoder. This gives a concrete candidate explanation for why assigned-query optimization can reward behavior that conflicts with broader retention. The earlier inference swaps show a separate effect on predictions; they do not supply the missing training history.

Unknown: when this tension develops, how sampled gradients and Adam moments steer the actual updates, and which remaining errors involve exposure, representation or optimization. Next proposed question: does tension precede the beta split during an instrumented reproduction of the original initialization, streams and budget? Freeze that replay and require endpoint agreement before interpreting its history. It has not run. No continuation, beta grid or architecture change follows automatically.

Sources inspected: [Yang et al. 2024, section 2.2](https://arxiv.org/html/2406.06484v2) for correction/read and input-dependent beta; [PyTorch 2.11 Adam equations](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html) for moment-dependent updates. Our scalar/tied-key reader differs from the full DeltaNet model. Objective expectations, chain rule, tangent and gradient alignment are local derivations. Papers do not predict the measured signs or scores. Two seeds and original opened training pools limit scope; no fresh confirmation or universal architecture conclusion is claimed. RL remains parked.
