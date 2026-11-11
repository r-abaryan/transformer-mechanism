# Same histories: writing interference across training

Executed and independently audited, 2026-10-05. **Most remaining errors lose a previously correct answer. The target write still contributes positively, while other-write contributions outweigh it. Successful controls also suffer temporary damage.** This narrows the observed computation, without identifying a training cause.

## Methods and scope

Two original exhaustive-feedback seeds, saved steps 0/3000/12000. All 768 original train-32 histories, every one of their 32 unique keys. Zero-start replay, 33 frames per history. Endpoint-defined cohorts stay fixed across stages: wrong answers, correct answers in incomplete histories, and correct answers in complete histories. No held-out evaluation or new training. [Protocol](../../protocols/write-interference-training-stages.md), [educational equations](write-interference-stages-guide.md).

Producer reuses the existing preservation replay and computes coefficients by reverse transport. Audit independently reconstructs labels, matrix recurrence and forward-transported write basis; checks every frame, coefficient, winner, cohort, position and reduction against saved original endpoints. CPU float64, one thread. Signed-count guard 1e-9; identical discrete winners required. No architecture or parameter changes.

## Learning improves the computation

Each row contains 24,576 final answers. Ratio is mean final/immediate coefficient of the queried write, not information retained.

| Seed | Training step | Beta | Target coefficient ratio | Final accuracy | Final CE | Complete histories / 768 |
|---|---:|---:|---:|---:|---:|---:|
| 3461 | 0 | 0.8808 | 0.4743 | 10.6242% | 2.082097 | 0 |
| 3461 | 3000 | 0.3221 | 0.8378 | 94.4499% | 0.192271 | 149 |
| 3461 | 12000 | 0.1660 | 0.9147 | 98.9990% | 0.040355 | 561 |
| 3463 | 0 | 0.8808 | 0.4761 | 29.0405% | 1.740846 | 0 |
| 3463 | 3000 | 0.2974 | 0.8500 | 94.9259% | 0.175781 | 181 |
| 3463 | 12000 | 0.1505 | 0.9227 | 98.5636% | 0.049351 | 510 |

Beta, key geometry, values and classifier all change. These observations do not isolate which change improves retention or accuracy. Original exact 99% fit gates remain unmet.

## Same failed queries, successful controls

Each cell is **correct immediately after its own write / correct at the final read**. Groups use step-12000 outcomes, not each stage's outcomes. Sizes are per seed, not independent replications.

| Seed | Fixed group | Queries | Step 0 | Step 3000 | Step 12000 |
|---|---|---:|---:|---:|---:|
| 3461 | Endpoint wrong | 246 | 0 / 32 | 244 / 160 | 222 / 0 |
| 3461 | Correct in incomplete histories | 6,378 | 0 / 675 | 6,365 / 6,015 | 6,315 / 6,378 |
| 3461 | Correct in complete histories | 17,952 | 0 / 1,904 | 17,914 / 17,037 | 17,888 / 17,952 |
| 3463 | Endpoint wrong | 353 | 169 / 87 | 341 / 225 | 323 / 0 |
| 3463 | Correct in incomplete histories | 7,903 | 3,951 / 2,276 | 7,891 / 7,508 | 7,885 / 7,903 |
| 3463 | Correct in complete histories | 16,320 | 8,199 / 4,774 | 16,314 / 15,596 | 16,255 / 16,320 |

At the endpoint, 234/246 and 336/353 failed queries undergo a later correct-to-wrong transition; some become correct only after another write. The 24/30 queries already wrong after their own write must remain visible. At the last position, each seed has six wrong answers and no later writes.

Among correct controls in complete histories, 872/17,952 and 794/16,320 undergo later damage yet finish correctly. Thus temporary damage is shared behavior, not an error-specific mechanism. Prefix reads are diagnostics, not the original supervised objective.

## Which content wins the final competition?

Mean final true-minus-strongest-other margin decomposition at step 12000. The competitor is fixed separately for each query at that stage. Values are logit contributions, not causal percentages; displayed rounding need not sum exactly.

| Seed | Group | Target write | Earlier writes | Later writes | Bias | Final margin |
|---|---|---:|---:|---:|---:|---:|
| 3461 | Endpoint wrong | +25.6063 | -9.1857 | -17.3922 | -0.0023 | -0.9739 |
| 3461 | Correct in incomplete histories | +26.1443 | -4.1182 | -5.5508 | -0.0001 | +16.4752 |
| 3461 | Correct in complete histories | +26.1154 | -3.8216 | -5.0338 | -0.0002 | +17.2599 |
| 3463 | Endpoint wrong | +23.9862 | -9.9255 | -15.0553 | -0.0012 | -0.9959 |
| 3463 | Correct in incomplete histories | +24.3967 | -4.1406 | -5.1372 | +0.0001 | +15.1190 |
| 3463 | Correct in complete histories | +24.3856 | -3.6040 | -4.7153 | -0.0002 | +16.0661 |

For endpoint errors, mean target contribution falls by 2.9195 / 2.3926 from its own write to the final read. Mean later contribution is -17.3922 / -15.0553. Thus the average failure is not simply disappearance of the direct target signal. These are endpoint-selected descriptive groups with different position distributions, not matched causal treatment groups.

Negative later contributions occur in 240/246 and 335/353 errors, and also 12,471/17,952 and 11,065/16,320 correct controls in complete histories. Negative competition alone is insufficient to predict failure. All 32 position strata, empty groups, means and signed counts are in [raw report](../../../results/write_interference_stages/report.json); no position was selected after evaluation.

## Verification and boundary

[Preflight](../../../results/write_interference_stages_preflight.json): two methods passed, including an initialized saved evaluator/audit, known destructive versus orthogonal writes, ties/absent cohorts and intentional label/logit/coefficient/position/cohort/summary/count errors. The known aligned-key control retains coefficient ratio 0.5 and loses its answer; the orthogonal control retains ratio 1 and preserves it.

[Independent audit](../../../results/write_interference_stages/audit.json): six stages, 792 summaries, 147,456 query records and 4,866,048 frame answers; maximum equation discrepancy 4.974e-14. All 20 scoped protected files unchanged. Producer runtime 8.21 seconds on these tiny models. Zero autograd, optimizer, solver calls and training updates. Frame answers are repeated observations from two seeds, not millions of independent experiments. [Producer](../../../experiments/write_interference_stages.py), [auditor](../../../experiments/audit_write_interference_stages.py), [tests](../../../tests/test_write_interference_stages.py).

Stop this six-stage comparison. It explains a route to failure at saved weights, not why optimization keeps it. A distinct next check can separate the effects of the observed beta change and key-geometry change on target transport, holding value/head weights fixed and retaining successful controls. No new training or parameter sweep follows automatically. RL stays parked.

Source inspected: [Yang et al. (2024), sections 2.2 and 3.1/3.2](https://arxiv.org/html/2406.06484v2) provides delta recurrence and unrolling. The score decomposition is derived locally; no published score is reproduced or used to predict these outcomes. Unlike the paper, this toy reader uses one scalar beta, tied normalized queries/keys and a separate affine classifier. Earlier [joint affine conflict](readout-conflict-stages-results.md) spans all four loads; this diagnostic uses load 32 only and does not explain its certificates.
