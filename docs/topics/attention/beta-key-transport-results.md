# Separating beta from learned key geometry

Executed and independently audited, 2026-10-05. **Lower beta accounts for most of the measured increase in mean target-write coefficient retention. Later key geometry slightly reduces that mean, but improves accuracy at lower beta. Their accuracy effects interact. None of the tested endpoint swaps is a net repair.**

## Methods

[Frozen protocol](../../protocols/beta-key-transport.md). Same original seeds 3461/3463, steps 3000/12000, all 768 train-32 histories and every unique key. Use all four observed beta/key combinations under each of two fixed value/head anchors: 16 cells. No sweep, held-out selection or training. Change only key.weight and write_logit in private copies; queries and write keys stay tied. Full native diagonals reproduce earlier audited traces.

Reuse [stage replay](../../../experiments/write_interference_stages.py) and its [independent matrix/write-basis equations](../../../experiments/audit_write_interference_stages.py). CPU float64, one thread, equation tolerance 1e-9 with matching winners. Endpoint-wrong and both endpoint-correct control cohorts remain fixed; every group and all 32 positions, including empty groups, are reported. A common rotation of tied keys cannot explain a change in reads. [Educational explanation](beta-key-transport-guide.md).

## Target transport

Mean final/immediate target coefficient ratio across 24,576 queries. Identical under both anchors because this ratio depends on keys, beta and history order, not values/head. It is not a percentage of retained information.

| Seed | Key stage | Beta stage | Target coefficient ratio |
|---|---:|---:|---:|
| 3461 | 3000 | 3000 | 0.837839 |
| 3461 | 3000 | 12000 | 0.915698 |
| 3461 | 12000 | 3000 | 0.836197 |
| 3461 | 12000 | 12000 | 0.914689 |
| 3463 | 3000 | 3000 | 0.850005 |
| 3463 | 3000 | 12000 | 0.923544 |
| 3463 | 12000 | 3000 | 0.848622 |
| 3463 | 12000 | 12000 | 0.922708 |

At fixed early keys, lower beta raises the ratio by 0.077860 / 0.073539. At fixed late keys, it raises it by 0.078492 / 0.074086. The later-key effect is small and negative at either beta. This identifies the dominant factor for this mean sensitivity, not for all performance or historical learning.

## Answers and losses under fixed values/head

Every cell has 24,576 final answers. Anchor denotes the stage supplying values and classifier. Swapped cells are inference counterfactuals, not trained models. Maps require all 32 answers correct.

| Seed | Anchor | Keys | Beta | Accuracy | CE | Complete maps / 768 |
|---|---:|---:|---:|---:|---:|---:|
| 3461 | 3000 | 3000 | 3000 | 94.4499% | 0.192271 | 149 |
| 3461 | 3000 | 3000 | 12000 | 95.2026% | 0.289429 | 163 |
| 3461 | 3000 | 12000 | 3000 | 93.4652% | 0.186222 | 95 |
| 3461 | 3000 | 12000 | 12000 | 98.6369% | 0.291132 | 508 |
| 3461 | 12000 | 3000 | 3000 | 94.2871% | 0.483268 | 153 |
| 3461 | 12000 | 3000 | 12000 | 95.1986% | 0.228626 | 177 |
| 3461 | 12000 | 12000 | 3000 | 93.3919% | 0.274368 | 98 |
| 3461 | 12000 | 12000 | 12000 | 98.9990% | 0.040355 | 561 |
| 3463 | 3000 | 3000 | 3000 | 94.9259% | 0.175781 | 181 |
| 3463 | 3000 | 3000 | 12000 | 96.0653% | 0.268410 | 222 |
| 3463 | 3000 | 12000 | 3000 | 93.3024% | 0.180510 | 77 |
| 3463 | 3000 | 12000 | 12000 | 97.9411% | 0.280243 | 426 |
| 3463 | 12000 | 3000 | 3000 | 94.8283% | 0.431952 | 169 |
| 3463 | 12000 | 3000 | 12000 | 95.9310% | 0.192402 | 232 |
| 3463 | 12000 | 12000 | 3000 | 93.4082% | 0.274321 | 67 |
| 3463 | 12000 | 12000 | 12000 | 98.5636% | 0.049351 | 510 |

Later keys harm accuracy at earlier beta but help at later beta in every seed/anchor. Accuracy interactions are +4.4189 / +3.4993 percentage points under anchor 3000 and +4.6956 / +4.0527 under anchor 12000. The effects are not additive. Lower beta raises accuracy in all eight fixed-key comparisons, yet raises CE under the earlier anchor in all four comparisons. Better retention or accuracy alone therefore cannot establish better loss.

## Keep gains and losses together

Answer transitions versus native step 12000, with endpoint values/head fixed. Original errors are 246/353. All correct-query controls are included; the raw report separates correct queries in complete and incomplete histories.

| Seed | Keys | Beta | Original errors rescued | Correct answers harmed | Net correct change |
|---|---:|---:|---:|---:|---:|
| 3461 | 3000 | 3000 | 163 | 1,321 | -1,158 |
| 3461 | 3000 | 12000 | 162 | 1,096 | -934 |
| 3461 | 12000 | 3000 | 63 | 1,441 | -1,378 |
| 3461 | 12000 | 12000 | 0 | 0 | 0 |
| 3463 | 3000 | 3000 | 229 | 1,147 | -918 |
| 3463 | 3000 | 12000 | 241 | 888 | -647 |
| 3463 | 12000 | 3000 | 103 | 1,370 | -1,267 |
| 3463 | 12000 | 12000 | 0 | 0 | 0 |

Every alternative loses accuracy, CE and complete maps relative to the native endpoint. A selected error cohort can improve while overall reliability worsens. No donor combination should be adopted from this diagnostic.

## Verification and limits

[Preflight](../../../results/beta_key_transport_preflight.json): two methods pass, exercising initialized saved factorial evaluator/audit, identical-donor null effects, exact intervention scope, common-key rotation, ties/absent groups and deliberate source/label/logit/coefficient/count/contrast defects. [Independent audit](../../../results/beta_key_transport/audit.json): 16 cells, 2,112 summaries and 528 contrasts, 393,216 query records and 12,976,128 frame answers; maximum equation discrepancy 7.994e-14. All 21 protected files unchanged. Producer runtime 23.11 seconds; zero gradients, optimizer/solver calls or training. Repeated frames and anchors are not independent seeds.

[Raw report](../../../results/beta_key_transport/report.json), [producer](../../../experiments/beta_key_transport.py), [auditor](../../../experiments/audit_beta_key_transport.py), [tests](../../../tests/test_beta_key_transport.py). Scope is two opened development trajectories and training load 32. Cohorts use known endpoint outcomes. This establishes fixed-computation intervention effects, not a learning cause, a new recipe, unseen-input robustness or all-load classifier feasibility. The original exact fit gate remains unmet.

Close this factorial comparison. The [remaining questions](../../assessments/learning-mechanism-agenda.md#remaining-questions-and-stopping-boundary) prioritize a narrowly justified learning intervention rather than more swaps or stages. RL stays parked.

Source inspected: [Yang et al. (2024), sections 2.2/3.2](https://arxiv.org/html/2406.06484v2) for delta recurrence and unrolling. The factorial/rotation reasoning is derived locally; scores are measured here and not paper predictions. Our scalar-beta tied-key reader differs from the full paper.
