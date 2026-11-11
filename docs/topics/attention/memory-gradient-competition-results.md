# Gradient opposition exists, but joint memory descent still favors error repair

Executed 2026-10-04. **Correct-answer contributions oppose load-32 error gradients in all 16 tested memory blocks. The total joint-memory gradient nevertheless favors lower error CE at all four checkpoints.** Some individual blocks favor higher error loss.

[Educational guide and diagram](memory-gradient-competition-guide.md) | [Frozen protocol](../../protocols/memory-gradient-competition.md) | [Executed source](../../../experiments/memory_gradient_competition.py) | [Full report](../../../results/memory_gradient_competition/report.json) | [All block metrics](../../../results/memory_gradient_competition/metrics.csv) | [Independent audit](../../../results/memory_gradient_competition/audit.json).

## Method

Saved exhaustive-feedback delta readers: seeds 3461/3463, steps 3000/12000. Evaluate every query in all four full training pools, 768 histories per load. Load 32 has unique keys; other loads repeat stable associations. Preserve original equal-load weighting. Compute correct/error/total gradients on raw keys, values and alpha; freeze the classifier and original prediction masks. Separate load-32 errors from errors at other loads.

CPU float64, one thread; 16 load cells and 178,176 primary answer targets. Four pooled objectives yield four blocks each: keys 512 parameters, values 48, alpha 1, joint memory 561. Only two training seeds are independent runs; endpoints, reused answers and joint/block summaries are not fresh confirmation. No optimizer call or training update.

Validation/test metrics remain in the [continuation report](query-feedback-continuation-results.md) and [loss decomposition](feedback-loss-decomposition-results.md). This diagnostic uses training gradients only; it tests neither transfer repair nor held-out selection.

## Results

Correct-contribution versus load-32-error cosine. Negative entries mean local opposition under the declared coordinates. Every entry clears the frozen sign-resolution rule.

| Seed | Step | Keys | Values | Alpha | Joint memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3461 | 3000 | -0.883134 | -0.863714 | -1.000000 | -0.981398 |
| 3461 | 12000 | -0.635098 | -0.169170 | -1.000000 | -0.708592 |
| 3463 | 3000 | -0.880953 | -0.957609 | -1.000000 | -0.980931 |
| 3463 | 12000 | -0.681313 | -0.250912 | -1.000000 | -0.758360 |

The stronger-looking scalar alpha cosine reflects opposite scalar signs. It does not establish that alpha dominates every effect or that a beta reset would help.

Joint-memory derivatives of conditional load-32 error CE under the named **raw descent direction**, per unit step multiplier. Positive raises error loss locally; negative lowers it. These are not observed training-step changes.

| Seed | Step | Load-32 errors / 24,576 | Other-load errors | Correct-only descent | Error-only descent | Total descent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3461 | 3000 | 1,364 | 4 | +0.016109000 | -0.018405430 | -0.002260269 |
| 3461 | 12000 | 246 | 0 | +0.014607637 | -0.021185101 | -0.006577464 |
| 3463 | 3000 | 1,247 | 1 | +0.017718323 | -0.021813526 | -0.004109350 |
| 3463 | 12000 | 353 | 0 | +0.011473291 | -0.017846084 | -0.006372793 |

Other-load errors occur only at load 16 in the earlier endpoint. Their gradient contributions are included in total descent and retained separately in the CSV. Later original training accuracies remain 98.9990%/98.5636%, below the exact 99% gate.

| Total descent in a single block | 3461: 3000 | 3461: 12000 | 3463: 3000 | 3463: 12000 |
| --- | --- | --- | --- | --- |
| Keys | Lower error CE | Lower error CE | Lower error CE | Lower error CE |
| Values | Higher error CE | Lower error CE | Higher error CE | Lower error CE |
| Alpha | Lower error CE | Higher error CE | Lower error CE | Lower error CE |
| Joint memory | Lower error CE | Lower error CE | Lower error CE | Lower error CE |

Those block exceptions remain visible. Across blocks, beneficial and harmful terms can sum to a beneficial joint direction. The native recipe uses Adam, so this does not establish that its actual updates follow the raw total direction.

## Verification and scope

Three preflight methods exercise the real saved-checkpoint evaluator/audit, supplied logits, ties, absent partitions, independent gradients, fixed-mask probes and deliberate count/label/gradient/sign/weighting defects. Formal audit checks all 16 cells with independently enumerated labels and matrix gradients, then scalar norms/dots and the derivative arithmetic.

All 32 unit probe directions run at both spacings and both signs: 3,145,728 additional answer evaluations on the same load-32 pools. These private perturbations are restored exactly; they are numerical checks, not new data, exposure or training.

Maximum errors: logits 8.526513e-14, gradient coordinates 8.326673e-16, scalar summaries 1.088019e-13, directional finite differences 2.773468e-10. All frozen tolerances pass; 178 protected files remain unchanged.

Primary sources inspected: [Yu et al. (2020), section 2.2](https://arxiv.org/html/2001.06782v3#S2.SS2), conflict definition and its insufficiency alone; [PyTorch 2.11 CrossEntropyLoss](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html), CE/reduction. Partition construction, directional formulas and all scores are local work, not a reproduction of multitask PCGrad results.

**Conclusion:** local opposition is real, but these checkpoints do not support a claim that total raw memory descent blocks error repair. Curvature, sampled Adam transport, later subgroup behavior and the full historical cause remain unresolved. No accuracy improvement, gradient-surgery benefit or architecture limit is established.

Stop this partition diagnostic. A distinct next check can project the already reconstructed original Adam **memory** movements onto frozen error gradients, using existing replay artifacts. That tests actual selected update directions rather than assuming Adam equals raw descent; it needs a separate protocol and no additional training budget.
