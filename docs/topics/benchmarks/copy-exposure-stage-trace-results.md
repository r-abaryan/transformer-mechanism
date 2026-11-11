# Where the exposure branches differ

**Recovered examples show a larger branch difference after the FFN residual and final normalization than at the cross-attention residual.** This is a descriptive projection, not proof that the FFN caused recovery. Correct terminal decisions remain intact on supplied correct prefixes.

## What was measured

We traced nine saved models: the shared update-6,000 start and both continuation branches at 6,500/7,000/7,500/8,000. No training or activation replacement. The same test6/test10 sources and correct decoder prefixes enter every model. Positions after an earlier generated stop are therefore counterfactual teacher-forced measurements, not successful generation.

Final generated outcomes define fixed long cohorts: **44 recovered, 29 persistent failures, 55 preserved successes, zero harmed**. Earlier checkpoints use those same retrospective cohorts. All content positions and true terminal positions are recorded. For failed control examples, we also inspect its first generated premature-stop position. These examples were already opened; they are not confirmation data.

At each stage we project the hidden vector onto the model's final classifier contrast:

`margin(h) = (w_EOS - w_content)^T h`.

Positive means an EOS preference along this direction. Only the final stage gives the actual model logit margin. Earlier layers were not trained to predict directly through this classifier. Each branch has its own learned classifier, so changes in its weights also affect the comparison.

At the true terminal position, the contrast is EOS versus the strongest final non-EOS competitor, fixed across stages within that model/example. This avoids the meaningless EOS-minus-EOS contrast.

## Final-checkpoint findings

Mean margins at the control's premature-stop positions for the 44 recovered examples:

| Stage | Continued cross-only | Added self exposure | Paired difference |
|---|---:|---:|---:|
| Pre-cross residual | -1.660 | -2.134 | -0.475 |
| Cross output alone | -4.493 | -5.239 | -0.746 |
| Cross residual sum | -6.153 | -7.373 | -1.221 |
| Cross LayerNorm | -3.702 | -4.357 | -0.655 |
| FFN residual sum | +1.927 | -1.705 | -3.632 |
| Final LayerNorm | +3.382 | -3.685 | -7.067 |

The cross output alone is a component projection, not a sequential residual stage. For the sequential stages, signed margin increments telescope exactly to the final margin; this bookkeeping does not isolate causal contributions.

All 44 control answers are wrong at these positions; all 44 intervention answers are correct. Their mean cross-output norms are 4.672 and 4.816. Norm differences alone do not indicate retrieval quality.

| Final-checkpoint check | Control | Added exposure |
|---|---:|---:|
| Persistent cohort: correct at control stopping position | 0/29 | 1/29 |
| Persistent cohort: mean margin there | +4.723 | +1.576 |
| All long examples: correct true terminal decision, supplied prefix | 128/128 | 128/128 |
| Short test6 content decisions, supplied prefix | 768/768 | 768/768 |
| Short test6 terminal decisions, supplied prefix | 128/128 | 128/128 |

The persistent cohort still fails full generation, including the one example improved at this selected position. Recovery cannot be reduced to passing a single token check. At true long terminal positions, cohort mean EOS margins remain strongly positive in both models, approximately +9.48 to +9.84. The improvement is not accompanied by loss of correct teacher-forced terminal decisions.

## What this identifies, and what it leaves open

The difference is already present before and inside cross-attention, then becomes larger downstream in this readout projection. The classifier control below confirms that this pattern survives a common readout. It still does not separate changed inputs from changed nonlinear processing. Negative intermediate margins are not evidence that an intermediate layer has already solved the task.

**Do not jump to an FFN swap.** Previous cross-model swaps damaged compatibility. A matched training test of FFN adaptation is now a candidate, subject to its own protocol and short-fit controls. The present diagnostic does not establish FFN causality or authorize further training automatically.

## Classifier control, added 2026-10-08

Using the existing saved tensors, we scored both branches through each checkpoint's control classifier and, separately, its intervention classifier. The non-EOS competitor is also held identical between branches. At terminal positions it is the control's strongest final non-EOS token; changing the intervention's rival is recorded separately.

At update 8,000, recovered examples retain downstream growth under **both** fixed classifiers. Values are paired intervention-minus-control margin differences:

| Stage | Fixed control classifier | Fixed intervention classifier |
|---|---:|---:|
| Cross LayerNorm | -0.638 | -0.630 |
| FFN residual sum | -3.602 | -3.566 |
| Final LayerNorm | -6.970 | -6.937 |

Magnitude increases at both transitions in all 44 recovered examples at this endpoint. The cohort means also increase under both classifiers at every earlier sampled continuation stage, although individual earlier examples have exceptions. These are retrospective cohorts, not independently selected confirmations.

For a common competitor, write `a = w_EOS - w_competitor` and `m = a^T h`. The exact symmetric identity is

`delta_m = mean(a)^T delta_h + delta_a^T mean(h)`.

It splits the bilinear interaction evenly into representation and classifier terms. Both ordered decompositions were also checked. This is coordinate-dependent bookkeeping, not causal percentages.

| Final margin difference | Representation term | Classifier term | Rival change | Total |
|---|---:|---:|---:|---:|
| Recovered, control stop position | -6.953 | -0.114 | 0 | -7.067 |
| Persistent, control stop position | -3.029 | -0.118 | 0 | -3.147 |
| Recovered, true terminal | +0.359 | -0.174 | -0.004 | +0.181 |

The large recovered-content difference cannot be explained solely by changing the scoring classifier. Terminal contrasts remain a separate measurement. These results do not establish whether FFN weight adaptation, upstream representation changes or normalization produced the useful states.

[Control method](../../protocols/copy-exposure-readout-control.md) · [all checkpoints, cohorts and terms](../../../results/copy_exposure_stage_trace/readout_control/report.json) · [code](../../../experiments/copy_exposure_readout_control.py). Eight paired model/pool cells, 48 stage cells and 9,216 paired positions pass persisted scalar audits; maximum discrepancy is 7.105e-15. Two reusable tests cover saved initialized traces, single-factor/interaction cases, unequal rivals and deliberate faults. Preflight corrected reversed expected terms in a hand-calculated fixture before saved-model scoring. All 155 protected files remain unchanged; zero training. The [previous educational report](../../../results/copy_exposure_stage_trace/readout_control/educational-report-before.md) preserves the original interpretation.

## Evidence and sources

[Protocol](../../protocols/copy-exposure-stage-trace.md) · [all stages/cohorts and checkpoints](../../../results/copy_exposure_stage_trace/report.json) · [code](../../../experiments/copy_exposure_stage_trace.py).

Initialized saved-model checks compare explicit attention, normalization and FFN equations with hooked native activations. Scalar classifier projections and telescoping identities pass; an intentionally corrupted stage is rejected and absent cohorts are explicit. Persisted tensors are reloaded and rechecked. All 18 model/pool cells, 20,736 positions and 149 protected files pass; maximum stage discrepancy is 8.371e-14. The reference shares architecture helpers; it is not a fully separate implementation. No independent seeds were added.

[Vaswani et al. (2017), §§3.1–3.3](https://arxiv.org/html/1706.03762v7#S3) supplies the attention, residual, normalization and FFN structure inspected in the earlier studies. Intermediate classifier projections, retrospective cohorts and measured scores are local diagnostics, not published predictions. The actual full-generation improvement remains documented in the [continuation report](copy-exposure-continuation-results.md).

[Section 3.4](https://arxiv.org/html/1706.03762v7#S3.SS4), inspected for the extension, describes the linear output transformation. The symmetric decomposition above is local algebra; the paper does not predict these measured terms.
