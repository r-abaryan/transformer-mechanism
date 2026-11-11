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

The difference is already present before and inside cross-attention, then becomes larger downstream in this readout projection. This does not separate better retrieved information from changed nonlinear processing: upstream states, FFN, normalization and readout all changed during training. Negative intermediate margins are not evidence that an intermediate layer has already solved the task.

**Do not jump to an FFN swap.** Previous cross-model swaps damaged compatibility. This trace supplies a candidate region for explanation, not a valid isolated intervention. Stop this diagnostic; no additional training, component sweep or causal claim follows automatically.

## Evidence and sources

[Protocol](../../protocols/copy-exposure-stage-trace.md) · [all stages/cohorts and checkpoints](../../../results/copy_exposure_stage_trace/report.json) · [code](../../../experiments/copy_exposure_stage_trace.py).

Initialized saved-model checks compare explicit attention, normalization and FFN equations with hooked native activations. Scalar classifier projections and telescoping identities pass; an intentionally corrupted stage is rejected and absent cohorts are explicit. Persisted tensors are reloaded and rechecked. All 18 model/pool cells, 20,736 positions and 149 protected files pass; maximum stage discrepancy is 8.371e-14. The reference shares architecture helpers; it is not a fully separate implementation. No independent seeds were added.

[Vaswani et al. (2017), §§3.1–3.3](https://arxiv.org/html/1706.03762v7#S3) supplies the attention, residual, normalization and FFN structure inspected in the earlier studies. Intermediate classifier projections, retrospective cohorts and measured scores are local diagnostics, not published predictions. The actual full-generation improvement remains documented in the [continuation report](copy-exposure-continuation-results.md).
