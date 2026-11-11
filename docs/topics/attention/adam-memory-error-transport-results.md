# Selected Adam memory steps help most errors, with two late reversals

Executed 2026-10-04. **Fourteen of 16 recorded memory movements lower CE on their before-model errors; two late movements from seed 3463 raise it.** Finite memory-only and full-model loss changes match the memory projection signs in all 16 cases.

[Educational guide](adam-memory-error-transport-guide.md) | [Frozen protocol](../../protocols/adam-memory-error-transport.md) | [Executed source](../../../experiments/adam_memory_error_transport.py) | [Raw report](../../../results/adam_memory_error_transport/report.json) | [All block projections](../../../results/adam_memory_error_transport/metrics.csv) | [Independent audit](../../../results/adam_memory_error_transport/audit.json).

## Method

Read the [already audited replay](readout-adam-transport-results.md): seeds 3461/3463, steps 3001–3004 and 11001–11004. The late cycle starts at 11000, not the earlier gradient diagnostic's 12000 endpoint. Earlier replay established checkpoint/optimizer/RNG fidelity and exact archived beta logs; these individual memory movements were not directly logged historically.

For each before-model, evaluate all 768 training load-32 histories and every key, 24,576 answers. Freeze its error mask. Compute conditional error-CE gradients on all parameters, then project recorded movements per block. Compare a hypothetical -lr × recorded minibatch gradient without applying it. Split actual Adam movement into stored/current numerator terms with the observed updated denominator.

Run actual forwards for before, after-memory with before-classifier, and complete recorded after-model. Report accuracy/CE, rescues/harms and whole-history correctness for all conditions. All queries here are unique; repeat groups are empty/null. No held-out evaluation or full-objective gradient is included.

CPU float64, one thread. Formal execution: zero optimizer calls or training updates; 393,216 baseline targets, 1,179,648 condition outputs and 1,572,864 same-pool numerical-probe outputs. Two seeds are the independent training runs; selected steps and repeated answers are not fresh confirmation.

## Every selected step

Positive projection/change raises mean CE on that step's before-error subset; negative lowers it. Projection is first order, while the two changes are measured finite effects. These subsets vary between steps.

| Seed | Step | Batch load | Before errors | Memory projection | Memory-only CE change | Full-model CE change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3461 | 3001 | 2 | 1,364 | -0.000798581 | -0.000794508 | -0.000639653 |
| 3461 | 3002 | 8 | 1,362 | -0.000697459 | -0.000694132 | -0.000553672 |
| 3461 | 3003 | 16 | 1,364 | -0.000294706 | -0.000292353 | -0.000154844 |
| 3461 | 3004 | 32 | 1,364 | -0.000515295 | -0.000509807 | -0.000327048 |
| 3461 | 11001 | 2 | 291 | -0.004331119 | -0.004161887 | -0.003964136 |
| 3461 | 11002 | 8 | 290 | -0.002896643 | -0.002763338 | -0.002564519 |
| 3461 | 11003 | 16 | 289 | -0.001231780 | -0.001124074 | -0.000944128 |
| 3461 | 11004 | 32 | 287 | -0.007594658 | -0.007422460 | -0.007208034 |
| 3463 | 3001 | 2 | 1,247 | -0.000983075 | -0.000980456 | -0.000784416 |
| 3463 | 3002 | 8 | 1,245 | -0.000867556 | -0.000865426 | -0.000686315 |
| 3463 | 3003 | 16 | 1,246 | -0.000476917 | -0.000474989 | -0.000292537 |
| 3463 | 3004 | 32 | 1,245 | -0.000663505 | -0.000658335 | -0.000431644 |
| 3463 | 11001 | 2 | 373 | -0.002019053 | -0.001839858 | -0.001720693 |
| 3463 | 11002 | 8 | 378 | -0.000823534 | -0.000679067 | -0.000572841 |
| 3463 | 11003 | 16 | 377 | +0.001753972 | +0.001868321 | +0.001958967 |
| 3463 | 11004 | 32 | 378 | +0.000617117 | +0.000728060 | +0.000914942 |

All block projections clear the frozen ±1e-10 sign-resolution rule. Tiny load-2 hypothetical SGD/current terms remain unresolved; they are not interpreted as zero gradients.

| Parameter movement | Lower error CE, first order | Higher error CE, first order |
| --- | ---: | ---: |
| Keys | 14 | 2 |
| Values | 0 | 16 |
| Alpha | 11 | 5 |
| Joint memory | 14 | 2 |
| Classifier | 0 | 16 |
| Full model | 14 | 2 |

Block summaries reuse the same coordinates and are not independent trials. The classifier offsets part of helpful memory movement in 14 cases. This is a first-order statement; no classifier-only finite intervention was run.

## The late reversal

Seed 3463, step 11004:

| Joint-memory direction | Conditional error-CE projection |
| --- | ---: |
| Hypothetical raw minibatch SGD | -0.000023828 |
| Adam current numerator term | -0.001489638 |
| Adam stored numerator term | +0.002106754 |
| Recorded Adam movement | +0.000617117 |

Thus the stored term outweighs the helpful current term in this step. The hypothetical SGD and current term are not identical controls: Adam scales coordinates. Splitting numerators is not a momentum-reset counterfactual. At step 11003, hypothetical raw SGD also favors higher error CE, so both late reversals cannot be described as momentum alone opposing helpful raw gradients.

| Measured condition | Rescued answer evaluations | Harmed answer evaluations | Accuracy rises / unchanged / falls across 16 steps |
| --- | ---: | ---: | --- |
| After memory, before classifier | 49 | 50 | 6 / 5 / 5 |
| Recorded full after-model | 57 | 48 | 9 / 2 / 5 |

These totals count repeated examples across steps. They are not numbers of distinct repaired histories. CE improvement on the frozen error subset does not guarantee accuracy improvement or preserve originally correct answers. Full per-condition accuracy, CE and map counts are in the JSON report.

## Verification and limits

Three preflight methods cover actual saved-state evaluator/audit, supplied tied logits/absent errors, independent gradients, fixed masks and corrupt labels, masks, gradients, Adam movements, numerator terms and probe signs. A unique initialized synthetic fixture reconstructs four optimizer steps solely for preflight; the formal study makes zero optimizer calls.

The formal audit independently enumerates labels, evaluates all three states with matrix recurrences, checks every conditional error-gradient coordinate, verifies scalar Adam for all coordinates, and reconstructs projections. Actual memory directions pass central differences at two spacings and both signs, with exact private parameter restoration.

Maximum errors: logits 4.307665e-14, gradients 1.332268e-15, scalar Adam 1.776357e-15, scalar projections 6.938894e-18, numerical slopes 9.880025e-11. All frozen tolerances pass; 188 protected files remain unchanged.

Primary sources inspected: [Kingma and Ba, Algorithm 1](https://arxiv.org/pdf/1412.6980) and [PyTorch 2.11 Adam](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html), moment/bias-correction/update equations. Projections, conditional subsets and local scores are our work, not published predictions.

**Established:** selected Adam updates can favor error repair despite correct/error gradient opposition; stored moments explain one observed numerator-direction reversal. **Unknown:** prevalence over the full training history, held-out effects, whether reset/tuning helps, and why joint learning leaves unreliable maps. No full-history optimizer-failure claim or learned repair follows.

Stop this diagnostic chain. A causal test of stored memory moments would require a separately frozen, paired training intervention and training-only outcome rules. The observed reversal justifies considering it, not assuming it succeeds or automatically extending the budget. RL remains parked.
