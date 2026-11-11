# Research rules

Scope: this workspace. User instructions take precedence. Keep explanations short and educational. Current status belongs in the [checkpoint](docs/assessments/research-checkpoint.md); earlier summaries belong in [history](docs/assessments/research-history.md). Read the [failure register](docs/assessments/failure-register.md) before proposing a repeat.

## Before training

- Name one distinct question tied to observed behavior. A closed budget does not settle a topic, but a coverage gap alone does not justify a new study.
- Cite primary sources in each new research guide/report. State what was inspected; distinguish published equations and claims, local derivations and measured results. Do not imply source-backed predictions for local scores.
- Freeze labels, splits, seeds, budget, metrics, baselines and stopping rules before evaluation. Report every seed and relevant group.
- Plan final training, validation and test accuracy/loss together. For language, use token loss/perplexity. If full training evaluation is impractical, freeze a representative subset and disclose its scope.
- Separate short held-out performance from longer-input transfer. Check shortcuts. Initial and reset values must vary independently: complementary label balancing can leave them correlated. Include disagreeing-value counterfactuals.
- Check labels independently, relevant outputs/gradients and an intentional error control. Derive Cartesian counts from declared factor lengths and check them independently. See the [count correction](docs/protocols/copy-source-end-marker-count-correction.md).
- Exercise the actual evaluator and independent audit entry on supplied logits and an initialized saved checkpoint, including missing groups and reload/import paths, before training. Compilation or helper tests alone are insufficient. See [EOS recovery](docs/topics/benchmarks/copy-eos-training-results.md#evaluation-defects-recovery-and-verification).
- Use the [safe test launcher](tests/README.md) for existing copy cases. Archived test scripts can overwrite frozen artifacts when invoked directly. Keep implementation arithmetic in reusable tests; a numerical identity alone is not a trained-model finding.

## Diagnose failures before proposing causes

| Evidence | Permitted conclusion |
|---|---|
| High training fit, low held-out performance | A generalization gap exists; its cause remains open |
| Low training and held-out performance | The task was not mastered under this setup; optimization, capacity, supervision and exposure remain candidate explanations |
| Short held-out success, longer-input failure | Length transfer fails within the tested conditions |
| Weak groups or seeds hidden by a good mean | Reliability fails even if the aggregate improves |

Accuracy alone does not establish architectural impossibility, convergence, memorization or a learned mechanism. Distinguish implementation defects, failed empirical criteria and missing measurements. Missing diagnostics limit interpretation without automatically invalidating verified runs. Check what combined fit fields mean: a validation failure need not imply incomplete training fit.

## Follow-up discipline

- Separate bounded recipe development from confirmation. Declare candidate settings and selection metrics, record every attempt, establish full training-group fit, then lock the recipe and reserve fresh confirmation. Declared validation selection is allowed; repeated confirmation-based selection is not. Preserve fixed-budget failures.
- Inspect existing checkpoints first when they can answer the question. Preserve executed sources, protocols, reports and checkpoints; add diagnostics separately. Never silently overwrite evidence.
- Match initial values, data streams and declared budgets where applicable. Historical scores are references, not paired controls. Joint changes to capacity, exposure, initialization or precision do not isolate a cause.
- Changing supervision changes the objective and target feedback. Report loss weighting, supervised positions and compute; do not label it an architecture change or assume equal learning cost.
- Separate observations, interpretations and unknowns. Numerical audits verify their tested arithmetic. Prediction counts, repeated timings and parameter counts are not independent seeds or measured performance.
- Stop at the declared rule. Failure does not automatically authorize tuning, longer training or replacement seeds. A new question needs its own bounded protocol.

## Controls and interpretation reminders

- Test both signs of padding perturbations: a large one-sided perturbation can suppress the invalid key and hide a missing mask. Source and decoder-memory masks both matter; exclude target padding from loss. See [cross-attention](docs/topics/attention/cross-attention-results.md) and [encoder-decoder](docs/topics/foundations/encoder-decoder-results.md).
- Separate teacher-forced, generated and oracle-content scores. Supplied source alignment uses task knowledge and changes mixture sharpness too; recovery is not a learned repair or proof of a single cause. See the [copy assessment](docs/assessments/copy-generalization-assessment.md).
- State, retrieval, language and graph comparisons differ in tasks and sometimes model variants/supervision. Incomplete simple-task fitting blocks composition-specific conclusions. Loss falling at budget end does not establish convergence. Matched total parameters need not match core capacity. See the [capability summary](docs/assessments/capability-summary.md).
- A mathematical collision need not explain a trained failure. Correct reset-aware answers do not prove literal history erasure. TV and overflow are identical under the tested MoE capacity policy, not two independent criteria. Synthetic residual-bypass success does not establish expert specialization. See [memory](docs/assessments/memory-update-assessment.md) and [MoE training](docs/topics/ffn/moe-balance-training-results.md).

## Hardware and scope

RTX 4060 laptop, 8 GB VRAM. Use native Windows; do not install WSL. Larger future budgets are permitted when justified, but unused VRAM does not reopen studies. Profile a complete training step including optimizer state before freezing model/context/batch/precision. Report effective batch, updates, examples/tokens and peak allocated/reserved memory. Larger batches at fixed updates increase exposure and do not automatically make comparisons fair or results representative of deployment.

The project verifies mechanisms and bounded behavior at toy scale. Broader architecture adoption or novelty needs adequate fitting, fresh confirmation, relevant tasks/resources and primary-source comparison. Keep failed criteria and losing cases visible.
