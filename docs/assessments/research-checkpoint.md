# Current research checkpoint

Latest control: [fixed-classifier extension of the exposure trace](../topics/benchmarks/copy-exposure-stage-trace-results.md#classifier-control-added-2026-10-08). Recovered-cohort downstream growth survives both fixed endpoint classifiers. At the final control stop positions, the -7.067 margin difference splits into -6.953 representation and -0.114 classifier terms under the symmetric convention. The classifier confound is addressed for this measurement; changed upstream inputs, FFN and normalization remain coupled. Eight paired cells and 9,216 positions pass persisted scalar checks, 155 protected files unchanged, zero training. A matched test of FFN adaptation is a candidate, not a frozen or queued run.

Completed, 2026-10-08: [shared-checkpoint exposure continuation](../topics/benchmarks/copy-exposure-continuation-results.md). From the identical weak update-6,000 model/Adam state, two matched 2,000-update branches retain full EOS supervision. Added self-site exposure reaches 99/128 length-10 complete copies versus cross-only continuation's 55/128: 44 rescues, zero harms. Length-8 counts are 128/128 versus 92/128. All short train/validation/test groups stay perfect at every measured checkpoint.

The effect is substantial but the 95% complete-repair gate fails: 29 intervention premature stops remain. Both branches improve over the original length-10 failure, so continuation alone also contributes. Exposure has a conditional effect from this saved state; its internal mediator, convergence and seed reliability remain unresolved. Failure does not prove exposure must begin early.

Verification: 4,000 updates, 638,000 examples, 3.36M targets; paired streams/generators, native starting scores, initialized evaluator/gradient checks and deliberate errors pass. Persisted audit covers 126 cells and 328 group rows within 1.434e-13; 129 protected files unchanged. Stop at update 8,000; no budget extension, component sweep or tuning is queued. Retention, state tracking and RL remain deferred. See [copy synthesis](copy-generalization-assessment.md) and [history](research-history.md).

## Evidence by branch

| Branch | Established | Still unresolved |
|---|---|---|
| [Copying](copy-generalization-assessment.md) | From-initialization exposure repair; matched continuation improves a weak solution | Internal learning mechanism and reliable complete recovery from that saved failure |
| [Compact retention](retention-learning-causal-account.md) | Controlled fitting and representation studies identify bounded effects and constraints | Why training selects poor-transfer representations |
| [Task comparisons](capability-summary.md) | Rankings depend on the task, fitting and recipe | Overall architecture superiority |
| [Transformer components](architecture-assessment.md) | Tested equations and task-specific intervention effects | Universal learned mechanisms or a novel replacement |
| [Sparse attention](sparse-attention-assessment.md) and [memory updates](memory-update-assessment.md) | Tested reach and update-rule limitations | General quality/resource superiority |
| [MoE](../topics/ffn/moe-balance-training-results.md) | Balancing affects routing under the tested setup | Expert specialization; quality preservation criterion failed |

## Direction and stopping boundary

The active research concerns learned copy failures and the effects of positional exposure. Retention and state findings remain available without reopening their budgets. The [direction review](research-direction-review.md) rejects coverage-driven experiments and repeated tuning against opened scores.

Before another study, name what it can distinguish beyond existing evidence and read the [failure register](failure-register.md). More scripts, batches or audited predictions do not establish novelty. No next experiment is currently frozen.

Earlier measurements belong in [research history](research-history.md) and the linked assessments, not another running list here. The [pre-cleanup snapshot](../../results/docs_tidy_20261008/before.json) preserves the previous full checkpoint. Use the [safe test launcher](../../tests/README.md) for existing copy checks.
