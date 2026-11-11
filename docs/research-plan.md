# Research workflow

Purpose: understand Transformer computation, explain learned failures, and identify genuine gaps that could justify a new method. Toy-scale experiments can expose mechanisms and counterexamples; they cannot establish deployment superiority or field-level novelty by themselves.

The goal is to identify failures and explain unresolved learning mechanisms. The [mechanism agenda](assessments/learning-mechanism-agenda.md) records the broader questions; the [checkpoint](assessments/research-checkpoint.md) identifies the active branch. The [RL roadmap](topics/rl/rl-roadmap.md) is parked. Completed experiments retain their original budgets and conclusions; new explanatory questions require bounded designs.

The [checkpoint](assessments/research-checkpoint.md) is the current status. The [history](assessments/research-history.md) records earlier stages. This document defines the workflow, not an experiment queue.

## Choose a question

Start from a measured behavior or precise claim. Read existing evidence and the [failure register](assessments/failure-register.md) first. Name what a new result would distinguish. Prefer checkpoint inspection when training is unnecessary. An uncovered component alone is insufficient reason for a study.

| Question | Evidence required | Claim boundary |
|---|---|---|
| Is a mathematical statement true? | Derivation, assumptions and a counterexample search | The stated mathematical setting |
| Is code implementing the equations? | Independent outputs/gradients, boundary fixtures, deliberate errors | Tested implementations and inputs |
| What did a trained model learn? | Full fitting diagnostics, held-out groups, controlled interventions | Saved models, data and intervention conditions |
| Does an alternative help? | Adequate fitting, declared resource comparison, fresh paired confirmation | Tested tasks, budgets and seeds |

Arithmetic checks belong in reusable tests. Empirical and explanatory questions warrant studies. Neither test counts nor closed-study counts are progress measures by themselves.

Before selecting the next study, write a short gap assessment in the existing agenda: the exact model/task and failure, what local evidence and primary literature already answer, the unresolved claim, competing explanations, and a result that would distinguish them. Include what a negative result would teach. Do not promote a standard control into the research goal. If the gap is already answered or no discriminating result is available, stop or change the question before running.

Distinguish four questions: can the model represent the solution, can its actual write process produce it, can training find it, and does it transfer? A finite training-fit witness answers none of these universally. A novel method is a conditional outcome of this process, not a required conclusion.

## Develop, freeze, confirm

1. Check labels, baselines, evaluator/audit runtime, masks, missing groups and checkpoint reload. Use the [safe launcher](../tests/README.md) for archived copy tests.
2. If fitting is uncertain, declare a small recipe-development phase. Record every candidate; judge declared training-fit and selection metrics. Validation selection is allowed when declared. Keep confirmation data/seeds reserved.
3. Freeze the selected recipe, task, split, seeds, budgets, per-group metrics, error controls, meaningful improvement margin and stopping rule. If no candidate fits, report that outcome rather than silently launching a comparison.
4. Run fresh confirmation where an adoption claim requires it. Match initial values/data streams where applicable; disclose parameter, exposure and compute differences.
5. Audit results independently. Record observations, interpretations and unknowns separately. Preserve errors, failed criteria and valid earlier evidence.
6. Stop at the declared limit. Another run needs a distinct question and explicit bounded design.

## Compare resources honestly

Report parameters, updates, examples/tokens, effective batch, precision, context, hardware and peak memory. Equal parameters, equal tokens and equal compute are different comparisons. For timings, state warmup, synchronization, kernels and repeated samples; repeats are not independent training seeds.

Profile a complete optimizer step before increasing GPU budgets. Larger batches increase exposure at fixed updates. Eight GB of VRAM is a resource constraint, not evidence that a bigger run answers a better question.

## Interpret and improve

Check training fit before calling a failure generalization. Separate possible information paths from learned use, initialized identities from trained behavior, and diagnostic recovery from an implementable repair. Current findings can support a task-specific choice without identifying its cause.

A future architecture proposal needs a supported failure mechanism, explicit equations, comparison with primary predecessors, reproducible implementation and fresh evidence on the intended tasks/resources. Retain losing cases. No novel proposal or new training study is selected by this workflow.

Exact pre-cleanup wording is retained in the [maintenance snapshots](../results/maintenance_20261004/before.json). Those byte snapshots preserve provenance; their original relative links are not a navigation index.
