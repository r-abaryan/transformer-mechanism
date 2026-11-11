"""Bounded active-document cleanup, with exact pre-edit snapshots preserved."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

DOCUMENTS = {
'AGENTS.md': '''# Research rules

Scope: this workspace. User instructions take precedence. Keep explanations short and educational. Current status belongs in the [checkpoint](docs/assessments/research-checkpoint.md); earlier summaries belong in [history](docs/assessments/research-history.md). Read the [failure register](docs/assessments/failure-register.md) before proposing a repeat.

## Before training

- Name one distinct question tied to observed behavior. A closed budget does not settle a topic, but a coverage gap alone does not justify a new study.
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
''',
'README.md': '''# Understanding Transformer mechanisms

We study Transformer mathematics and what small trained models actually compute. Tests, controlled interventions and independent audits support bounded claims. They do not establish an overall architecture winner or a novel replacement.

Read the [current research checkpoint](docs/assessments/research-checkpoint.md) for status and remaining questions. The [project quality review](docs/assessments/project-quality-review.md) records what was checked and its limits. Earlier milestones are in [research history](docs/assessments/research-history.md).

| Entry | Purpose |
|---|---|
| [Documentation by topic](docs/topics/README.md) | Attention, FFNs, normalization, positions, embeddings and benchmarks |
| [Educational guides](docs/guides/README.md) | Short explanations, equations and examples |
| [Study reports](docs/results/README.md) | Methods, measured findings and limitations |
| [Assessments](docs/assessments/README.md) | Conclusions across studies |
| [Sources and claims](docs/reference/README.md) | Primary papers and claim boundaries |
| [Protocols](docs/protocols/README.md) and [decisions](docs/decisions/README.md) | Frozen designs and rationale |
| [Research workflow](docs/research-plan.md) | How questions become tests and studies |
| [Rules](AGENTS.md) and [failure register](docs/assessments/failure-register.md) | Prevent repeat mistakes |

Run the reusable copy correctness checks from the workspace root:

```powershell
.venv-gpu/Scripts/python.exe tests/run_tests.py
```

The launcher uses fresh fixtures and checks that frozen files remain unchanged. See [test instructions](tests/README.md) for scope and artifact safety. It performs no training. Archived standalone scripts can write fixed historical paths, so inspect their protocol and output handling before rerunning anything.

`experiments/` contains executed implementations and independent audits. `results/` contains raw measurements/checkpoints; `docs/` contains explanations and reports. Dependencies are recorded in [requirements-cpu.txt](requirements-cpu.txt), [requirements-gpu.txt](requirements-gpu.txt) and [requirements-data.txt](requirements-data.txt). GPU work uses native Windows on an RTX 4060 laptop with 8 GB VRAM.

The project began on 2026-09-29. Frozen evidence remains at its original paths; document cleanup does not change the measurements.
''',
'docs/research-plan.md': '''# Research workflow

Purpose: understand and challenge Transformer mechanisms at toy scale, using equations, trained behavior and controlled interventions. This scale can expose counterexamples and mechanisms; it cannot establish deployment superiority.

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
''',
'docs/assessments/research-checkpoint.md': '''# Current research checkpoint

Status on 2026-10-04: the copy branch is closed. The authorized maintenance pass separates current guidance from history and provides safe reusable test runs. No new training study or architecture proposal is selected.

## What we know

| Branch | Supported result | Remaining limit |
|---|---|---|
| [Copy generalization](copy-generalization-assessment.md) | Marker-trained RoPE copies all tested short groups. Supplying the matching source route recovers all evaluated longer sequences | Native transfer remains weak; supplied alignment is task-informed and also changes mixture sharpness |
| [State tracking](../topics/benchmarks/state-momentum-training-results.md) | Recipe development and reset-cue controls change performance; the declared all-group fit target remains unmet | No isolated optimizer cause, reliable tracking solution or architectural impossibility |
| [Architecture comparisons](capability-summary.md) | Rankings depend on the task and tested recipe | Incomplete fitting and different model/supervision settings prevent an overall winner |
| [Component findings](architecture-assessment.md) | Equations and interventions clarify routing, residual/FFN contributions, positions, normalization and caches | Initialized arithmetic and task-specific interventions do not establish a universal trained mechanism |
| [Memory alternatives](memory-update-assessment.md) and [sparse attention](sparse-attention-assessment.md) | Update rules and information reach have precise tested boundaries | Mathematical counterexamples do not by themselves explain trained failures or measure speed |
| [MoE training](../topics/ffn/moe-balance-training-results.md) | Balancing reduces overflow in each matched pair | Groupwise CE preservation fails; the synthetic classifier does not establish expert specialization |

## Copy result to remember

Across three seeds, marker-trained RoPE achieves 100% short-copy accuracy. Native length-8/10 ordinary accuracy averages 10.42%/2.34%; oracle-content accuracy averages 14.06%/4.17%. Forced matching-source alignment gives 100% copying in all evaluated seed/group cells; the shifted control gives 0% full accuracy. No retraining occurred in this diagnostic. See [alignment methods and results](../topics/benchmarks/copy-marker-alignment-results.md).

This establishes that the saved models can decode the content under the supplied route. It does not establish why training failed to learn that route, or provide a usable learned repair. The earlier native failures remain valid.

## Direction and stopping boundary

The [direction review](research-direction-review.md) shifted work from coverage to explanatory questions. State recipe checks and copy interventions followed that direction. The [quality review](project-quality-review.md) verifies recent evidence within a declared scope, rather than certifying every archived script.

Future work should select one unresolved question with a distinguishing control and finite budget. Architecture adoption requires adequate training fit and fresh confirmation. More scripts, batches or audited predictions alone do not establish novelty. Neither the closed copy branch nor failed state criteria automatically authorize more tuning.

Use the [safe tests](../../tests/README.md) for current correctness checks. Maintenance evidence is in [verification](../../results/maintenance_20261004/verification.json); earlier status wording remains in [research history](research-history.md) and exact [snapshots](../../results/maintenance_20261004/before.json).
''',
}


def main():
    before = json.loads((HERE / 'before.json').read_text())
    hashes = {row['path']: row['sha256'] for row in before['snapshots']}
    for name in DOCUMENTS:
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != hashes[name]:
            raise RuntimeError(f'Active file changed since backup: {name}')
    history = ROOT / 'docs/assessments/research-history.md'
    old = (HERE / 'snapshots/docs/assessments/research-checkpoint.md').read_text(encoding='utf-8')
    with history.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write('# Historical research checkpoint\n\nArchived on 2026-10-04 before guidance cleanup. The text below preserves the earlier checkpoint. Its “latest”, “next” and pending-study wording describes past stages, not the current queue. Read the [current checkpoint](research-checkpoint.md) for today\'s status. Numerical evidence remains in linked reports and raw records.\n\nExact old plan, memory and navigation files are preserved as byte snapshots listed in [before.json](../../results/maintenance_20261004/before.json). Their original relative links are not rebased.\n\n---\n\n')
        handle.write(old)
    for name, content in DOCUMENTS.items():
        (ROOT / name).write_text(content, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
