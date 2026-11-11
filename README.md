# Understanding Transformer mechanisms

We study Transformer mathematics and what small trained models actually compute. Tests, controlled interventions and independent audits support bounded claims. They do not establish an overall architecture winner or a novel replacement.

Read the [current research checkpoint](docs/assessments/research-checkpoint.md) for status and remaining questions. The [project quality review](docs/assessments/project-quality-review.md) records what was checked and its limits. Earlier milestones are in [research history](docs/assessments/research-history.md).

The focus is [explaining learned failures](docs/assessments/learning-mechanism-agenda.md). The [copy assessment](docs/assessments/copy-generalization-assessment.md) connects inference-time diagnostics to a learned positional-exposure repair and its limits. The [retention synthesis](docs/assessments/retention-learning-causal-account.md) preserves that branch's established effects and unresolved causes. No experiment is currently queued; the [RL roadmap](docs/topics/rl/rl-roadmap.md) remains parked.

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
