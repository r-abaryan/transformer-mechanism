# Documentation

Start with the [research checkpoint](assessments/research-checkpoint.md) for current status. The [architecture assessment](assessments/architecture-assessment.md) explains component findings and limits.

## Browse by subject

| Section | Includes |
|---|---|
| [Reinforcement learning](topics/rl/README.md) | Foundations, applied agents and language-model policy training. |
| [Feed-forward networks](topics/ffn/README.md) | Activations, gates, width and branch contributions. |
| [Attention](topics/attention/README.md) | Q/K/V, heads, routing capacity, kernels and K/V caching. |
| [Normalization and residuals](topics/normalization/README.md) | Gradient flow, placement, LayerNorm and RMSNorm. |
| [Word embeddings and output prediction](topics/embeddings/README.md) | Input vectors, vocabulary scores and shared input/output weights. |
| [Positional representations](topics/positions/README.md) | Offsets, layouts, sinusoidal encodings and RoPE. |
| [Retrieval mechanisms](topics/retrieval/README.md) | Binding, counterfactuals, interventions and retrieval failures. |
| [Training and transfer](topics/training/README.md) | Optimization, depth, exposure and held-out generalization. |
| [Benchmark comparisons](topics/benchmarks/README.md) | State tracking, retrieval, copying, language, majority and composition. |
| [Transformer foundations](topics/foundations/README.md) | Architecture, interpretability and numerical verification. |

## Other indexes

[Guides](guides/README.md) · [Study reports](results/README.md) · [Primary sources and claims](reference/README.md) · [Protocols](protocols/README.md) · [Assessments](assessments/README.md) · [Decisions](decisions/README.md) · [Figures](figures/)

[Research workflow](research-plan.md) defines methods and stopping rules. [Research history](assessments/research-history.md) preserves earlier status; [full reading order](reading-order.md) preserves the earlier sequence. Use the [safe launcher](../tests/README.md) for reusable correctness checks.

Existing protocols retain their bytes and fixed paths for provenance checks. A few guides/reports linked by those protocols also retain their original paths; the subject indexes include them. Future documents belong in their subject folders; future frozen protocols belong in `protocols/`.

Subject folders contain Markdown explanations and reports. Raw measurements and checkpoints remain in the workspace `results/` directory.
