# Cross-study assessments

[Project quality and direction review](project-quality-review.md): checks the response to earlier criticism, recent code evidence, test-artifact safety and remaining documentation duplication. No new training.

[Copy generalization assessment](copy-generalization-assessment.md): eight completed studies. Marker-trained RoPE achieves 100% short copying; supplied alignment recovers 100% longer copying. Native transfer remains weak, and the intervention is not a trained repair. Earlier boundaries and failed criteria remain valid. Copy branch closed without further tuning or subdivisions.

[Research direction review](research-direction-review.md): prioritizes an explanatory state-tracking question over further topic coverage. [Checkpoint diagnostic](../topics/benchmarks/state-sensitivity-results.md).

Start with the [architecture coverage review](architecture-coverage-review.md): established findings, conditional results, unknowns and one recommended next priority.

[Memory-update assessment](memory-update-assessment.md): closes additive, gated, delta and gated-delta mathematical studies.

[Sparse-attention assessment](sparse-attention-assessment.md): closed arithmetic, reachability and learned relay findings.

These documents combine evidence and decisions across experiments. Study reports are indexed in [results](../results/README.md), explanations in [guides](../guides/README.md), and frozen plans in [protocols](../protocols/README.md). Raw evidence remains in the workspace `results/` folder.

| Document | Purpose |
|---|---|
| [Capability summary](capability-summary.md) |Six task families, four architecture comparisons, GPU fitting and a sequence-copy baseline; no overall winner. |
| [Architecture assessment](architecture-assessment.md) | Start here: component findings and boundaries, including the encoder-decoder diagram and closed copying investigation. |
| [Recipe assessment](recipe-assessment.md) | Supported choices, task-transfer boundary and the closed external comparison. |
| [Normalization assessment](normalization-assessment.md) | Closed gradient, placement/budget and LN/RMS investigations. |
| [FFN assessment](ffn-assessment.md) | Closed activation, width and gated-FFN comparisons; why the ReLU baseline stays. |
| [Positional representation](position-representation-assessment.md) | Closed sinusoidal/RoPE comparisons: improvements and reliability limits. |
| [Evidence summary](evidence-summary.md) | Detailed overview with links to individual studies. |
| [Research checkpoint](research-checkpoint.md) | Closed branches and stopping decisions. |
| [Failure register](failure-register.md) | Failed claims and conditions for revisiting them. |
| [Mechanism assessment](mechanism-assessment.md) | Historical synthesis of the earlier retrieval diagnostics. |
| [Joint-routing assessment](joint-routing-assessment.md) | Historical Q/K comparison and its stopping decision. |

Historical assessments describe the evidence available at their stage. Read the capability summary for the latest task-level synthesis and the architecture assessment for component findings.
