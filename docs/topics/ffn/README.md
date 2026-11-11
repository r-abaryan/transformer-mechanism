# Feed-forward networks

[Learned balancing guide](moe-balance-training-guide.md) | [Six-run results](moe-balance-training-results.md). Routing improvement with a failed CE-cost criterion.

[MoE load-balancing guide](moe-balancing-guide.md) | [Results](moe-balancing-results.md): auxiliary gradients, assignment counts and overflow counterexamples.

[MoE routing: educational guide](moe-routing-guide.md) | [Audited results](moe-routing-results.md). Top-one expert selection, probability gradients and capacity handling; no training.

Activations, gates, width and branch contributions.

[Consolidated FFN assessment](../../assessments/ffn-assessment.md): findings, limits and stopping boundary.

## Learn

- [FFN activation: ReLU versus GELU](../../ffn-activation-guide.md)
- [Next subject: the feed-forward network](ffn-investigation.md)
- [FFN width: what32 versus64 changes](../../ffn-width-guide.md)
- [SwiGLU: what the gate actually does](swiglu-guide.md)

## Findings

- [FFN branches: tested contribution and math](ffn-branch-results.md)
- [Next step completed: FFN effects through final normalization](ffn-normalization-results.md)
- [FFN width results:32 versus64](ffn-width-results.md)
- [FFN activation results: ReLU versus GELU](gelu-comparison-results.md)
- [SwiGLU versus ReLU: matched FFN projection cost](swiglu-results.md)

[All subjects](../README.md) · [Frozen protocols](../../protocols/README.md)
