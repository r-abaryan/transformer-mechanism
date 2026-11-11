# Positional representation: closed assessment

**Prefer RoPE for the tested recall recipes. Do not call it a general solution.** Both comparisons are closed.

| Task, five paired seeds each | Sinusoidal six-pair mean | RoPE six-pair mean | Frozen decision |
|---|---:|---:|---|
| [Unique-key lookup](../positional-comparison-results.md) | 81.91% | 99.86% | All criteria pass. |
| [Latest-value repeated-key lookup](../topics/training/repeated-key-results.md) | 57.18% | 93.51% | Large gains in every seed, but two seeds below90%; reliability rule fails. |

These are separately trained cohorts on different tasks. Compare methods within each row, not task difficulty from percentages alone. Both studies passed independent checkpoint audits.

RoPE rotates projected Q/K instead of adding positions to embeddings. This changes several information paths. The paired comparisons support that complete replacement within their recipes; they do not isolate the mechanism responsible for improvement. Neither study proves query-token dependence on unrestricted repeated-key inputs or general language-model superiority.

**Stopping decision:** preserve both gains and failures. No extra positional tuning, alternative search or seed replacement on these opened tests. The next bounded question concerns [FFN activation](../ffn-activation-guide.md), keeping RoPE fixed.
