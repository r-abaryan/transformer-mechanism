# GRU versus Transformer retrieval

Frozen before execution, 2026-10-02. One comparison; no tuning or replacement seeds.

Question: does the GRU state-tracking advantage carry over to latest-value retrieval?

| Control | Fixed choice |
|---|---|
| Data | Reuse `results/mqa_training_data.json`; expand every context to all distinct query keys using the existing generator. Preserve family splits. These are previously opened data, not fresh confirmation. |
| Labels | Last value paired with the query key. Twelve key symbols, six value symbols, separator18. Exactly one repeated key per context. |
| Lengths | Train/validation3 and4 pairs; test3,4 and6. Sequence length2n+2. |
| Models | Existing MQA Transformer: width16, two blocks, two Q heads, one KV head, RoPE, ReLU32, LayerNorm;4,544 parameters. Single-layer GRU: embedding16, hidden27, bias-free19-class output;4,462 parameters. Zero initial hidden state. |
| Initialization | Seeds2131/2143/2161. Copy each Transformer's initial embedding values into its GRU control. Other tensors use native initialization; complete initial states cannot match across architectures. |
| Optimization | Each arm6,000 Adam updates, lr0.001, batch64, alternate3/4 pairs. Identical sampled indices within each seed. CPU float64, deterministic, one thread, installed torch2.11.0+cu128. Alternate arm order across seeds. |
| Objective | Final-query19-class cross-entropy only. No intermediate targets. This differs from the earlier state-tracking study. Equal updates/examples do not imply equal compute. |
| Measurements | Full training and validation pools after training; test only after all six attempts and a saved pretest snapshot. Every split/length: accuracy and loss overall and separately for repeated/unique queries. Periodic training minibatch loss every1,000 updates is descriptive, not full training fit. |
| Baselines | Uniform choice among six values: expected1/6. Always12, first matching value (ignores overwrite), final context value (ignores query), and exact last-match oracle. |
| Checks | Independent family/label audit; manual GRU gates and independent attention forward; all parameter gradients on small batches; causal prefix and batch isolation; intentional reset-at-every-token error. Final independent logits/metrics on every full pool, tolerance1e-10. |

For each architecture, reliability passes only if every seed and query type reaches99% on training/validation/test at3/4 pairs and90% on test6. Report each seed's six-pair repeated/unique mean and paired GRU minus Transformer difference. A consistent advantage requires the same direction and at least10 percentage points in every seed. Interpret a long-length gap as transfer-specific only if **both** architectures pass the short training/validation99% fit gate. Otherwise report incomplete fitting too. No universal winner or causal isolation of recurrence follows.

Stop after six attempts and audit, including failures. Nonfinite training is a failed attempt, not permission to restart. Save sources/protocol/input hashes and checkpoints. No language or reasoning experiment follows automatically.
