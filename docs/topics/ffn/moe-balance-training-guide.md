# Does balancing change learned routing?

This study trains a small MoE classifier twice per seed. One version minimizes classification loss. The other adds the audited balancing objective:

\[
L=L_{CE}+\alpha E\sum_i f_iP_i,\qquad \alpha\in\{0,.01\}.
\]

All weights, data streams and budgets match within each pair. Only alpha changes. This isolates the added objective within this recipe; it does not compare MoE with dense Transformers.

## What does it learn?

Each input supplies a group indicator, two signed numbers and two distractors. Group 0 asks whether the signs agree. Group 1 asks whether they differ. Labels and groups are balanced. The rules are shared across disjoint training, validation and test examples.

The model transforms the six inputs into 16 features, routes to one of four small experts, adds a residual and predicts one of two classes. The residual can help solve the task without expert computation. Correct answers therefore do not prove expert specialization.

## Three measurements, three questions

| Measurement | Question |
|---|---|
| Group accuracy and cross-entropy | Does it learn both rules, and at what predictive loss? |
| Assignment distance from uniform | Are token choices spread across experts? |
| Overflow | How often is a chosen expert already full? |

We measure per-batch total variation, \(TV=\frac12\sum_i|f_i-1/E|\). Zero means equal assignment counts. With batch 128, four experts and 32 slots each, overflow fraction equals TV exactly: positive excess counts equal half the total absolute deviations. These are related measurements here, not independent confirmations.

Batch order matters. The first 32 arrivals to an expert are accepted; later arrivals skip its branch. Evaluation uses the same fixed batches for both models. Results do not establish invariance to other batching policies.

## How to read the outcome

Better routing balance is useful evidence even if a strict adoption rule fails. Preserve both results. A rise in cross-entropy can fail the rule even when class accuracy stays unchanged. Poor training fit would limit conclusions about generalization.

The frozen rule requires adequate fitting, lower imbalance and overflow in every pair, and limited quality cost in every group. No extra training follows a failed rule. [Protocol](../../protocols/moe-balance-training.md) | [Balancing mathematics](moe-balancing-guide.md).

[Executed results](moe-balance-training-results.md): both fit gates passed and overflow improved in every pair. The CE-preservation rule failed. All test answers were correct; their assigned probabilities still differed.
