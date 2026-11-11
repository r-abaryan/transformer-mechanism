# Direct end-marker reading does not explain the weak seed's early EOS

2026-10-07. **Removing the source-end position's direct value contribution rescues 0/128 first failures. It increases EOS preference on every example.** The frozen support criterion fails. A matched-size position-zero subtraction rescues 42/128 (32.81%), so its effect is not interchangeable with end-position removal.

[Protocol](../../protocols/copy-end-route.md) · [Report and audit](../../../results/copy_end_route/report.json) · [Every group and mode](../../../results/copy_end_route/metrics.csv) · [Saved logits and norms](../../../results/copy_end_route/evidence.pt)

## What was changed

At fixed native attention weights, the projected contribution from position j is

`c_j = concat_h(A[h,t,j] v[h,j]) W_O^T`.

Subtract `c_end` before decoder normalization. Keep the attention weights, other contributions, output bias and residual unchanged. Then run the real normalization, FFN and readout. The control subtracts `c_0 × ||c_end|| / ||c_0||`, using source position zero fixed in advance. It matches vector size, not meaning. No key mask or softmax renormalization is used.

Four models are evaluated at the weak model's 128 first-error rows on test10, test10 terminal rows, test6 content position 4, and test6 terminal rows. Prefixes contain the correct preceding tokens. These are single-decision interventions, not generated full-copy scores. In this one-block model, the operations after the intervention are positionwise; changing other rows does not feed back into the selected row.

## Weak-seed first failures

Margin means EOS logit minus correct-content logit. Positive values favor EOS over that content token; final correctness still uses all classes.

| Mode | Correct /128 | EOS /128 | Mean EOS margin |
|---|---:|---:|---:|
| Native | 0 | 128 | 6.452 |
| Remove end contribution | 0 | 128 | 8.465 |
| Matched-size control | 42 | 86 | 1.559 |

End removal raises the margin in all 128 cases. Under this operator, that direct contribution counteracts rather than creates the premature-EOS preference. This rejects the proposed direct-route recovery explanation. It does not rule out end-marker information distributed across other encoder positions or interactions downstream.

The control's partial recovery is not a discovered repair: it is an artificial scaled subtraction on opened examples. It does not show that the unscaled position-zero contribution alone caused the errors.

## Successful and terminal controls

Counts below are correct decisions, each out of 128. “Content” uses the same test10 examples and rows selected from the weak seed's failures.

| Model | Content: native → end removed | Terminal test10: native → end removed | Terminal test6: native → end removed |
|---|---:|---:|---:|
| Cross-only 3767 | 0 → 0 | 128 → 128 | 128 → 128 |
| Cross-only 3779 | 128 → 128 | 128 → 0 | 128 → 0 |
| Cross-only 3793 | 128 → 128 | 128 → 0 | 128 → 0 |
| Combined 3767 | 128 → 128 | 128 → 0 | 128 → 0 |

All models retain 128/128 correct short-content decisions under either subtraction. Successful models lose every tested terminal decision after end removal; the weak model retains all. Thus these checkpoints differ in their dependence on the direct end-position route for correct stopping. This is a conditional endpoint finding, not a complete stopping circuit or explanation of its learning history.

## Control limitations matter

The control is eligible on every weak-model row. On its first-failure rows, scaling ranges from 1.25 to 12.94. On its short-content rows, tiny position-zero contributions require factors up to 1.86 billion. Such a control matches the removed norm but is far from a native contribution.

For each successful model, the control norm is at most 1e-12 on the long-content, long-terminal and short-content cohorts. As frozen, these control rows receive no subtraction and are marked ineligible, not evidence of specificity. On short terminal rows it is eligible: correct stopping falls to 0/128, 124/128 and 66/128 for cross-only 3779, cross-only 3793 and combined 3767. Preserve these harms. Do not conclude that only the marker route can affect stopping.

## Evidence and boundary

Initialized saved-reload, native identity, contribution reconstruction, zero-removal, matched-norm, zero-norm, synthetic rescue/harm and wrong-sign checks passed before scoring. Producer hooks and explicit reference equations agree across 16 cells ×128 rows ×3 modes = 6,144 selected-row evaluations, with zero prediction differences and maximum logit discrepancy 1.750e-13. Native predictions match archived scores; 46 protected files remain unchanged. Shared projection/attention utilities limit reference independence. Zero training and no free-running intervention evaluation.

[Newman et al. (2020), sections 1 and 3](https://arxiv.org/html/2010.07174v1#S3), inspected in the preceding review, motivates separating stopping and content; it does not predict this result. Contribution subtraction is a local derivation from the executed model equations. No novelty claim.

Stop this diagnostic. The hypothesis failed informatively: the weak model's early EOS is not repaired by removing this direct end route, while successful models depend on it at tested stopping positions. The remaining origin of the EOS preference stays unresolved; no additional route or head sweep is queued.
