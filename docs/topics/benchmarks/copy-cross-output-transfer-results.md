# Transferring the whole cross-attention output

**The transplant fails. Both directions damage short copying.** The successful model's output is not a working replacement inside the weak model. This does not isolate the cause of the learned exposure benefit.

## What was changed, and why?

Cross-attention returns a vector, not a final answer. The decoder combines it with its own residual, then applies normalization, its FFN and classifier:

\[
c=W_O\operatorname{concat}_h(A_hV_h)+b_O,\qquad u=\operatorname{LN}(x+c).
\]

We replaced \(c\) with the other model's output while keeping the recipient's \(x\) and downstream layers. Both models read the same source and the recipient's current generated prefix. No correct future tokens or stopping decisions were supplied. All heads move together, so matching head identities is unnecessary.

The norm control keeps the recipient direction: \(c'=c\|c_{donor}\|/\|c\|\). It tests size alone. It cannot guarantee compatibility between learned representations.

## Method

One reused seed, 3767; original cross-only and combined-exposure checkpoints at update 6,000. Both native models fit the ordinary short task. Two recipients × four modes × 14 pools = **112 evaluated cells**, with zero training updates. Generation stops at EOS or the existing length+3 limit. Teacher-forced loss is measured separately. All ordinary train/validation/test pools at lengths 3–6 and test pools at 8/10 are included; layout and uniform pools are excluded. CPU float64, one thread. Formal scoring and its first reference checks took 21.7 seconds; tiny saved models require no training here.

The [frozen protocol](../../protocols/copy-cross-output-transfer.md) requires at least 95% complete copying and correct stopping in every long group, a 10-point length-10 gain, and at least 99% in every short group. Reciprocal sensitivity and magnitude specificity have separate criteria.

## Results

Complete generated copies, including EOS. Length-8 groups contain 64 examples each; length 10 contains 128 sequences with repeated symbols and no distinct group. “Distinct” means all source symbols differ.

| Recipient | Cross output | Length 8 distinct | Length 8 repeated | Length 10 repeated |
|---|---|---:|---:|---:|
| Weak cross-only | Native | 59/64 | 60/64 | 0/128 |
| Weak cross-only | Combined donor | 1/64 | 1/64 | 2/128 |
| Weak cross-only | Donor norm only | 62/64 | 62/64 | 0/128 |
| Successful combined | Native | 64/64 | 64/64 | 128/128 |
| Successful combined | Weak donor | 0/64 | 0/64 | 0/128 |
| Successful combined | Donor norm only | 61/64 | 58/64 | 126/128 |

Self-transfers reproduce all native logits within tolerance and every generated token exactly.

Short-task preservation matters: a transplant that breaks familiar inputs cannot cleanly explain the long-input failure. Minimum complete-copy accuracy across lengths 3–6 and distinct/repeated groups:

| Recipient / mode | Training | Validation | Test |
|---|---:|---:|---:|
| Either native or self | 100% | 100% | 100% |
| Weak / full transfer | 3.13% | 0% | 4.69% |
| Weak / norm only | 100% | 100% | 100% |
| Combined / full transfer | 0% | 0% | 0% |
| Combined / norm only | 97.66% | 96.88% | 96.88% |

Stopping and content are separate measurements. At length 10:

| Recipient / mode | Correct full content | On-time EOS | Early EOS | Missing EOS | Teacher CE |
|---|---:|---:|---:|---:|---:|
| Weak / native | 0/128 | 0/128 | 128/128 | 0/128 | 0.6281 |
| Weak / full transfer | 2/128 | 2/128 | 126/128 | 0/128 | 1.0457 |
| Weak / norm only | 0/128 | 0/128 | 128/128 | 0/128 | 0.6543 |
| Combined / native | 128/128 | 128/128 | 0/128 | 0/128 | 0.000254 |
| Combined / full transfer | 8/128 | 19/128 | 0/128 | 109/128 | 0.7972 |
| Combined / norm only | 126/128 | 128/128 | 0/128 | 0/128 | 0.004767 |

No late EOS occurs in these cells. On-time stopping need not mean correct content: the combined transplant stops correctly in 19 cases but completes no correct copies.

On length-10 teacher rows, weak recipient/donor mean norms are 4.980/5.791; donor/native norm ratios range 0.811–2.055. The reverse range is 0.487–1.233. No near-zero fallback occurs anywhere. Generated-step norms use only active rows and are recorded separately because each mode produces different prefixes.

## What follows?

All three support gates fail; both short-preservation gates fail. Full-vector replacement causes much greater damage than the tested magnitude control. This rules out treating the successful cross output as a directly portable repair in these recipient coordinates. It does **not** prove a unique direction-based cause, that cross-attention is irrelevant, or that coordinate mismatch alone caused the failure. Residual and downstream compatibility remain unresolved.

The earlier fifth-token pattern transfer tested a different intervention within one model. Its success is compatible with this failed cross-model transplant. Neither establishes how positional exposure learned the successful behavior.

**Stop here without a component, head or checkpoint sweep.** Any follow-up must address compatibility explicitly and distinguish two mechanisms while preserving familiar-task behavior. Another arbitrary transplant would repeat this limitation.

## Evidence and source boundary

The producer uses hooks in the unchanged native forward. A separate reference reconstructs matrix RoPE, attention, normalization, FFN and scalar generation. Saved initialized-model fixtures cover arbitrary prefixes and absent groups; deliberately corrupted generation is rejected. Native outputs match the historical saved scores. The persisted audit passes all 112 cells and 328 CSV group rows; maximum logit difference is 1.403e-13. All 102 protected files remain unchanged. These checks verify the tested computation, not a general learned mechanism.

- [Full metrics and norm statistics](../../../results/copy_cross_output_transfer/report.json), [all group scores](../../../results/copy_cross_output_transfer/metrics.csv), [persisted audit](../../../results/copy_cross_output_transfer/audit.json).
- [Producer](../../../experiments/copy_cross_output_transfer.py), [independent reference/audit](../../../experiments/audit_copy_cross_output_transfer.py), [preflight](../../../results/copy_cross_output_transfer/preflight.json).
- Primary equation source inspected: [Vaswani et al. (2017), §§3.1–3.2](https://arxiv.org/html/1706.03762v7#S3), multi-head output and residual normalization. The transplant, thresholds and measured scores are local work; the paper makes no prediction for them.

Limits: one selected same-seed pair, opened pools, final checkpoints, toy model and no fresh confirmation. Historical optimizer causes and causal mediation of positional exposure remain open.
