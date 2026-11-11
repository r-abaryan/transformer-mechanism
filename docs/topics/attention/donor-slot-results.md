# Donor target slot versus donor answer

The QK patch mostly transfers the donor's selected slot. In varied-position models, changing that slot makes predictions follow the recipient's value at the new slot about 96.6% of the time. Matching the donor answer does not prevent this.

## What was tested

Use all ten frozen models, +4 donors and +8 recipients. Every recipient gets four unrelated donors crossing same/different target slot with same/different answer. Each donor comes from another whole mapping. Selection uses labels and ranks, never model predictions. One deterministic donor draw per condition is shared across all models. The [protocol](../../donor-slot-protocol.md) and donor indices were saved before evaluation.

Replace block-2 Q/K only, in both heads. Recipient V, residual and downstream computation stay active. This is exploratory on the same opened 400 mappings, with 18 order/query variants each. Donors may be reused; observations are dependent. Other donor content is not held constant across the four conditions.

## Learn the operation

The patch computes A_d = softmax(Q_d K_d^T / sqrt(8) + mask), then uses A_d V_r. Subscripts d and r mean donor and recipient. Both Q and K come from the donor, so their routing computation does not compare the recipient query with recipient keys.

Example: the recipient values are [red, blue, green], and its correct answer is blue. A donor route pointing to slot 3 can read green from the recipient V, even if the donor's own correct answer is blue. The experiment checks how closely actual predictions follow this expectation; residual paths and contextualized V could have changed the outcome.

## Results

Five-seed means for varied-position models. "Slot value" means the recipient's value at the donor's target slot. Agreement columns can overlap when labels coincide.

| Donor target slot | Donor answer | Correct recipient answer | Recipient slot-value agreement | Donor-answer agreement |
|---|---|---:|---:|---:|
| Same | Same | 99.18% | 99.18% | 99.18% |
| Same | Different | 97.81% | 97.81% | 0.27% |
| Different | Same | 0.75% | 96.59% | 0.75% |
| Different | Different | 1.15% | 96.62% | 20.36% |

For context, unchanged +8 accuracy is 77.65%; same-example +4 QK gives 99.44%. Sampled donors themselves answer correctly at +4 at least 99.958% on average in each condition. Different-slot failure is therefore not explained by broadly inaccurate donors.

The final row needs care. The donor answer equals the recipient slot value in 20.40% of examples by coincidence. Restricting to the 5,731 examples where these labels differ gives **96.24% slot-value agreement versus 0.44% donor-answer agreement**. This distinguishes routing to recipient content from copying the donor's answer.

Per-seed recipient accuracy (%). The last column also shows whether errors follow the selected slot.

| Varied seed | Same slot/same answer | Same slot/different answer | Different slot/same answer | Different slot/different answer | Last condition: slot agreement |
|---|---:|---:|---:|---:|---:|
| 461 | 99.78 | 99.47 | 0.31 | 0.46 | 98.96 |
| 487 | 99.64 | 99.01 | 0.32 | 0.57 | 98.71 |
| 509 | 99.88 | 99.62 | 0.31 | 0.50 | 98.46 |
| 541 | 98.60 | 97.25 | 1.46 | 1.44 | 96.94 |
| 571 | 98.01 | 93.69 | 1.38 | 2.76 | 90.01 |

Fixed-position models are weaker controls. Their corresponding four recipient accuracies are 50.44%, 48.08%, 18.44% and 19.12%; different-slot agreement is only about 48%. Their sampled +4 donor accuracy is about 57%, so donor routes are not a reliable source in this group.

## What this establishes, and what remains unknown

Observed: slot alignment strongly predicts rescue under these QK replacements. Answer agreement alone does not. Same-slot rescue remains high when donor answers differ; moving the slot redirects most answers to recipient content at that slot.

Bounded interpretation: the earlier 99.44% rescue supplies a useful route to retained recipient values. It does not demonstrate that the recipient independently recovered the correct key-value relationship. QK and V remain contextualized representations; this is not proof that either contains only routing or only content.

Unknown: why normal +8 routing breaks, which of Q versus K contributes most, and how block 1 constructs those representations. This diagnostic does not settle those questions or prove length generalization. One sampled donor assignment and this task family limit generality.

Follow-up completed: [Q-only versus K-only replacements](query-key-results.md). K-only raises +8 accuracy to 97.49%, versus 79.86% with Q-only; joint QK reaches 99.44%. Further token-key breakdown is now dropped. See the [joint-routing assessment](../../assessments/joint-routing-assessment.md) and fresh-mapping confirmation plan. More patches on this opened set would not be fresh confirmation.

## Evidence and checks

[Experiment](../../../experiments/donor_slots.py), [donor indices](../../../results/donor_slots_indices.json), [results](../../../results/donor_slots.json), [saved full logits](../../../results/donor_slots_scores.json), [audit](../../../results/donor_slots_audit.json). Loss, margins, rescue/harm and donor correctness are retained in the raw results. All 50 intervention summaries were independently recomputed from saved logits with zero discrepancy. Mapping exclusion, all four donor conditions, immutable hashes, finite outputs and prior baseline/matched-reference metrics passed. Maximum sham/reconstruction discrepancy: 4.93e-14. The audit does not rerun an independent forward implementation.

The [preceding report's primary-source comparison](../positions/position-routing-results.md#sources-and-verification) explains why attention inspection alone is insufficient. Here, controlled replacements add behavioral evidence, while the slot/answer controls narrow its interpretation. This remains a local diagnostic, not a reproduction of those papers or a novel architecture claim.
