# Joint routing on fresh mappings

The fixed joint QK patch transfers to fresh mappings at +8 and +12. Both predeclared donor-assisted criteria pass. Ordinary model robustness fails at both offsets. This confirms a bounded diagnostic rescue, not an unassisted solution.

## What was held fixed

Five varied-position checkpoints, seeds 461/487/509/541/571. No training, parameter selection, donor filtering or architecture changes. The [protocol](../../joint-routing-confirmation-protocol.md) was fixed before generating inputs. Its opening status records that pre-execution state; this report records the completed run.

Exclude all 4,800 mappings in the reserved-cycle and position-training splits. The matching-task code/data inventory identified those two source corpora; later routing diagnostics reuse their inputs. Select 400 new mappings, balanced over 20 value triples, and enumerate 18 order/query variants each. All 7,200 sequences were absent at whole-mapping level from those prior corpora. Save inputs and provenance before predictions.

This is new-input confirmation using existing models and the same task grammar. It is not independent training replication or proof of semantic novelty.

## How the joint patch works

For recipient offset s, replace block-2 Q/K with the same example's +4 Q/K from the same model:

A = softmax(Q_4 K_4^T / sqrt(8) + causal mask), followed by A V_s.

Both heads and all rows are patched. Recipient values, residual path, FFN, normalization and output remain active. A donor forward pass supplies the route; the model does not generate it independently at s. Ordinary +4 donor accuracy is 99.97% on average. No example is removed when the donor is wrong.

## Results

Five-seed means. Loss is cross-entropy in nats; lower is better. Mapping correctness requires all 18 variants to succeed, so it is stricter than sequence accuracy.

| Offset / execution | Accuracy | Loss | Mean logit margin | Fully correct mappings |
|---|---:|---:|---:|---:|
| +4 ordinary donor | 99.97% | .0019 | 8.08 | 99.55% |
| +8 ordinary | 77.51% | 1.1126 | 3.32 | 6.35% |
| +8 joint QK | 99.31% | .0214 | 6.63 | 91.55% |
| +12 ordinary | 60.78% | 1.9090 | .98 | .90% |
| +12 joint QK | 96.99% | .0851 | 5.50 | 74.75% |

The logit margin is the correct answer's score minus the best competing score. Its mean can be positive despite many errors. Mapping variants and seed/example observations are dependent.

| Seed | +8 ordinary | +8 joint | +12 ordinary | +12 joint |
|---|---:|---:|---:|---:|
| 461 | 88.36% | 99.82% | 74.21% | 95.88% |
| 487 | 84.35% | 99.65% | 77.44% | 99.96% |
| 509 | 82.15% | 99.69% | 53.40% | 96.24% |
| 541 | 62.94% | 99.13% | 49.08% | 96.74% |
| 571 | 69.72% | 98.26% | 49.78% | 96.14% |

| Predeclared endpoint | Outcome |
|---|---|
| +8 joint: every seed >=95%, mean gain >=10 points | Passed; minimum 98.26%, gain 21.81 points |
| +12 joint: every seed >=90%, mean gain >=5 points | Passed; minimum 95.88%, gain 36.21 points |
| Ordinary: every seed >=90%, separately at +8/+12 | Failed both; all five seeds below 90% at each |

Across 36,000 seed/example observations, +8 joint rescues 7,882 predictions and harms 32. At +12 it rescues 13,076 and harms 42. High mean accuracy does not mean every mapping is reliable: only 74.75% of mappings have all variants correct at +12, averaged across seeds.

## What this resolves

The original +8 joint rescue (99.44% on opened mappings) persists at 99.31% on this fresh set. These are separate input sets, not a controlled comparison of that small difference. The +12 stress endpoint also passes without modifying the patch.

Together with the [slot/answer controls](../attention/donor-slot-results.md), this supports a specific statement: suitable donor routing lets the recipient values and downstream computation answer most examples correctly. It does not show how to create that routing reliably in ordinary execution, isolate an upstream cause, or establish longer-sequence performance. Every input is still eight tokens.

This closes the donor-assisted confirmation cycle. Keep joint QK as a validated diagnostic within this scope. A subsequent [fresh evaluation of canonical positions](../positions/canonical-position-results.md) supplies a donor-free, single-pass solution for irrelevant global offsets by resetting input position IDs. It changes the whole position-dependent computation and does not establish learned routing invariance or a novel architecture. No additional patching or retraining was performed in this confirmation experiment.

## Evidence and verification

[Experiment](../../../experiments/confirm_joint_routing.py), [fresh inputs](../../../results/joint_routing_confirmation_inputs.json), [summary/provenance](../../../results/joint_routing_confirmation.json), [full logits](../../../results/joint_routing_confirmation_scores.json), [audit](../../../results/joint_routing_confirmation_audit.json). Whole-mapping exclusion, complete variants, exact label/rank/value-triple balance, finite outputs, checkpoint/source hashes and criterion arithmetic passed.

Library/explicit baseline, causal normalized attention and sham checks passed with maximum error 1.05e-13. The separate audit reproduced all 25 summaries, including losses and margins, from saved logits with zero discrepancy. It shares PyTorch and does not independently rerun interventions. No new paper claim is tested; [prior source comparisons](../positions/position-routing-results.md#sources-and-verification) remain context, not external validation of these measurements.
