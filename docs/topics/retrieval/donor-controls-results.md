# Does the attention donor need to match the example?

**No, not for near-complete rescue in this test.** An unrelated compact donor with the same answer also works. A different-answer donor usually makes the model predict the donor's answer instead.

## Short explanation

Previously, we copied block 2's attention output from the normal version of the same input. Now we copy it from another input. That input has a different query key, mapping and parent episode.

When both inputs have the same answer, predictions remain highly accurate. When their answers differ, predictions usually switch to the donor's answer. The copied activation therefore carries answer-relevant information that strongly influences the output. We have not proved it contains only the answer.

## Method

The [fixed protocol](../../donor-controls-protocol.md) uses five frozen models and the same 5,120 inspected crops per model. No training. Replace block 2's attention output at every token position using the verified explicit forward. Reverse means copying shifted activations into the compact recipient. Other donor conditions copy compact activations into shifted recipients.

Choose unrelated donors by a deterministic cyclic scan starting at the next parent episode. Require a different parent, query key and sorted three-pair mapping. Separate same-answer and different-answer donors. Exact indices are saved. Selection uses labels, so this is a diagnostic control, not a practical repair. Donors can repeat and observations are dependent.

## Findings

Means across five seeds. Accuracy uses the recipient's true answer; donor agreement uses the donor's true answer.

| Condition | Recipient accuracy | Donor-answer agreement | Cross-entropy, nats |
|---|---:|---:|---:|
| Compact baseline | 99.996% | n/a | 0.0005 |
| Shifted baseline | 60.60% | n/a | 2.2016 |
| Same-example compact donor into shifted | 99.984% | 99.984% | 0.0015 |
| Reverse: shifted donor into compact | 58.71% | 58.71% | 2.3533 |
| Unrelated compact donor, same answer | 99.36% | 99.36% | 0.0201 |
| Unrelated compact donor, different answer | 0.18% | 99.34% | 8.8422 |

| Seed | Reverse accuracy | Same-answer unrelated accuracy | Different-answer recipient accuracy |
|---|---:|---:|---:|
| 137 | 66.25% | 99.88% | 0.14% |
| 163 | 58.34% | 99.80% | 0.08% |
| 191 | 59.10% | 99.20% | 0.14% |
| 223 | 55.47% | 99.04% | 0.57% |
| 257 | 54.37% | 98.87% | 0% |

| Predeclared check | Outcome |
|---|---|
| Reverse lowers compact accuracy by >=10 points in every seed | Passed |
| Same-example rescue beats same-answer unrelated donor by >=10 points on average | Failed: only 0.625 point advantage |
| Different-answer donor agreement >=90% in every seed | Passed |

## What this changes

The earlier rescue is not strongly specific to the matching example under the stated criterion. Copying answer-relevant output can override the recipient's computation. This limits claims that the patch repaired its original retrieval mechanism. Reverse harm supports influence under intervention, not necessity in every natural computation. Neither result localizes the original error to block 2; donors include upstream information.

We replaced all tokens, so this does not isolate the final query position. Donor labels and examples are controlled, but donor magnitudes and other features are not matched. These are mixed internal states on reused data. No general architecture or paper benchmark claim follows; [prior source comparisons](../training/depth-comparison-results.md#paper-comparison-and-remaining-question) remain applicable.

## Verification and next step

[Script](../../../experiments/donor_controls.py) and [raw results with donor indices](../../../results/donor_controls.json) preserve provenance. Compact, shifted and same-example rescue metrics reproduced within 1e-12. Same-state attention shams passed below 1e-10. Donor restrictions, labels, finite outputs and unchanged checkpoint hashes were checked. Source hashes and paired rescue/harm arithmetic passed separate checks.

Follow-up completed: [Q/K versus V interventions and learning guide](../attention/qkv-results.md). Matched QK largely rescues while keeping shifted V; unrelated partial donors do not reproduce full-output transfer. Head/token localization remains open.
