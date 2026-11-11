# Joint routing: evidence and stopping decision

Keep Q and K together for the next evaluation. Joint replacement is the strongest tested rescue. The separate tests helped interpret that rescue; they are not proposed standalone model changes. Stop the planned token-key breakdown and further fine-grained patches in this cycle.

## What the evidence supports

Five varied-position models, same opened 400 test mappings. Percentages are seed means unless stated otherwise.

| Observation | Evidence | Bounded conclusion |
|---|---|---|
| Ordinary +8 accuracy: 77.65% | [Position training](../topics/positions/position-training-results.md) | Training on offsets 0..4 does not ensure transfer to +8 |
| +4 donor Q: 79.86%; K: 97.49%; joint QK: 99.44% | [Q/K comparison](../topics/attention/query-key-results.md) | Joint replacement is best in every seed; K is the stronger individual intervention |
| Joint QK rescues 7,870 answers and harms 25 across 36,000 seed/example observations | [Q/K comparison](../topics/attention/query-key-results.md) | Large diagnostic benefit, with residual errors and dependent observations |
| Same-slot/different-answer donor: 97.81% correct | [Slot/answer control](../topics/attention/donor-slot-results.md) | Rescue does not require the donor to have the same answer |
| Different-slot donors: about 96.6% recipient slot-value agreement | [Slot/answer control](../topics/attention/donor-slot-results.md) | Transferring QK largely transfers the selected slot |

## Why this is not yet a repair

Attention uses A = softmax(QK^T/sqrt(d_head) + mask), then combines V using A. Our joint patch supplies Q/K from the same model running the same example at +4. The +8 computation retains its values and residual path.

The model therefore receives a route from a more reliable input position. The rescue does not show that it can construct that route independently at +8. Nor does joint performance prove special mathematical synergy: its measured accuracy gains were not super-additive. The practical finding is that the joint intervention performs best here.

Known: these donor-assisted routes can support high accuracy. Unknown: whether that result transfers to untouched mappings and farther offsets; how to obtain reliable routes in ordinary execution; whether longer sequences behave similarly. No architecture improvement or novelty claim follows.

## Next bounded question

Does the fixed joint QK intervention retain its benefit on fresh whole mappings, while ordinary performance is measured separately? The [confirmation protocol](../joint-routing-confirmation-protocol.md) fixes data exclusion, checkpoints, offsets, criteria and stopping rules before generating that set.

Status: [fresh-mapping confirmation completed](../topics/retrieval/joint-routing-confirmation-results.md). Joint accuracy reaches 99.31% at +8 and 96.99% at +12; both predeclared donor-assisted criteria pass. Ordinary robustness fails both. This is new-input confirmation using existing models, not independent training replication or ordinary routing robustness. The diagnostic cycle is closed.
