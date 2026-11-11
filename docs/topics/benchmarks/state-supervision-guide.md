# Intermediate-state supervision

The architecture receives the same initial bit, operations and QUERY token in both arms. The difference is where training supplies the correct state as a target.

| Input position | Correct state | Final-only loss | Intermediate arm |
|---|---:|---|---|
| Initial0 |0|None|None|
| SET1 |1|None|Predict1|
| FLIP |0|None|Predict0|
| SET0 |0|None|Predict0|
| FLIP |1|None|Predict1|
| QUERY |1|Predict1|Predict1|

These targets are training feedback. They are never extra input tokens. At evaluation the model must still answer using only the operations. No previous predicted state is fed back.

## Loss and gradients

For two-state logits `z`, probabilities are `softmax(z)`. Cross-entropy for correct class `y` is `-log p_y`; its logit derivative is `p - one_hot(y)`.

Final-only training uses `L_final = CE(z_QUERY, s_final)`.

Intermediate training uses `L_final + L_aux`, where `L_aux` is mean cross-entropy over all operation positions in the batch. The target at position`t` is the state after that operation. Averaging keeps the auxiliary coefficient independent of operation count. It does not make the two objectives equivalent.

The shared parameters receive `gradient(L_final) + gradient(L_aux)`. This can change which representations training favors. Whether it actually helps is an empirical question; the gradient formula does not guarantee improvement.

## What the comparison can establish

Each fresh seed has two equal-initialized controls, the same sampled inputs, optimizer and update budget. We measure final-answer accuracy on the complete training pool, validation and test, separately for each operation group. We also record accuracy of intermediate readouts.

If training fit improves, this objective package helped learning in our setup. If final held-out accuracy stays low despite good training fit, a generalization gap remains. Good intermediate readouts alone do not establish good final answers or a learned recurrent algorithm.

The intermediate arm receives1,920,000 extra operation-state targets per run. Its combined objective also has a different scale. We cannot attribute a change solely to target placement, or claim equal supervision/runtime cost. The previously opened data provide a matched comparison, not fresh confirmation.

[Frozen protocol](../../protocols/state-supervision.md). Six fixed attempts, followed by one audit and a stopping decision. Earlier experiments remain valid.

**Observed result:** [the completed comparison](state-supervision-results.md) improves mean twelve-operation no-reset test accuracy48.70% to71.09%. Training gains vary strongly by seed, and24-operation performance remains near50%. The fixed success rule fails. More feedback helped some learning, but did not establish reliable sustained tracking.
