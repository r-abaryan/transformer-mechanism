# Beta: writing a new value while disturbing old reads

Beta is the strength of a delta-memory correction. The [DeltaNet equations, Yang et al. 2024, section 2.2](https://arxiv.org/html/2406.06484v2) give, in our transposed state convention:

\[
S'=S+\beta k(v-S^{\mathsf T}k)^{\mathsf T}.
\]

First read the value already associated with k. Compute its error relative to v. Write a fraction beta of that correction. Our beta is one learned scalar; the paper's beta depends on the input.

| Beta | Meaning for a unit key |
|---|---|
| 0 | No update; an initially empty memory stays empty |
| Between 0 and 1 | Partial correction toward the new value |
| 1 | The current key reads its new value exactly after this write |

The last statement concerns a value vector, not a guaranteed correct classifier output or protection of other keys.

## Why another key can be affected

For another query q, one write changes its read by:

\[
\Delta r(q)=\beta(q^{\mathsf T}k)(v-S^{\mathsf T}k).
\]

This is a local derivation from the published update. Orthogonal q/k give zero change. Otherwise the overlap, correction error and beta all matter. Smaller beta shrinks this one correction with S/k/v fixed. Across a full history, earlier states also change; accuracy need not improve monotonically.

For two unit-key writes from an empty state, let `c=k_0^T k_1`. The first key's read becomes:

\[
r(k_0)=\beta(1-\beta c^2)v_0+\beta c v_1.
\]

Our preflight verifies this formula and its beta derivative. It is a fixture, not a diagnosis of trained errors.

## What the swap tests

Copy the opposite arm's beta into a saved model. Keep its learned keys, values and classifier fixed. This isolates an inference effect within that checkpoint. The new read scale may suit its classifier poorly, so a failed swap does not prove beta was irrelevant during training.

Immediate answers use each key's first-write prefix. Complete maps use one shared final state. These are different measurements. [Results](beta-swap-results.md) report both directions and their harms; [protocol](../../protocols/beta-swap.md) fixes the scope. No retraining or fresh confirmation occurs.
