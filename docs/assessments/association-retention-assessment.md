# Association load and retention: what the evidence now supports

2026-10-04. Scope: compact softmax/delta readers, known symbols and six value classes. The load task uses consistent assignments. This is not a general architecture ranking.

| Question | Evidence | What remains unresolved |
|---|---|---|
| Can the recipes fit high-load training contexts? | [Association load](../topics/attention/association-load-results.md): both readers fit every load at 100%; delta held-out accuracy falls to 58.42% at 32 keys | Why the learned solution transfers poorly |
| Does repeating the same information repair the gap? | [Repetition](../topics/attention/history-repetition-results.md): four passes improve 16 keys to 97.92%; 32 keys remain near 58% with rescues and harms | A general high-load repair; repetition is not uniformly beneficial |
| Are the native final errors already present when the requested value is written? | [Order trace](../topics/attention/query-write-order-results.md): every immediate read is correct; later writes leave 479 wrong; query-last recovers all 479 | Training cause, literal information erasure and a repair retaining all queries simultaneously |
| Is compact state automatically faster overall? | [GPU diagnostic](../topics/attention/retrieval-gpu-resources-results.md): delta state is smaller and queries faster, but eager prefill is slower and training peaks higher | Optimized kernels and full-model quality/resource tradeoffs |

We now have a concrete behavioral explanation for the native 32-key errors: correct queried classifications do not survive the remaining writes. Earlier evidence only showed load sensitivity and mixed cross-effects. The trace localizes when the native failures arise.

The update equation permits interference through overlapping learned keys and nonzero correction residuals. It does not isolate which learned geometry, write strength or training objective should change. Sixteen key dimensions alone do not establish a 16-association classification ceiling.

Query-last is informative because all information and length remain unchanged. It is also privileged: a different query requires a different reordered history. Its success is not proof that one final matrix retains every association correctly.

Close the three-step load/repetition/order investigation. A future change should target retention, use adequate training fit and fresh confirmation, and demonstrate benefit without query-specific input reordering. No new training recipe or architecture is selected here.
