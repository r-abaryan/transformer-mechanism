# More events versus more things to remember

`A:1, B:4, A:1, B:4` has four events but only two associations. Repeating it makes the history longer without enlarging the current map.

Our earlier successful delta task used eight key symbols. This follow-up holds the history at **32 events** and varies how many different keys it contains:

| Distinct associations | Writes per key | What stays fixed |
|---|---:|---|
| 2 | 16 | 32 events; 16x8 delta state |
| 8 | 4 | Same widths and 615 parameters |
| 16 | 2 | Same query and value vocabulary |
| 32 | 1 | Same training budget per load |

Every key keeps the same value within its context. Order is unnecessary for answering this task. Softmax therefore uses similarity without recency. Both readers must learn the key/value geometry and classifier.

Softmax keeps every event's K/V. Delta shares one matrix among the associations. Its write corrects what the incoming key currently reads:

\[
S'=S+\beta k(v-S^{\mathsf T}k)^{\mathsf T}.
\]

An existing query q changes by:

\[
S'^{\mathsf T}q-S^{\mathsf T}q
=\beta(q^{\mathsf T}k)(v-S^{\mathsf T}k).
\]

Two factors matter: how much the keys overlap, and how large the write correction is. Orthogonal keys give zero cross-change. A write already read perfectly also gives zero change. Learned keys need not be orthogonal.

The diagnostic adds one redundant write for another key. The answer to the original query stays the same. Changed read vectors need not change the predicted class; harmful class changes are measured separately.

**16 key dimensions is not automatically a 16-key classification limit.** Values come from six classes, and the classifier only needs the correct label. A dimensional argument about arbitrary exact vector reads does not settle this discrete task.

At fixed length, increasing distinct keys also reduces repetitions. If high-load training and test both fail, the recipe did not master that condition. If training succeeds and held-out contexts fail, a generalization gap exists. Neither pattern alone identifies a hard capacity limit.

[Frozen methods](../../protocols/association-load.md). Established update: [DeltaNet 2024, section 2.2](https://arxiv.org/html/2406.06484v2). This study tests compact learned readers, not a complete language model.
