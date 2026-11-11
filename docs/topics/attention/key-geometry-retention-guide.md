# Why separate key directions preserve separate associations

Delta stores values in one matrix. Another write changes a query read by:

\[
\Delta r=\beta(q^{\mathsf T}k)(v-S^{\mathsf T}k).
\]

If two different keys are orthogonal, their dot product is zero. Their writes cannot change each other's read. This follows from the [delta correction rule](https://arxiv.org/html/2406.06484v2), section 2.2; it is an established algebraic property, not a new architecture idea.

## Three geometries

| Intervention | Meaning | Expected distinction |
|---|---|---|
| Native | Keep learned 16-dimensional keys | Original shared geometry |
| Padded | Append 16 zeros to every key and query | More storage, identical dot products |
| Orthogonal | Give each of 32 known symbols a different coordinate | Distinct keys have zero overlap |

Padding does not change `q dot k`. Its extra coordinates stay unused under this frozen intervention. It is not a test of training a wider model.

For identity keys, each symbol has its own matrix row. With zero initial state and one write per key:

\[
S^{\mathsf T}k_i=\beta v_i.
\]

The value is scaled by beta, not necessarily stored as exactly v_i. Classification still depends on the saved value embeddings and readout. The experiment measures whether that decoding succeeds.

## What makes this different from query-last

Query-last changes the history for each requested key. Here, history order never changes. One final state answers every key. The geometry uses key identities only; test values and labels do not choose its coordinates.

The cost is explicit: 32 orthogonal directions require at least 32 dimensions. State grows from 16x8 to 32x8. This gives each known symbol a dedicated slot and does not establish efficient learned retention for arbitrary symbols or a solution at width 16.

[Measured results](key-geometry-retention-results.md). Earlier orthogonal-preservation [arithmetic](delta-memory-results.md) remains valid; this diagnostic tests its effect on saved learned values/readouts and complete-map classification.
