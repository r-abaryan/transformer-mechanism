# Sparse attention: what we established

This branch is closed. Four studies connect arithmetic, graph structure and learned behavior without establishing a general architecture winner.

| Evidence | Supported finding | Boundary |
|---|---|---|
| [Sliding-window math](../topics/attention/sliding-window-results.md) | Explicit local attention matches the same dense mask; L causal layers of radius r have maximum lookback Lr | Possible access does not establish learned use |
| [Learned local retrieval](../topics/attention/window-retrieval-results.md) | All seeds retrieve reachable marked values exactly; unreachable groups score1/6 | Simple task; longer inputs preserve retrieval distances |
| [Global-token math](../topics/attention/global-attention-results.md) | Global edges shorten paths; causal relay requires source<=global<=query | Future-reading global states can leak through later layers |
| [Learned global relay](../topics/attention/global-retrieval-results.md) | Global distant retrieval works at length17 and falls to1/6 after edge removal | One length21 seed scores83.33%; all-seed transfer and combined dependence rules fail |

**Architectural boundary:** when no permitted path exists, changing an inaccessible input cannot change the output under these local-only operations. Matched six-answer families make the one-sixth ceiling explicit.

**Learning boundary:** a permitted path can be learned and still generalize unreliably. The weak global seed has a path; its error cause remains unidentified. No amount of counting edges establishes a learned representation.

**Execution boundary:** fewer useful attention pairs do not establish lower runtime or measured memory. Our trained sparse patterns used dense masked arithmetic. Efficient sparse kernels remain a separate question.

No further window, global-position or seed tuning follows. The next distinct mathematical topic is feature-map linear attention: replace pairwise storage with accumulated state and verify the resulting function. That changes the similarity rule, whereas these sparse studies changed visibility.
