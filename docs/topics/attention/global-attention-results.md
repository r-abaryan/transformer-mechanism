# Global-token results

Executed and closed2026-10-02.96 numerical cases,96 reachability cases and64 pair-count cases pass. No training.

[Educational guide and diagrams](global-attention-guide.md) | [Frozen protocol](../../protocols/global-attention.md)

## Arithmetic

CPU float64, one thread. Seeds2473/2477/2503; lengths5/9; radius0/2; global sets empty, first, middle, first+last; causal and bidirectional. Batch2, heads2, Q/K width3 and V width2. Dense masked softmax is compared with independently enumerated gathered keys. All Q/K/V gradients and one normalized joint directional derivative per case are checked. Empty-global cases match the existing local-attention implementation exactly.

| Seed | Cases | Maximum output error | Maximum gradient error | Directional error |
|---|---:|---:|---:|---:|
| 2473 | 32 | 6.66e-16 | 1.73e-17 | 2.93e-11 |
| 2477 | 32 | 3.33e-16 | 2.08e-17 | 4.07e-11 |
| 2503 | 32 | 4.44e-16 | 1.39e-17 | 2.67e-11 |

Output/gradient tolerance1e-10; central-difference tolerance1e-7 with step1e-6. These are supplied Q/K/V checks; learned projection parameters and a complete Transformer were not tested.

## Reach and controlled interventions

Independent set propagation and adjacency powers agree for depths1/2/3 over the32 visibility configurations. A separate uniform-averaging stack has positive input/output derivatives exactly where a path exists and zero elsewhere. All nonempty bidirectional global sets yield all-pairs reach at depth2. Every causal case excludes future inputs at every tested depth.

| Controlled example, length9/radius1 | Measured effect |
|---|---:|
| Remove global query connections, distant value impulse |1.00 output change |
| Remove global key connections, global value impulse |2.25 output change |
| Unsafe global0 reads input8: effect on output4 after one layer |0 |
| Same unsafe construction after two layers |0.3333333333333333 |
| Correct causal construction after two layers |0 |
| Correct derivative of output4 with respect to input8 |0 |
| Correct causal middle-global4 relay: derivative input0 to output8 after two layers |0.06666666666666667 |

All deliberate violations exceed the frozen1e-6 detection requirement. The uniform-score fixtures use specified value impulses, so weak random preferences cannot conceal the omitted edges. They demonstrate possible influence and a concrete leakage bug; they do not identify a trained mechanism.

## Pair counts

Per-head, per-example counts. Independent key enumeration, global/nonglobal row partition and dense visibility sums agree in all64 cases. Examples below use radius2 and global position at the middle.

| Length | Policy | Local only | Local + middle global | Full |
|---|---|---:|---:|---:|
| 9 | Bidirectional | 39 | 47 | 81 |
| 9 | Causal | 24 | 28 | 45 |
| 128 | Bidirectional | 634 | 880 | 16,384 |
| 128 | Causal | 381 | 504 | 8,256 |
| 1024 | Bidirectional | 5,114 | 7,152 | 1,048,576 |
| 1024 | Causal | 3,069 | 4,088 | 524,800 |

Adding global edges increases the count relative to a local window. Their value is shorter information paths. Counts establish neither lower measured memory nor faster execution. The reference computes dense score matrices.

## Conclusion and limits

Global tokens can connect distant positions within two layers in the bidirectional pattern. In the causal pattern, relay depends on where the global position sits. Correct causal masking must constrain global queries as well as all other connections.

No failed numerical cases or changed thresholds occurred. Padding, dropout, global-specific projections, learned information retention, GPU execution and training remain outside scope. Paths establish possible access, not useful content preservation. Study closed without automatic training or extension.

[Raw cases and provenance](../../../results/global_attention.json) | [Code](../../../experiments/global_attention.py)
