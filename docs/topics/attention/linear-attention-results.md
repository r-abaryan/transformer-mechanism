# Linear attention: arithmetic verified, softmax equivalence rejected

Executed2026-10-03, closed. No training. Read the [short guide](linear-attention-guide.md) and [frozen protocol](../../protocols/linear-attention.md).

## Methods and results

Three seeds, lengths3/7/17, feature widths3/5, value widths2/4, with/without invalid keys:72 cases. Batch2, heads2, CPUfloat64, one thread. Compare explicit pairwise weights with prefix sums, serial recurrence and two chunks carrying state. All queries are evaluated; masking applies to keys.

| Check | Largest error / result | What it establishes |
|---|---:|---|
| Outputs | 8.89e-16 | Four formulations agree in tested cases |
| Q/K/V gradients | 2.78e-17 | Backward arithmetic agrees |
| Final S/z state | 0 | Chunk boundaries preserve accumulated state |
| Directional finite difference | 3.03e-11 | Joint derivative agrees with numerical perturbation |
| Future perturbation | 0 | Tested earlier outputs remain causal |
| Invalid K/V gradients | 0 | Invalid keys/values contribute nothing |

Thresholds were1e-10 for arithmetic and1e-7 for finite differences, unchanged. Independent scalar equations additionally verify the collision witness and six resource counts. This is not an independent reimplementation of every random gradient check.

## Controls that must fail

| Deliberate mistake | Observed difference |
|---|---:|
| Remove denominator | 5 |
| Ignore key mask | 49 |
| Reset state at chunk boundary | 3 |
| Detach carried state | Old-value gradient changes by0.5; outputs stay identical |
| Read before adding current value, scalar fixture | 3 |
| Empty valid prefix / severe negative-feature underflow | Both rejected |

Matching outputs alone can miss a broken training graph. Detachment demonstrates that directly.

## A concrete limitation

With keys(0,1), value histories(0,1) and(2,0) both give S=2,z=3. Every query therefore reads2/3 in this one-dimensional feature map. Softmax at q=1 distinguishes them:0.731059 versus0.537883. This is a counterexample to lossless history retention for this state, not proof that all linear-attention models fail retrieval.

The main report's illustrative `query_reads` field accidentally used default float32 scalar tensors. The72-case matrix remained float64. A separate preserved [audit](../../../results/linear_attention_audit.json) verifies both histories using double-precision scalar equations; maximum deviation from2/3 is below1e-15. No thresholds or original artifacts were changed.

## State size is not a speed benchmark

| Feature/value widths | Length | Recurrent scalars | Stored KV scalars |
|---|---:|---:|---:|
| 4/3 | 16 | 16 | 112 |
| 4/3 | 1024 | 16 | 7168 |
| 64/64 | 16 | 4160 | 2048 |
| 64/64 | 1024 | 4160 | 131072 |

Multiply by8 for float64 payload bytes. These counts exclude other activations and allocator overhead. At short lengths the matrix state can exceed the KV payload. Prefix sums and naive autograd retain sequence-sized intermediates.

## Paper comparison and stopping point

[Katharopoulos et al.](https://proceedings.mlr.press/v119/katharopoulos20a.html), sections3.2-3.4, supplies the kernel and recurrent construction. Our results support that arithmetic within the declared cases. They do not validate language quality, practical GPU savings, numerical behavior in low precision or every later linear-attention variant.

This mathematical study is closed. No trained comparison or architectural replacement is selected. [Sparse-attention findings](../../assessments/sparse-attention-assessment.md) remain separate: visibility restrictions and a changed similarity function are different design choices.

Evidence: [implementation](../../../experiments/linear_attention.py), [raw report](../../../results/linear_attention.json), [scalar audit](../../../experiments/audit_linear_attention.py).
