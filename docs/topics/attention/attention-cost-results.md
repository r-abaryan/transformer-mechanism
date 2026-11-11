# Attention cost: dense versus tiled online computation

**The same attention function can use different workspace.** Outputs agree within 1e-15. Our tiled CPU implementation is faster in this fixed matrix and has lower whole-process peaks at the larger lengths. Neither result establishes FlashAttention performance or linear-time attention.

## What we tested

Earlier [S06/S08 probes](../foundations/test-clarifications.md) checked streaming-softmax arithmetic and dense element counts. This adds actual resource measurements, without repeating a training study.

One causal head, batch1, query/key/value width64, CPU float64, one PyTorch thread, no gradients or dropout. Lengths256/512/1024/2048; two input seeds. Each method runs in a fresh process with identical Q/K/V. One warmup and three timed forwards per process: sixteen processes, 48 timings. Method order reverses for the second seed. Tile size128 is fixed. [Protocol](../../attention-cost-protocol.md).

## Math: what does tiling preserve?

Dense attention computes `S=QK^T/sqrt(d)`, then `O=softmax(S+mask)V`. Materializing S requires n*n numbers per head. At float64 that is `8*n^2` bytes for one matrix, excluding probabilities, masks, inputs and outputs. Doubling n quadruples this storage.

For each query row, online softmax keeps maximum m, denominator l and weighted numerator u. For a new score block s and value block V:

```
m_new = max(m, max(s))
a = exp(m - m_new)
p = exp(s - m_new)
l_new = a*l + sum(p)
u_new = a*u + p*V
output = u/l
```

Start with m=-infinity and l=u=0. Rescaling preserves the relative weights when the maximum changes. Each query has at least one visible key; fully future tiles are skipped. The algebra gives the same softmax-weighted value in real arithmetic, with floating-point rounding differences in execution.

Query/key tiles bound the score workspace by tile size. For fixed tile sizes, total forward storage scales as O(n*d) for inputs/output plus bounded tile and accumulator workspace. Arithmetic remains O(n^2*d). This inference-only implementation does not establish backward-pass memory savings.

## Runtime measurements

Median and range of six timed forwards per cell, pooled across two processes. These are descriptive measurements, not confidence intervals.

| Tokens | Dense ms, median [range] | Tiled ms, median [range] |
|---|---:|---:|
| 256 | 1.416 [1.308, 1.456] | .721 [.711, .758] |
| 512 | 5.749 [5.538, 6.409] | 2.365 [2.317, 2.474] |
| 1,024 | 22.681 [22.366, 24.025] | 8.763 [8.518, 9.815] |
| 2,048 | 92.596 [89.031, 98.298] | 33.857 [32.165, 34.885] |

Tiled code skips fully masked future tiles; dense code computes the full square before masking. Timings therefore reflect both blocked execution and avoided future work, not a pure memory-layout comparison. Neither implementation is a fused production kernel. Do not generalize this ranking to other widths, devices, dtypes, batch sizes or optimized libraries.

## Memory: formula versus operating-system measurement

MiB means 2^20 bytes. Score-buffer columns are analytical sizes of **one** matrix/tile, not total allocation. Peak columns are the observed range over two fresh processes.

| Tokens | Dense score MiB | Tiled score MiB | Dense process peak MiB | Tiled process peak MiB |
|---|---:|---:|---:|---:|
| 256 | .5 | .125 | 281.20-281.67 | 280.75-281.15 |
| 512 | 2 | .125 | 285.38-285.50 | 283.07-283.96 |
| 1,024 | 8 | .125 | 299.17-300.66 | 292.18-292.29 |
| 2,048 | 32 | .125 | 358.88-360.59 | 304.70-309.22 |

We read Windows peak working set before warmup and after timed execution. It measures resident process memory, including runtime and libraries, not exact tensor allocation. See [Microsoft's memory-counter documentation](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters).

Input hashing uses temporary Python/JSON allocations before the baseline. Their earlier peak can exceed later attention memory. Tiled peak growth is zero at lengths512/1024/2048 for this reason: **zero growth does not mean zero allocation**. At2048, dense growth above its own setup peak is52.15-54.83 MiB. Full before/after counters are saved. These measurements cannot independently establish an asymptotic memory law; the implementation's tensor shapes support the workspace analysis.

## Paper comparison and verification

[Dao et al., FlashAttention (2022)](https://arxiv.org/abs/2205.14135) use IO-aware tiling for exact attention on GPUs. Our Python/PyTorch CPU loops illustrate the online computation principle. They do not reproduce its GPU kernels, IO analysis or reported speedups. The source abstract was consulted; this is a focused foundational comparison, not a latest-kernel survey.

All eight paired output comparisons pass; maximum discrepancy9.99e-16. An additional length129 case checks incomplete tiles. The audit reuses the earlier scalar streaming implementation on first, boundary and last rows: maximum discrepancy5.83e-16. Input/output hashes, finite outputs, all sixteen matrix cells and all48 timing records pass. Resource measurements themselves are observations, not reproducible constants.

[Benchmark](../../../experiments/attention_cost.py), [raw environment/timings/counters](../../../results/attention_cost.json), [audit code](../../../experiments/audit_attention_cost.py), [audit and numerical summaries](../../../results/attention_cost_audit.json).

**Closed:** exact tiled attention avoids materializing the full score matrix; it does not remove quadratic arithmetic. No tile tuning, new architecture, GPU performance claim or further benchmark expansion follows from this study.
