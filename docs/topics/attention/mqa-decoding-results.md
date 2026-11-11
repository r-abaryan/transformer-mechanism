# MQA cached decoding on the RTX 4060

**Cache savings pass; the repeatable speedup criterion fails for every shape.** All 36 numerical checks and the independent audit pass. Smaller persistent K/V storage does not establish faster execution in this implementation.

## Method

Native Windows, RTX 4060 Laptop GPU (driver 617.14), PyTorch 2.11.0+cu128, FP32, TF32 disabled. The fixed seed-1411 trained MHA/MQA pair has width 16, two query heads and two layers. No new training or model selection.

Batch 1/16, prefixes 14/128/512, eight identical predetermined continuation IDs. Uniform vocabulary inputs are execution fixtures; longer lengths and continuations are outside the recall distribution. This tests token-by-token execution, not text generation quality.

Each layer uses a preallocated compact K/V buffer. Timing includes the complete decoder step and output collection; it excludes prefill, allocation of buffers, input transfer, sampling and serialization. Attention uses explicit CUDA dot products/softmax, with shared-KV broadcast views; no fused GQA kernel or compilation. [Frozen protocol](../../mqa-decoding-protocol.md).

Three fresh sequential processes, with reversed method/shape order in process 1. Each cell has three warmup and ten timed eight-token sequences: 36 cells, 360 timing samples. The ten samples within a cell are repeats. Three processes on one laptop session are not independent hardware experiments.

## Latency findings

Cells show synchronized wall-clock medians for an **eight-token batch sequence**, in milliseconds. Ratios are MQA/MHA, so below 1 favors MQA.

| Batch / prefix | Process 0: MHA / MQA ms | Process 1: MHA / MQA ms | Process 2: MHA / MQA ms | Wall ratios, processes 0/1/2 |
|---|---:|---:|---:|---|
| 1 / 14 | 14.501 / 14.277 | 32.804 / 29.089 | 13.008 / 13.197 | 0.985 / 0.887 / 1.015 |
| 1 / 128 | 30.832 / 31.261 | 31.273 / 31.867 | 12.713 / 30.028 | 1.014 / 1.019 / 2.362 |
| 1 / 512 | 29.918 / 30.649 | 33.606 / 32.205 | 31.377 / 32.515 | 1.024 / 0.958 / 1.036 |
| 16 / 14 | 30.154 / 31.988 | 29.997 / 31.990 | 30.682 / 32.268 | 1.061 / 1.066 / 1.052 |
| 16 / 128 | 31.723 / 32.691 | 32.266 / 32.800 | 31.798 / 31.597 | 1.031 / 1.017 / 0.994 |
| 16 / 512 | 30.006 / 31.836 | 29.651 / 13.985 | 31.631 / 30.980 | 1.061 / 0.472 / 0.979 |

The fixed rule requires at least 10% improvement in both CUDA-event and wall medians in **every process** for a shape. Zero of six shapes pass. For example, batch16/prefix512 has wall ratios 1.061, 0.472 and 0.979. The fast second-process result does not repeat. Causes of the variability were not isolated.

CUDA events measure the recorded stream span; eager Python dispatch can leave idle gaps inside it. These are not isolated kernel times. Both event/wall samples and every allocation endpoint are retained in the raw reports. No confidence interval or universal performance ranking is claimed.

## Cache and temporary allocation

Compact payload reserves all P+8 slots: `B L G(P+8)(d_k+d_v)*4` bytes. Here L=2 and d_k=d_v=8. G is 2 for MHA and 1 for MQA.

| Batch / prefix | MHA / MQA compact payload, KiB | Extra peak tracked allocation: MHA / MQA, KiB |
|---|---:|---:|
| 1 / 14 | 5.5 / 2.75 | 10.5 / 10.5 |
| 1 / 128 | 34 / 17 | 11.5 / 11.5 |
| 1 / 512 | 130 / 65 | 20.5 / 20.5 |
| 16 / 14 | 88 / 44 | 24.5 / 44.5 |
| 16 / 128 | 544 / 272 | 66.5 / 186.5 |
| 16 / 512 | 2080 / 1040 | 211 / 667 |

These extra-allocation values repeat in all three processes. Extra allocation is peak PyTorch allocation minus the post-warmup baseline. Persistent cache is already in that baseline. The delta includes temporary attention/output work; it is not total GPU memory. External CUDA/context allocations are excluded.

## Why smaller storage need not mean faster decoding

MQA reduces stored K/V and their projection count. Both query heads still compute their own scores and weighted values across T visible tokens. These attention products remain proportional to `B H T(d_k+d_v)` per step. Shrinking G does not shrink the number of query-head attention products.

A storage formula cannot determine latency. Actual kernels, tensor layouts and host dispatch must be measured. This experiment verifies the tradeoff for this small eager implementation; it does not identify the runtime bottleneck. Profiling or a fused-kernel comparison would be a separate question.

## Correctness and evidence

GPU cached logits are checked against full causal CPU float64 logits using the same weights rounded to FP32. All 36 cells satisfy both fixed `1e-4` error tolerances. Maximum absolute error is `3.85e-05`, relative L2 error `8.55e-07`. No argmax differences occur in the saved comparisons. These checks establish numerical agreement, not task accuracy.

The independent audit recomputes the CPU reference using explicit normalization, rotation matrices and SDPA; maximum reference disagreement `8.62e-14`. All fixture, checkpoint, source/protocol hashes, cache sizes, raw-sample medians and decision arithmetic pass. It cannot independently certify physical timing from saved records.

Evidence: [summary](../../../results/mqa_decoding.json), process reports [0](../../../results/mqa_decoding_worker0.json), [1](../../../results/mqa_decoding_worker1.json), [2](../../../results/mqa_decoding_worker2.json), [audit](../../../results/mqa_decoding_audit.json). Reports hash their saved input/output tensor artifacts. Code: [runner](../../../experiments/mqa_decoding.py), [auditor](../../../experiments/audit_mqa_decoding.py).

```powershell
.venv/Scripts/python.exe experiments/audit_mqa_decoding.py
```

## Sources and stopping decision

[Shazeer (2019)](https://arxiv.org/html/1911.02150v1) motivates MQA through incremental bandwidth savings. Its model sizes, execution and hardware differ from this test. [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html) supplies timing and allocator conventions. Our result does not replicate or refute the paper's reported speedups.

**Study closed.** Preserve half-size cache and the failed repeatable-speed claim together. The [trained quality/resource acceptance](../../mqa-training-results.md) remains valid under its separate rule. Other models, fused kernels, FP16, longer contexts and independent laptop sessions remain unmeasured. No automatic timing sweep follows.
