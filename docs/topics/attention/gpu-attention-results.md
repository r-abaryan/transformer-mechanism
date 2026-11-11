# Attention on the RTX 4060 Laptop GPU

**The educational tiled loop saves tracked GPU memory but is much slower than dense attention here.** Eight dense/tiled cells pass numerical checks. Four forced FlashAttention cells fail because this Windows PyTorch build lacks that backend; no fallback was used. [Fixed protocol](../../gpu-attention-protocol.md).

## What is being compared?

The same causal attention function, one head of width64, batch1, float16 inputs/output. Lengths512/1024/2048/4096. Three implementations: explicit dense, Python-loop online tiles128, and PyTorch SDPA with its FlashAttention backend forced. An unavailable backend is recorded, not silently replaced.

The dense and tiled examples convert inputs to float32 inside each call and return float16. The fused backend uses its own accumulation strategy. This compares implementation choices, not identical arithmetic instructions. Each output must agree with CPU float64 attention on the same rounded inputs to maximum absolute error .005. This tolerance was fixed before measurement.

## Why GPU measurement differs

CUDA work is asynchronous. Timing only the Python call can measure dispatch rather than completion. We synchronize each sample and save both CUDA-event elapsed time and wall time. Event intervals can include stream idle time while Python issues later operations; they are not summed active-kernel time.

Five warmups precede ten measurements. After warmup, unused allocator cache is released outside timing, so the first measured call includes fresh allocations. Inputs already reside on the GPU; transfers, reference computation and serialization are excluded. Methods run sequentially in rotated order, without clock or power changes.

Allocated memory measures live allocations tracked by PyTorch. Reserved memory measures blocks held by its caching allocator. Peak allocated minus the measured post-warmup baseline describes incremental tracked allocation. That baseline contains inputs and persistent allocations, not inputs alone (8.31-9.63 MiB here). Neither counter includes all driver/context or external-library memory. See [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html).

## Measured GPU timings

Milliseconds, median [minimum, maximum] of ten synchronized samples. Ranges describe this run, not statistical confidence or cross-session reproducibility.

| Tokens | Dense CUDA-event ms | Tiled CUDA-event ms | Dense / tiled wall medians ms |
|---|---:|---:|---:|
| 512 | .195 [.175, .991] | 3.586 [3.424, 3.857] | .224 / 3.605 |
| 1,024 | .247 [.177, .812] | 11.836 [11.056, 12.298] | .275 / 11.871 |
| 2,048 | .629 [.585, 2.100] | 73.776 [70.995, 77.668] | .677 / 73.832 |
| 4,096 | 2.900 [2.848, 6.147] | 269.631 [265.673, 283.993] | 2.957 / 269.702 |

The code exposes many small GPU operations through Python loops. Host dispatch, stream gaps and intermediate operations can contribute to this cost; no profiler attribution was performed. The data establish the performance difference, not a unique cause. The first timed call follows cache release, so its fresh-allocation cost contributes to the range.

## Measured tracked GPU memory

MiB. Additional allocation is peak allocated minus the measured baseline. Reserved is total peak allocator reservation. These are separate metrics, not whole-board VRAM measurements.

| Tokens | Dense / tiled total allocated peaks | Dense / tiled additional peaks | Dense / tiled reserved peaks |
|---|---:|---:|---:|
| 512 | 10.813 / 9.111 | 2.500 / .799 | 24 / 22 |
| 1,024 | 17.500 / 9.799 | 9.000 / 1.299 | 24 / 22 |
| 2,048 | 42.875 / 11.298 | 34.000 / 2.423 | 56 / 24 |
| 4,096 | 141.625 / 14.298 | 132.000 / 4.673 | 154 / 28 |

The dense float32 score matrix alone costs `4*n^2` bytes: 64 MiB at4,096 tokens. Probabilities and other intermediates raise the measured peak. Tiled score buffers are bounded by128*128 elements, while inputs, casts, output and accumulators still consume memory. The earlier [online-softmax equations](attention-cost-results.md#math-what-does-tiling-preserve) describe the same mathematical operation. Quadratic arithmetic remains.

## Fused backend failure and numerical audit

Forced FLASH_ATTENTION returned `No available kernel` at all four lengths. The separate environment probe reports `is_flash_attention_available() == False`. This is a build/backend limitation, not evidence that an RTX4060 cannot run fused attention. No fused timing or memory result is available.

All eight dense/tiled outputs pass the .005 absolute-error criterion against float64 reference on the same half-rounded inputs. Maximum error is .000946583. The audit reconstructs references one causal row at a time, checks input generation, saved hashes, sample counts and memory arithmetic. Eighty timing samples are retained. No trained-model accuracy or backward-gradient claim follows.

## Scope and limitations

The [earlier CPU benchmark](attention-cost-results.md) used float64, a different PyTorch version and different implementations. Do not divide its timings by these results and call that a hardware speedup. This is forward inference only, one random input seed, one head and one laptop. It does not measure training, whole-model throughput or energy efficiency. Power, temperature, background work and Windows scheduling can affect timings.

Tiled exact attention changes storage and execution order; it still performs quadratic attention arithmetic. A Python tile loop is not equivalent in execution efficiency to a fused GPU kernel. The [SDPA documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html) describes backend selection and numerical differences. Forcing FLASH_ATTENTION prevents fallback; it is not a reproduction of a paper's training benchmark.

## Environment and execution notes

NVIDIA tools detect an RTX4060 Laptop GPU with approximately8 GiB VRAM. The existing `.venv` retains torch2.14.0+cpu. Separate `.venv-gpu` uses torch2.11.0+cu128 with CUDA12.8, CPython3.14.7 and driver617.14. CUDA availability, device identity and a tensor-operation smoke test passed. The initial venv command failed in ensurepip, but created a usable interpreter; uv installed directly into it without pip. A freeze attempt initially used an inaccessible default cache; rerunning with the established workspace cache succeeded. Package download took about16 minutes; this setup time is not benchmark time.

Reproduction dependencies: [requirements-gpu.txt](../../../requirements-gpu.txt). Run `.venv-gpu/Scripts/python.exe experiments/gpu_attention.py` only with new output paths, since completed artifacts are protected from overwriting. CPU audit: `.venv/Scripts/python.exe experiments/audit_gpu_attention.py`.

[GPU runner](../../../experiments/gpu_attention.py), [raw samples/memory/device snapshots](../../../results/gpu_attention.json), [inputs/reference/outputs](../../../results/gpu_attention_tensors.pt), [environment probe](../../../results/gpu_environment.json), [CPU-side audit](../../../experiments/audit_gpu_attention.py), [audit results](../../../results/gpu_attention_audit.json).

**Closed after all twelve attempts.** Lower workspace is established for these manual implementations; automatic GPU speedup is not. Fused performance remains unmeasured in this environment. No backend substitution, installation search or optimization loop follows automatically.

## Live verification after completion

On September 30, 2026, a targeted rerun used the saved 4,096-token inputs with the same GPU functions and TF32 disabled. Both inputs and outputs were on `cuda:0` (RTX 4060 Laptop GPU). Dense and tiled outputs exactly matched their saved outputs; each differed from the float64 reference by at most .000453904. One timed call after two warmups took 2.946 ms and 263.853 ms respectively. These are verification samples, not replacement benchmark estimates. Forced FlashAttention failed again: this PyTorch build reports that it was not compiled with FlashAttention. The independent CPU audit also passed again. [Live verification record](../../../results/gpu_attention_recheck.json).

Follow-up: [native Windows alternatives](native-attention-results.md) now measures forced efficient and cuDNN backends successfully. The original FlashAttention failure remains valid.
