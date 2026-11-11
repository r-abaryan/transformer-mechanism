# Cached MHA/MQA decoding: frozen GPU protocol

Question: does accepted MQA reduce measured full-decoder latency on this laptop under one explicit implementation? Use seed 1411's final MHA/MQA checkpoints, fixed before timing. No new training, checkpoint selection or task-quality claim.

RTX 4060 Laptop GPU, native Windows, existing torch 2.11.0+cu128 environment. CPU thread count 1, inference mode, float32 weights/activations/cache, TF32 disabled. Two layers, model width 16, two query heads of width 8. MHA stores two K/V heads; MQA stores one. Single-token attention uses explicit dot products, softmax and a broadcast view for shared K/V. No automatic backend, compilation, fused GQA kernel or WSL.

## Fixed cases

- Batch B in {1,16}; prefix P in {14,128,512}; eight predetermined continuation tokens, identical between arms.
- CPU generator seed `20270109 + P`, uniform vocabulary IDs 0-18, shape [B,P+8]. These are execution fixtures, not recall tests. Longer inputs are outside the training distribution.
- Both layers use separately preallocated compact K/V of capacity P+8. GPU prefill happens outside timing. Each sample overwrites eight continuation slots using their original positions (offset 4); the prefix stays intact.
- Precompute RoPE trigonometric tables outside timing. Include token lookup, Q/K/V, cache writes, attention, residuals, FFN, norms, readout and gathering eight output tensors. Exclude prefill, model loading, sampling, input transfer and serialization.
- Three sequential fresh worker processes. Reverse method order in worker 1; workers 0/2 use MHA then MQA. Alternate batch/prefix traversal direction by worker. No concurrency on the GPU.
- Three warmup sequences and ten timed sequences per cell: 36 cells, 360 repeated timing samples. CUDA events measure the eight-token stream span; synchronized wall time includes host dispatch. Retain raw samples and process-level medians. Repeated samples are not independent hardware experiments.

## Correctness, payload and memory

Before timing each cell, compare cached GPU continuation logits with full causal CPU float64 computation using the same weights first rounded to float32. Require finite outputs, absolute error <=1e-4 and relative L2 error <=1e-4. Disclose argmax differences; this checks numerical execution, not generated-text quality. A failure is retained, and that cell is not timed.

Measure actual compact cache storage and shapes. Expected reserved payload is `B L G(P+8)(d_k+d_v)*4` bytes, exactly half for MQA. Record post-warmup allocated/reserved baselines and timed peaks. Extra allocation is peak allocated minus baseline; external CUDA/driver allocations and allocator overhead are not K/V payload. Read memory endpoints before any CPU output transfer. Preserve failures without replacing cases or kernels.

Save all weights/data/source/protocol hashes, GPU outputs, CPU references, raw timings and nvidia-smi snapshots before/after each process. One independent CPU audit recomputes continuation logits with explicit normalization, rotation matrices and SDPA, checks every size/metric/hash and recomputes timing summaries. Independent-reference tolerance <=1e-10; GPU tolerances remain as above.

Claim a repeatable >=10% latency improvement for a shape only if all three process medians satisfy MQA/MHA <=.9 for both wall and event metrics, and both arms' correctness checks pass. Ratios within this rule can still vary with power, thermals and launch overhead; no statistical population claim. Report every shape even if none passes. Stop after these three workers and audit. No new sizes, precision changes, kernel tuning or automatic follow-up.

Sources: [PyTorch CUDA timing/memory semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html), [Shazeer 2019](https://arxiv.org/html/1911.02150v1), and [our trained MQA result](mqa-training-results.md). Paper hardware/speed claims are not replicated.
