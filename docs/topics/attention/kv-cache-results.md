# KV caching: verified equivalence

**All 120 cases and the independent audit pass.** Cached decoding reproduces the full causal forward within the fixed `1e-10` tolerance. No prediction changes were observed. Both deliberate errors were detected in every case.

## What caching changes

A layer keeps earlier keys and values. It computes projections and FFN outputs only for the new tokens. It still compares each new query with the visible prefix. The weights and inference equations stay fixed.

The [educational decision](../../decisions/0002-kv-cache-equivalence.md) contains the equations, causal argument and diagram. Those pre-execution notes and the [protocol](../../kv-cache-protocol.md) remain frozen for provenance.

## Method and scope

All ten final query-exposure models, with their original weights. CPU float64, evaluation mode, one thread, dropout zero, no training. Test rows 0, 1, 36 and 37 from each 3/4/6-pair split were reused. They are four context variants from two families per length, not fresh performance evidence.

Offsets 4 and 17; two schedules: individual tokens, or half-prefix prefill followed by individual tokens. All-token logits were compared, including the learned biases, both layers, residuals, FFNs and final normalization/readout. This gives 120 cases, 480 sequence executions and 5,120 token decisions. These share models and fixtures; they are not independent statistical trials.

## Numerical findings

| Pairs / input tokens | Schedule | Cases | Largest logit difference | Final K/V payload, batch of four |
|---|---|---:|---:|---:|
| 3 / 8 | single | 20 | 2.00e-14 | 16 KiB |
| 3 / 8 | prefill | 20 | 2.00e-14 | 16 KiB |
| 4 / 10 | single | 20 | 2.26e-14 | 20 KiB |
| 4 / 10 | prefill | 20 | 2.26e-14 | 20 KiB |
| 6 / 14 | single | 20 | 4.75e-14 | 28 KiB |
| 6 / 14 | prefill | 20 | 4.75e-14 | 28 KiB |

All **5,120 argmax decisions match**. There are no reference top-two gaps <=1e-10; the smallest gap is 0.003339. Full and cached execution answer all 480 selected final queries correctly. This small diagnostic does not replace the broader accuracy results.

## Error controls

| Deliberate error | Detected cases (>1e-6) | Largest logit discrepancy | Changed token predictions |
|---|---:|---:|---:|
| Reset new-token RoPE positions | 120/120 | 17.86020 | 1576/5,120 |
| Hide oldest past entry in each layer | 120/120 | 18.84103 | 680/5,120 |

Resetting positions gives new Q/K the wrong relative rotations. Hiding a stored entry changes the available information and softmax normalization. The controls show that this validation detects these specific errors. Their prediction changes do not establish a deployment failure rate or identify which individual token information was essential.

## Why the math supports the result

Earlier causal states cannot depend on future tokens. Layer normalization and the FFN operate per token. With unchanged weights and prefix positions, earlier K/V therefore remain valid at each layer. This induction explains exact-arithmetic equivalence; the numerical experiment checks our implementation.

For a chunk starting after P cached tokens, new query i must see key indices `j <= P+i`. A mask that restarts at zero can incorrectly hide the prefix. Each rotated key is stored once at its original position.

Payload is `B L h T (d_k+d_v) s` bytes. Here `B=4`, `L=2`, `h=2`, `d_k=d_v=8`, `s=8`, so payload is **2,048T bytes**. Every recorded chunk length and payload agrees. This measures K/V tensors only, excluding model weights, allocator overhead, temporary concatenation and attention tensors.

## Independent audit and evidence

The auditor reproduced all four output variants in every case using separate Q/K/V projections, explicit rotation matrices, explicit normalization arithmetic and SDPA. Largest discrepancy from saved logits: `1.19e-13`. Final cached K/V agree with both the independent chunked execution and full forward; largest discrepancy: `6.22e-15`. Fixture selection, case coverage, metric values and hashes also pass.

Evidence: [raw case results](../../../results/kv_cache_equivalence.json), [saved logits and caches](../../../results/kv_cache_equivalence_tensors.pt), [audit](../../../results/kv_cache_equivalence_audit.json). Code: [experiment](../../../experiments/kv_cache_equivalence.py), [independent auditor](../../../experiments/audit_kv_cache_equivalence.py).

Recheck saved evidence:

```powershell
.venv/Scripts/python.exe experiments/audit_kv_cache_equivalence.py
```

## Interpretation and stopping decision

The result agrees with the causal-reuse description in [Hugging Face documentation](https://huggingface.co/docs/transformers/main/cache_explanation). [Shazeer (2019)](https://arxiv.org/abs/1911.02150) motivates incremental-decoding bandwidth and shared-KV alternatives; its speed/quality benchmarks were not reproduced. The [source register](../../reference/sources.md) records reading depth.

**Close the equivalence question for these conditions.** Caching is an execution improvement opportunity. This study measured correctness and tensor payload, not speed, convergence or an architecture-quality gain. Actual generated sequences, GPU/lower precision, padding, changed prefixes, quantization and cache eviction remain outside scope. Further study needs a new bounded decision.
