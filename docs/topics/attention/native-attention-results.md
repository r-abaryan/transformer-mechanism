# Native Windows attention results

Both forced native backends work on the RTX 4060 Laptop GPU. All twelve cells pass; no WSL, package changes or fallback. This follow-up closes the practical alternative-backend question for the tested shapes.

## Method and math

Batch 1, one head, width 64, causal forward, lengths 512 through 4,096. FP16 inputs/outputs; manual dense uses FP32 internally. Each cell uses five warmups and ten synchronized samples, with rotating method order. Dense is rerun as a contemporaneous control. Backend flags are asserted inside every forced SDPA call. [Frozen protocol](../../native-attention-protocol.md).

All methods target `softmax(QK^T / sqrt(64) + M)V`, where M is zero for visible tokens and negative infinity for future tokens. Fused execution combines operations to reduce intermediate traffic. It does not remove the quadratic number of query-key interactions. Floating-point accumulation can change the last digits.

## Measurements

Median CUDA-event milliseconds. Extra allocation means peak PyTorch allocated bytes minus the post-warmup baseline. The first measured call includes cold allocator work after emptying unused cache.

| Tokens | Dense ms | Efficient ms | cuDNN ms | Extra allocated MiB: dense / efficient / cuDNN |
|---|---:|---:|---:|---|
| 512 | 0.203 | 0.077 | 0.098 | 2.500 / 0.062 / 0.063 |
| 1,024 | 0.221 | 0.090 | 0.135 | 9.000 / 0.125 / 0.126 |
| 2,048 | 0.529 | 0.142 | 0.232 | 34.000 / 0.250 / 0.251 |
| 4,096 | 2.827 | 0.289 | 0.363 | 132.000 / 0.500 / 0.501 |

At 4,096 tokens, efficient attention is 9.79x faster than the fresh dense control; cuDNN is 7.79x faster by these event medians. Synchronized wall medians are 2.850 / .304 / .376 ms respectively. Total allocated peaks are 141.625 / 10.125 / 10.126 MiB; total reserved peaks are 154 / 22 / 24 MiB. The .500 MiB efficient extra allocation equals the output tensor size here. It does not mean the operation uses only .500 MiB of GPU memory: persistent baseline allocations and external library allocations are outside that difference. All sample ranges and counters remain in the artifacts.

## Correctness and meaning

The independent CPU audit reconstructs each causal row in float64, checks generated inputs, source/protocol/tensor hashes, all 120 timing samples and memory arithmetic. All twelve outputs are finite. Maximum absolute error is .001006543, below the predetermined .005 tolerance. There are no failed cells.

This demonstrates a usable native Windows alternative, not successful execution of the unavailable FLASH_ATTENTION backend. PyTorch lists EFFICIENT_ATTENTION and CUDNN_ATTENTION separately in its [backend API](https://docs.pytorch.org/docs/2.11/generated/torch.nn.attention.SDPBackend.html). Backend selection and numerical differences are described in the [SDPA documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html).

The lower tracked allocation and shorter runtime are consistent with the IO-saving motivation of [FlashAttention](https://arxiv.org/abs/2205.14135). This is not a reproduction of that paper's kernels or reported speedups. The earlier Python tiling result showed that saving storage alone does not guarantee speed; these compiled backends show why implementation matters.

## Limits and closure

One GPU, one input seed, four shapes and forward only. Manual FP32 versus native accumulation makes this an implementation comparison, not an isolated test of fusion. CUDA events include stream idle time; no kernel profiler was used. External allocations, full-model throughput, backward performance and training quality remain unmeasured. Timing ratios apply to this run, not all transformers. No further tuning or repeated benchmark expansion is needed for this question.

[Runner](../../../experiments/native_attention.py), [raw measurements](../../../results/native_attention.json), [saved tensors](../../../results/native_attention_tensors.pt), [CPU auditor](../../../experiments/audit_native_attention.py), [audit and full summary](../../../results/native_attention_audit.json). The follow-up runner/auditor retain the frozen original measurement procedure separately so the original evidence stays reproducible; dense math is imported from the original runner.

Run `.venv-gpu/Scripts/python.exe experiments/native_attention.py` with new output paths to repeat; completed files are protected. Audit with `.venv/Scripts/python.exe experiments/audit_native_attention.py`.

Follow-up: [forward-plus-backward results](native-backward-results.md) now verify Q/K/V gradients and combined execution cost.
