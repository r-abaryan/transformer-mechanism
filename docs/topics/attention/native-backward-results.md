# Native attention: forward and backward

Both native Windows backends pass all tested output and gradient checks. Twelve cells, 120 timing samples, no runtime or numerical failures. At 4,096 tokens, cuDNN has the lowest forward-plus-backward median in this run. Efficient attention uses less tracked memory. No WSL or installation change.

## What backward means

Forward computes attention's output O. Backward asks how changing Q, K or V would change a scalar objective L. A supplied upstream gradient G represents the derivative arriving from later layers. We use `L = sum(G * O)` and return its Q/K/V derivatives. This exercises attention's training operation, but does not train a model or update parameters. [PyTorch's autograd.grad](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html) computes this vector-Jacobian product.

Let `S = QK^T/8 + M`, `A = softmax(S)`, `O = AV`. M is the causal mask. With `dO = G`:

| Equation | Meaning |
|---|---|
| `dV = A^T G` | Send the output gradient back to the values, weighted by attention. |
| `dA = G V^T` | Measure how each attention weight affects the objective. |
| `dS = A * (dA - row_sum(A*dA))` | Differentiate softmax. The row sum broadcasts across keys; `*` is elementwise. |
| `dQ = dS K/8`, `dK = dS^T Q/8` | Send the score gradient back through the scaled dot product. |

Masked weights are zero, so masked scores contribute no gradient. These are derivatives of real-valued attention. FP16 implementations approximate them; passing a tolerance is not exact equality or a general stability proof.

## Method

[Protocol frozen before execution](../../native-backward-protocol.md). RTX 4060 Laptop GPU, Windows, torch 2.11.0+cu128. One head, batch 1, width 64, FP16 inputs/outputs and Gaussian upstream gradient. Manual dense computes internally in FP32; TF32 disabled. Q/K/V seed 919; independent upstream seed 929. Force each native backend without fallback. Reuse the earlier forward functions.

Each cell uses five warmups and ten fresh forward-plus-backward graphs. No gradient accumulation, optimizer, dropout or loss scaling. Timings exclude input transfer, CPU reference and saved-file work. CUDA events and synchronized wall times are retained. The first sample follows release of unused cached allocations.

## Results

Median CUDA-event milliseconds for forward plus backward. Additional allocation is peak tracked PyTorch allocation minus the post-warmup baseline.

| Tokens | Dense ms | Efficient ms | cuDNN ms | Extra allocated MiB: dense / efficient / cuDNN |
|---|---:|---:|---:|---|
| 512 | 1.308 | 0.578 | 0.663 | 4.688 / 0.442 / 0.568 |
| 1,024 | 1.021 | 0.912 | 0.645 | 17.875 / 0.883 / 1.134 |
| 2,048 | 1.479 | 0.783 | 0.751 | 69.750 / 1.766 / 2.267 |
| 4,096 | 8.079 | 1.255 | 0.884 | 275.500 / 3.532 / 4.533 |

At 4,096 tokens, dense / efficient / cuDNN wall medians are 8.206 / 1.313 / .924 ms. Event medians imply 6.44x and 9.14x speedups over this dense implementation. Event ranges are 8.045-13.630 / 1.079-2.141 / .833-1.886 ms; this is one session, not a confidence interval. Total tracked allocated peaks are 293.750 / 21.782 / 22.783 MiB, and reserved peaks are 348 / 26 / 26 MiB. External library allocations and CUDA context are not captured by these counters.

The earlier [forward-only test](native-attention-results.md) favored efficient attention at 4,096 tokens. This run favors cuDNN for the combined operation, while efficient still uses less tracked allocation. A universal backend winner is not established. Do not subtract the earlier timings to estimate backward-only cost: these are separate sessions and execution modes.

## Independent correctness audit

CPU float64 autograd supplies the initial reference. A separate CPU audit reconstructs output and the equations above one causal row at a time, without autograd. Its maximum disagreement with the reference is 1.34e-15 (rounded upward). It also checks inputs, source/protocol/tensor hashes, samples and memory arithmetic.

All twelve outputs pass the .005 maximum absolute-error threshold; observed maximum is .001006543. All 36 Q/K/V gradient arrays pass both thresholds: maximum absolute error <= .02 and relative L2 error <= .005. These are predeclared engineering tolerances, not paper-derived guarantees. Relative L2 error is `norm(actual-reference)/max(norm(reference),1e-12)`.

| Derivative | Worst absolute error | Worst relative L2 error |
|---|---:|---:|
| Q | 0.00105620 | 0.00038750 |
| K | 0.00165279 | 0.00037805 |
| V | 0.00123231 | 0.00028700 |

## Paper connection and stopping point

[FlashAttention](https://arxiv.org/abs/2205.14135) motivates exact attention implementations that reduce memory traffic. Our results support the practical value of optimized attention execution on this laptop, including backward. They do not reproduce that paper's specific kernel or training speedups; the named FLASH_ATTENTION backend remains unavailable. Backend identities follow [PyTorch's API](https://docs.pytorch.org/docs/2.11/generated/torch.nn.attention.SDPBackend.html).

The bounded execution question is closed: native Windows supports both tested alternatives for forward and Q/K/V backward. One shape family and one upstream vector per length do not establish every Jacobian entry, arbitrary gradient scales, reproducibility across runs, or whole-model convergence. No full-model speedup or architecture improvement is claimed. Further benchmark sweeps are not needed for this decision.

## Evidence and reproduction

[Runner](../../../experiments/native_backward.py), [raw report](../../../results/native_backward.json), [saved inputs/outputs/gradients](../../../results/native_backward_tensors.pt), [independent auditor](../../../experiments/audit_native_backward.py), [audit and complete summary](../../../results/native_backward_audit.json).

Run `.venv-gpu/Scripts/python.exe experiments/native_backward.py` with new output paths for a repeat; existing evidence is protected. Audit with `.venv/Scripts/python.exe experiments/audit_native_backward.py`.
