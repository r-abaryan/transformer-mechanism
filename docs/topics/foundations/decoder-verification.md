# Decoder verification results

Executed on 2026-09-29 with CPython 3.14.7, PyTorch 2.14.0+cpu, CPU float64. [Code](../../../experiments/verify_decoder.py) and [raw report](../../../results/decoder_verification.json) include seeds, tolerances and source hashes. The [paper audit](../../reference/foundation-audit.md) explains the architecture choice and its limits.

## What was compared

1. The fixed-weight standard-library walkthrough versus PyTorch LayerNorm and scaled-dot-product attention.
2. Explicit pre-norm block equations versus `TransformerEncoderLayer`, with matching weights, causal visibility, padding and zero dropout. Three seeds; batch 2, length 3, width 4, two heads, FFN width 7.
3. Input and all block-parameter gradients from both paths. Numerical central differences additionally checked every input coordinate via `gradcheck` and every block-parameter coordinate directly, using step 1e-6.
4. A trainable one-block decoder (vocabulary 7, width 8, two heads, FFN width 12) with embeddings, sinusoidal positions, final normalization and vocabulary projection. Separate Q/K/V parameters replace the walkthrough's tied teaching projections.

| Check | Observed result |
|---|---|
| Walkthrough logits versus tensor implementation | Maximum difference 4.441e-16 |
| Block outputs, three seeds | Maximum difference 4.441e-16 |
| Autograd input/parameter comparisons | Maximum difference 1.777e-15 |
| Numerical input derivatives | All three gradchecks passed |
| Numerical block-parameter derivatives | Maximum absolute error 2.254e-9 |
| Decoder causal isolation | Earlier-logit difference 0 |
| Padded key isolation | Valid-query logit difference 0 |
| Remove padding mask, change hidden token | Logit difference 0.16624, demonstrating a sensitive control |
| One SGD step, learning rate 0.001 | Loss decreased from 1.924997 to 1.923763 |

The explicit and library paths share the PyTorch engine, so they are not wholly independent implementations. Central differences add evidence that does not use automatic differentiation. ReLU is not differentiable at zero; the passing numerical checks concern the sampled points, not all possible parameters.

## Reproduce

From the project root, with Python 3.14.7 and uv installed:

```powershell
python -m venv --without-pip .venv
uv pip install --python .venv/Scripts/python.exe --cache-dir .uv-cache -r requirements-cpu.txt
.venv/Scripts/python.exe experiments/verify_decoder.py
```

Use the interpreter's full path if `python` is a Windows Store alias. The initial venv command in this session failed during pip bootstrapping, but created a usable isolated interpreter; uv successfully installed into that environment without pip. The requirements file records all ten installed packages. These are CPU wheels; CUDA was not installed or tested. PyTorch emitted an optional missing-NumPy warning; these checks do not use NumPy and completed with exit code 0.

## What remains unverified

No multi-layer model, low-precision arithmetic, GPU kernel, dropout, all-masked sequence, or training benchmark was tested. Padding hides keys, not query outputs; future training must exclude padding from loss and prohibit all-masked rows. Finite differences covered the block, not every embedding and vocabulary parameter of the complete decoder. The whole decoder had finite gradients and an optimizer smoke check, not a convergence test.

Subsequent work: [controlled first-token training](../retrieval/first-token-results.md) is now complete. The verification above remains a separate implementation check, not evidence of architectural superiority.
