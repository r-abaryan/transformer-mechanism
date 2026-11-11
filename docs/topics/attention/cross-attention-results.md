# Cross-attention correctness results

Executed2026-10-02. All36 final cases pass. This mathematical study is closed; no training study is selected.

[Short guide and diagram](cross-attention-guide.md) · [Protocol](../../protocols/cross-attention.md) · [Raw cases](../../../results/cross_attention.json) · [Executable checks](../../../experiments/cross_attention.py)

## Method

CPU float64, PyTorch2.11.0+cu128, batch2, width8, two heads, supplied encoder memory width6, dropout0. Seeds2309/2311/2333 × three target/source length pairs × biases on/off × source padding on/off =36 cases. These are numerical fixtures, not independent training runs.

PyTorch MultiheadAttention is compared with explicit per-example/per-head projections, stable softmax and weighted sums. A random scalar output probe checks all input/parameter gradients. A separate central finite difference checks their joint directional derivative, epsilon1e-6. A two-attention residual assembly checks target causality; encoder, FFN, normalization and positional encoding are omitted.

## Arithmetic and invariants

Each row states a bounded property and the observed worst error. A small error supports agreement within the declared tolerance.

| Check | Cases | Maximum error | Limit |
|---|---:|---:|---:|
| Output versus independent equations |36 |3.331e-16 |1e-10 |
| Per-head attention weights |36 |2.220e-16 |1e-10 |
| All input/parameter gradients |36 |5.551e-17 |1e-10 |
| Joint directional finite difference |36 |3.400e-11 |1e-7 |
| Permute supplied memory rows and mask together |36 |4.441e-16 |1e-10 |
| Change masked source content |18 |0 |1e-10 |
| Gradient on masked source rows |18 |0 |1e-10 |
| Change future target inputs, compare earlier outputs |36 |0 |1e-10 |

The permutation result applies to supplied encoded memory. Permuting raw source tokens before an encoder can change those representations and is not covered.

| Seed | Cases | Output error | Gradient error | Directional error |
|---|---:|---:|---:|---:|
|2309 |12 |2.220e-16 |5.551e-17 |3.400e-11 |
|2311 |12 |2.220e-16 |2.082e-17 |1.707e-11 |
|2333 |12 |3.331e-16 |2.776e-17 |1.797e-11 |

## Deliberate errors

A check must detect a known violation. All numerical violations below exceed the fixed1e-6 detection threshold. All-masked input and wrong shifting are rejected by explicit contract checks.

| Injected error | Detections | Smallest output change |
|---|---:|---:|
| Ignore source padding, perturb invalid content in both signs |18/18 |37.821 |
| Apply a triangular cross-attention mask |36/36 |0.422 |
| Remove target self-attention's causal mask |36/36 |3.008 |
| Supply an entirely masked source example |36/36 |Rejected before attention |
| Feed unshifted answers as target inputs |36/36 |Rejected by index check |

| Target/source length | Cases | Future-target prefix error | Minimum wrong cross-mask effect |
|---|---:|---:|---:|
|2/5 |12 |0 |0.667 |
|5/3 |12 |0 |0.422 |
|4/4 |12 |0 |0.601 |

## Recorded fixture correction

The first attempt stopped after13 completed cases. For seed2311, lengths2/5, no bias and padding present, a one-sided large perturbation nearly suppressed the invalid key's attention weight. Ignoring its mask changed output by only1.063e-9, so the negative control failed. Correct padding output and gradient errors were both zero.

The revised fixture tries both signs of the same perturbation. This exposes the missing mask without changing any tolerance or replacing configurations. All36 cases were rerun. [Initial report](../../../results/cross_attention_initial_attempt.json), [original source](../../../results/cross_attention_initial_source.py) and [execution issue](../../../results/cross_attention_execution_issue.json) are preserved. This was a weak error-detection fixture, not evidence that cross-attention was incorrect.

## What this establishes

**Observed:** independent equations and native attention agree in the tested cases. Padding is inert at the memory interface; the causal assembly preserves earlier outputs; all deliberate errors are detected.

**Limit:** this does not validate a complete encoder-decoder, raw-source padding through an encoder, target padding/loss masking, GPU kernels, reduced precision, caching or learned sequence generation. No trained-model accuracy or speed claim follows.

The distinction between decoder causal self-attention and encoder-decoder attention matches [Vaswani et al., sections3.1-3.2.3](https://arxiv.org/html/1706.03762v7). Boolean mask semantics match the [installed-version PyTorch reference](https://docs.pytorch.org/docs/2.11/generated/torch.nn.MultiheadAttention.html). We test these conventions; we do not reproduce the paper's translation benchmark.
