# Intact models: retrieval and downstream differences grow together

2026-10-08. **The weak seed develops a sharp short-source marker read in one head, while its long-source read stays mixed. The paired difference also grows downstream.** This localizes a concrete pattern without the broken parameter hybrids from the preceding study. It does not yet establish its cause.

[Protocol](../../protocols/copy-intact-development.md) · [Report and audit](../../../results/copy_intact_development/report.json) · [Every group and measurement](../../../results/copy_intact_development/metrics.csv)

## What was measured

Trace the intact 500/1,000 checkpoints in the same four runs, on 128 paired six/ten-symbol sources. Supply identical correct decoder prefixes and predict symbol 5. Parameters and activations are unmodified. Groups use distinct/repeated six-symbol prefixes (10/118 examples). All reported values below are means across the 128 pairs; detailed ranges and groups are in the report.

```mermaid
flowchart LR
  A[Pre-cross decoder state] --> B[Add retrieved vector]
  B --> C[Normalization 2]
  C --> D[Add FFN output]
  D --> E[Normalization 3]
  E --> F[Final classifier and EOS margin]
```

Within each short/long pair, the pre-cross decoder state is exactly identical. Source extension first enters this decoder through cross-attention. Encoder representations can already differ, so this does not isolate attention weights from their inputs or values.

## The weak seed's attention pattern

Head numbers are implementation indices, not matched semantic roles across independently trained seeds. Target means the aligned fifth source position, even when its symbol repeats elsewhere. Marker means the actual source-end position.

| Update | Source | Head 0 target mass | Head 0 marker mass | Head 0 added-suffix mass | Head 1 target mass |
|---|---|---:|---:|---:|---:|
| 500 | Short | 19.5% | 52.3% | 0% | 99.3% |
| 500 | Long | 28.6% | 2.5% | 23.7% | 99.2% |
| 1,000 | Short | 0.1% | 99.7% | 0% | ~100% |
| 1,000 | Long | 14.6% | 19.2% | 26.1% | ~100% |

At 1,000 the short-source marker read is almost exclusive; the long read remains distributed. The second head still reads the aligned content position. Thus a nearly perfect content-reading head can coexist with premature EOS. Attention mass alone does not say whether the retrieved vector promotes or suppresses EOS.

Projected values change too. Head 0's mean target-value norm grows from 2.291 to 2.877 on short sources and 2.197 to 2.748 on long sources. These are vector sizes after that head's output-projection block, before attention weighting, not measures of useful content. All head/value norms, entropy and marker measures are preserved in the report.

## Where the paired vector difference grows

Each distance is the mean per-example L2 norm of long minus short vectors. FFN refers to its branch output; final refers to the actual post-normalization hidden state. These raw distances depend on scale and are not causal contributions.

| Model | Update | Cross output distance | After norm2 | FFN branch distance | Final hidden distance | Extension margin D |
|---|---:|---:|---:|---:|---:|---:|
| Cross-only 3767 | 500 | 1.540 | 1.055 | 1.032 | 1.960 | 2.732 |
| Cross-only 3767 | 1,000 | 3.019 | 2.006 | 3.046 | 5.236 | 10.482 |
| Cross-only 3779 | 500 | 3.014 | 1.419 | 0.844 | 1.757 | 2.272 |
| Cross-only 3779 | 1,000 | 1.641 | 0.701 | 0.472 | 1.045 | 1.188 |
| Cross-only 3793 | 500 | 2.366 | 1.071 | 0.618 | 2.077 | 2.861 |
| Cross-only 3793 | 1,000 | 1.553 | 0.528 | 0.329 | 1.543 | 2.373 |
| Combined 3767 | 500 | 1.063 | 0.603 | 0.287 | 0.911 | 0.668 |
| Combined 3767 | 1,000 | 1.529 | 0.616 | 0.552 | 1.211 | 0.546 |

The weak cross-output distance roughly doubles while the final distance grows more. Yet its mean common-position encoder distance decreases from 0.270 to 0.188. Therefore, a simple claim that every earlier representation difference grows is false. The combined control also grows in cross-output distance but decreases in extension margin; distance alone does not diagnose failure.

## Connecting the final hidden state to the real decision

For each example, `D = (w_EOS - w_target) · (h_long - h_short)`. Classifier bias cancels within a pair. This identity uses the actual final hidden state and final classifier, with no intermediate prediction lens.

In the weak seed, the mean classifier-row-difference norm rises from 1.634 to 2.140. The mean cosine between that row difference and the final paired hidden difference rises from 0.829 to 0.935. Hidden difference size, classifier size and alignment all change. Their means cannot be multiplied to reconstruct mean D; the identity holds per example. Successful controls also grow their classifier norms, so growth alone is not a failure explanation.

## Interpretation and stopping boundary

The evidence now identifies a specific candidate: the first head's short/long routing mismatch, followed by changed downstream processing. The earlier parameter swap does not contradict it: attention weights depend on encoder/decoder states as well as Q/K projection parameters.

It remains unknown whether imposing a matched marker-read pattern would change the weak model's decision while preserving the rest of its computation. That would require a separately frozen activation intervention with controls. This trace alone cannot allocate the margin increase between routing, values, normalization, FFN and classifier, or explain why optimization learned them. No new intervention is launched here.

## Verification and sources

Eight paired model/stage cells, 2,048 selected predictions per implementation, zero training. Saved initialized reload, paired-input null, absent/missing groups and intentional cross-reconstruction/logit faults pass. Captured cross output reconstructs from attention and projected values; the final margin identity passes both tensor and scalar checks. Native/reference and archived snapshot predictions agree, maximum discrepancy 6.351e-14. Saved traces are reloaded for scoring; all 81 protected files remain unchanged. Shared forward helpers limit independence. Opened examples, two selected stages and supplied prefixes limit scope.

[Vaswani et al. (2017), sections 3.1–3.3](https://arxiv.org/html/1706.03762v7#S3), inspected in the preceding study, defines the attention/residual/normalization/FFN sequence. The model here is the local one-block RoPE variant. Trace measures and final-margin decomposition are local algebra and measurements, not source-backed predictions of these scores.
