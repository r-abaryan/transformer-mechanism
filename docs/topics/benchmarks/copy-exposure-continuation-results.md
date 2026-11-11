# Broader exposure improves an existing weak solution

**Adding self-attention positional exposure improves length-10 copying from 55/128 to 99/128 compared with matched continued training.** Short copying stays perfect. The complete-repair criterion still fails: 29 premature stops remain.

## Question and method

Earlier exposure comparisons started from initialization. This test starts both branches from the *same already-trained failure*: seed 3767, update 6,000, with identical weights, Adam state and random generators.

Both branches receive another 2,000 updates, identical examples and sampled cross-attention positions. One retains cross-only exposure. The other also applies those positions to encoder and decoder self-attention. All content and EOS targets remain supervised. We evaluate fixed checkpoints every 500 updates; only update 8,000 determines the frozen decision. There is no best-checkpoint selection.

Ordinary evaluation uses contiguous positions, full self-generated sequences and the original stopping rule. The 14 reused pools cover every short train/validation/test group and long tests at lengths 8/10. These are opened diagnostic data, not fresh confirmation. [Frozen protocol](../../protocols/copy-exposure-continuation.md).

## What positional exposure changes

Token order and answers stay the same. During training, a short sequence may receive spaced position IDs such as `0, 3, 7, 10`. In RoPE, position changes the rotations used in attention scores:

\[
s_{ij}=\frac{(R(p_i)q_i)^\top(R(p_j)k_j)}{\sqrt{d_k}}.
\]

The intervention exposes self-attention to different position spacings while learning the same task. This changes the training computation and gradients, not the architecture or target labels. It does not tell us which resulting weight changes explain the improvement.

## Results

Complete generated copies, including correct EOS. Each aggregate long pool has 128 sequences.

| Update | Cross-only: length 8 | Added self exposure: length 8 | Cross-only: length 10 | Added self exposure: length 10 |
|---|---:|---:|---:|---:|
| 6,000, shared start | 119 | 119 | 0 | 0 |
| 6,500 | 86 | 128 | 10 | 89 |
| 7,000 | 88 | 128 | 19 | 87 |
| 7,500 | 85 | 128 | 57 | 91 |
| 8,000, fixed endpoint | 92 | 128 | 55 | 99 |

Final length-8 distinct/repeated groups: control **45/64 and 47/64**; intervention **64/64 and 64/64**. Length 10 has repeated symbols only; its distinct group is absent, not a zero score.

At length 10, the intervention corrects 44 control failures and harms none of the control's correct copies, a **34.38 percentage-point** paired gain. Both branches improve over the shared start, so extra training itself contributes; its effect cannot be credited to exposure.

| Final measure | Continued cross-only | Added self exposure |
|---|---:|---:|
| Length-10 complete copies | 55/128 | 99/128 |
| Length-10 premature EOS | 73/128 | 29/128 |
| Length-10 teacher-forced CE | 0.22272 | 0.04934 |
| Length-8 correct content | 128/128 | 128/128 |
| Length-8 missing EOS | 36/128 | 0/128 |

For length 10, correct content and on-time stopping counts equal complete-copy counts; remaining failures stop early. No late or missing EOS occurs there. At length 8, the control copies all content but sometimes never stops. Length transfer is therefore not summarized adequately by content alone.

Every distinct/repeated short group remains at **100% complete-copy accuracy in both branches at every measured checkpoint**. Final short losses, pooled by target count:

| Short split | Continued cross-only CE | Added self exposure CE |
|---|---:|---:|
| Training | 0.0001661 | 0.0001483 |
| Validation | 0.0001648 | 0.0001526 |
| Test | 0.0001836 | 0.0001463 |

All per-group losses, teacher accuracy and stopping scores are retained in the [CSV](../../../results/copy_exposure_continuation/metrics.csv) and [report](../../../results/copy_exposure_continuation/report.json).

## Interpretation and stopping decision

This paired intervention supports an exposure effect **from this saved state**: broader self-site exposure improves copying and stopping beyond the matched control without observed short-task damage. Unlike the failed cross-model transplant, each model learns its own compatible parameters.

Neither branch reaches the required 95% full copying in every long group. Thus the frozen complete-repair and repair-qualified benefit gates fail despite the measured gain. The result does not show convergence, necessity of early exposure, a unique internal mediator or reliability across seeds. Encoder and decoder self-attention exposure remain grouped. The successful from-initialization model is historical context, not a matched third arm.

Stop at update 8,000. No budget extension or exposure tuning follows automatically. The remaining explanatory question is what changed during learning to remove those premature stops; this experiment identifies the intervention's effect, not that mechanism.

## Verification, compute and sources

Actual training: 4,000 total updates, 638,000 sampled examples and 3.36 million supervised tokens. Per-step batches cycle through 210/168/140/120 examples, always 840 targets, without accumulation. CPU float64, one thread; 1,888 parameters. Training segments took 46.1 seconds excluding checkpoint evaluation. Full-step fixtures include Adam state: 15,104 parameter bytes and 30,340 optimizer bytes. Peak process working set was 625.45 MiB, including imports/evaluation; this is not isolated tensor memory. GPU allocation/reservation is inapplicable.

Saved initialized evaluators, supplied correct/wrong logits, missing groups and deliberately corrupted generation were checked before training. Both exposure paths' parameter gradients match explicit reference equations. Streams and generator states match between branches; initial outputs reproduce archived scores. The persisted reference/scalar audit passes 126 cells and 328 CSV rows, maximum error 1.434e-13. All 129 protected files remain unchanged. The reference shares frozen model helpers, so it is not a fully independent implementation.

[Code](../../../experiments/copy_exposure_continuation.py) · [audit code](../../../experiments/audit_copy_exposure_continuation.py) · [preflight](../../../results/copy_exposure_continuation/preflight.json) · [persisted audit](../../../results/copy_exposure_continuation/audit.json).

[Ruoss et al. (2023), §3](https://arxiv.org/html/2305.16843v1#S3), inspected for this study, motivates ordered position sampling and its use with RoPE. Our equation expresses the rotated-dot-product computation; site-specific continuation, controls, thresholds and scores are local work. The paper does not establish recovery from this saved failure. Tested position IDs remain within the training sampling range.
