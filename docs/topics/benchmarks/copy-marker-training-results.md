# Copying after source-marker training

2026-10-04 assessment of six matched runs completed on 2026-10-03, independently audited. **This is training with the marker**, unlike the earlier [inference-only insertion](copy-source-end-marker.md). It asks whether models learn correct content and stopping in the marked format.

## What changed and why

| Earlier checkpoint diagnostic | This learning diagnostic |
|---|---|
| Models trained without a source marker | Source EOS present in every training example |
| Insert marker only during evaluation | Evaluate the same source format used during training |
| Shows whether fixed logits can distinguish length | Measures learned copying and stopping |

Encoder input is content+EOS. Decoder prefix is BOS+content; targets remain content+EOS, with **no second EOS target**. Ordinary generation starts at BOS and uses only its own predictions. The source marker supplies a deterministic boundary; it does not supply the content answers or decoder length control. At termination, it makes the required EOS available as an encoder value.

With identical content values v and distinct end value u, one attention mixture can be \(v+P_{\mathrm{END}}(u-v)\). Its output can depend on the marker weight. Training updates the source embedding and attention/readout parameters through CE; sharing the token ID alone did not teach this in the earlier diagnostic. Position-matched reads are possible, but accuracy does not prove that mechanism was learned.

## Methods

Seeds 3163,3167,3169, sinusoidal/RoPE arms, 1,888 parameters, width 8, two width 4 heads, one post-norm encoder/decoder, ReLU FFN16, dropout zero. Identical initial tensors and sampled indices within each pair. One fixed 6,000-update Adam .001 recipe; no selection or tuning. Content lengths 3/4/5/6, 840 target rows/update, 957,000 sampled sequences and 5.04M supervised targets each. Source input also has 5.04M tokens, including 957,000 markers; visible attention work is 71.82M pairs before the head factor. CPU float64, one thread, GPU unused. Training-loop total 543.04 seconds includes evaluations/saves and is not a benchmark.

Opened random pools are reused. Uniform probes are **not** added to minibatches. Exact uniform overlap with original splits: length 3: training 2/8, validation 0/8, test 0/8; length 4: training 0/8, validation 0/8, test 0/8; length 5: training 0/8, validation 0/8, test 0/8; length 6: training 0/8, validation 0/8, test 0/8; length 8: training 0/8, validation 0/8, test 0/8; length 10: training 0/8, validation 0/8, test 0/8. Uniform scores mix previously seen and unseen inputs as disclosed; they are not an independent held-out split. All final checkpoints precede longer-input scoring. [Frozen protocol](../../protocols/copy-marker-training.md).

## Results

Three-seed means on random test pools. Full exactness requires all content and immediate EOS. Oracle content excludes EOS and supplies n, retaining learned routing and its own prefix.

| Test symbols | Ordinary sinusoidal | Ordinary RoPE | Oracle sinusoidal | Oracle RoPE |
|---|---|---|---|---|
| 3 | 97.92% | 100.00% | 100.00% | 100.00% |
| 4 | 98.44% | 100.00% | 98.70% | 100.00% |
| 5 | 99.74% | 100.00% | 99.74% | 100.00% |
| 6 | 95.83% | 100.00% | 97.40% | 100.00% |
| 8 | 0.78% | 10.42% | 2.60% | 14.06% |
| 10 | 0.00% | 2.34% | 0.00% | 4.17% |

Worst distinct/repeated group within lengths 3/4/5/6, per seed. Training requires >=99%; seen-length tests >=95%.

| Seed | Training sinusoidal | Training RoPE | Seen test sinusoidal | Seen test RoPE |
|---|---|---|---|---|
| 3163 | 100.00% | 100.00% | 87.50% | 100.00% |
| 3167 | 100.00% | 100.00% | 89.06% | 100.00% |
| 3169 | 100.00% | 100.00% | 98.44% | 100.00% |

| Seed | Longer group | Ordinary sinusoidal | Ordinary RoPE | Oracle sinusoidal | Oracle RoPE |
|---|---|---|---|---|---|
| 3163 | 8 / distinct | 0.00% | 7.81% | 0.00% | 20.31% |
| 3163 | 8 / repeated | 0.00% | 4.69% | 3.12% | 9.38% |
| 3163 | 10 / repeated | 0.00% | 0.00% | 0.00% | 3.91% |
| 3167 | 8 / distinct | 0.00% | 6.25% | 0.00% | 6.25% |
| 3167 | 8 / repeated | 0.00% | 1.56% | 0.00% | 1.56% |
| 3167 | 10 / repeated | 0.00% | 1.56% | 0.00% | 2.34% |
| 3169 | 8 / distinct | 1.56% | 20.31% | 4.69% | 23.44% |
| 3169 | 8 / repeated | 3.12% | 21.88% | 7.81% | 23.44% |
| 3169 | 10 / repeated | 0.00% | 5.47% | 0.00% | 6.25% |

Uniform challenge sources contain one symbol repeated n times, eight sources per length. Scores below require full ordinary copying, including correct stopping; minima expose weak seeds.

| Symbols | Mean sinusoidal | Mean RoPE | Worst seed sinusoidal | Worst seed RoPE |
|---|---|---|---|---|
| 3 | 87.50% | 100.00% | 75.00% | 100.00% |
| 4 | 91.67% | 100.00% | 75.00% | 100.00% |
| 5 | 87.50% | 100.00% | 62.50% | 100.00% |
| 6 | 62.50% | 100.00% | 62.50% | 100.00% |
| 8 | 0.00% | 58.33% | 0.00% | 12.50% |
| 10 | 0.00% | 50.00% | 0.00% | 12.50% |

Length 8: mean content CE 2.3741/1.1561; early-EOS rate 31.25%/27.60% (sinusoidal/RoPE).
Length 10: mean content CE 4.4288/1.6155; early-EOS rate 66.67%/34.90% (sinusoidal/RoPE).

## Conclusions and limits

Both training-fit gates pass; the held-out results can be separated from failure to fit the supplied examples.

RoPE gets every random test sequence at lengths 3–6 and every short uniform probe correct in all three seeds. This establishes learned copying and stopping on those evaluated inputs. Longer random copying remains weak, including under oracle content decoding; excluding EOS does not solve these content predictions. The marker removes the previous forced ambiguity, but the remaining learning cause is not identified.

Training fit: sinusoidal passes, rope passes.
Seen-length reliability: sinusoidal fails, rope passes.
Short uniform-copy rule: sinusoidal fails, rope passes.
The frozen RoPE benefit rule fails.
sinusoidal extrapolation: length 8 fails, length 10 fails.
rope extrapolation: length 8 fails, length 10 fails.

Paired ordinary RoPE gains at length 8/10: 3163: +6.25/+0.00 points; 3167: +3.91/+1.56 points; 3169: +18.75/+5.47 points. Seeds failing seen-length preservation: none. Extrapolation gates include seen-length reliability; keep that prerequisite separate from raw longer scores.

Both arms are marked. There is no new paired unmarked training arm, so historical scores cannot establish the marker's isolated training effect. This is one opened-data recipe, not fresh confirmation, convergence or a general positional winner. The original unmarked collision and diagnostic remain valid. Remaining errors do not by themselves identify routing, readout or optimization causes.

Close this budget without replacement seeds, extra updates or a position/LR grid. No new experiment is selected automatically.

## Verification and sources

All 18 initialized cases pass logits/all-parameter gradients, a nonzero source-marker gradient, checkpoint reload and the actual final evaluator/audit cell. Known-answer logits, absent groups, marker masking, duplicate EOS targets and criterion corruption are checked. The final audit verifies 120 pool cells, 68,064 teacher rows, 10,464 sequences, 65,770 ordinary generated tokens and 57,600 oracle tokens. Zero teacher argmax, ordinary-sequence or oracle-sequence differences. Maximum logit error 2.345e-13; scalar metric error 2.315e-14, tolerance 1e-9. Streams, initialization, optimizer counters, exposure and relevant dependency hashes pass. Reference rotations/attention/layer equations share projection/shape utilities with the base implementation.

[Report](../../../results/copy_marker_training/report.json), [per-seed/group losses and metrics](../../../results/copy_marker_training/metrics.csv), [audit](../../../results/copy_marker_training/audit.json), [preflight](../../../results/copy_marker_training/preflight.json), [profile](../../../results/copy_marker_training/profile.json). [Training source](../../../experiments/copy_marker_training.py), [format adapter](../../../experiments/copy_marker_model.py), [independent audit](../../../experiments/audit_copy_marker_training.py).

[RoFormer, section 3](https://arxiv.org/html/2104.09864v5) supports the retained Q/K rotation identity, not these training outcomes. The value-averaging boundary and marker hypothesis come from the local [copy guide](copy-rope-guide.md#a-limit-of-this-particular-variant). No paper benchmark is reproduced.
