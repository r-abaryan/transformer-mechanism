# Moving all positions together

**The models are sensitive to absolute position, even when distances stay fixed.** This conclusion applies to these frozen models and cropped inputs.

## Short explanation

Start with positions 0,1,2,3,4,5,6,7. Add four to every position. They become 4,5,6,7,8,9,10,11. Every distance stays the same. The tokens and answer also stay the same. Yet predictions change and accuracy falls. Relative distances alone therefore cannot explain the model's behavior.

The second condition moves only the six pair tokens. The marker and query stay at 6 and 7. That also changes pair-to-query distances. Some position numbers collide or pass the query number. These are artificial diagnostic inputs; causal masking still follows token order.

## Method and math

The [protocol](../../shift-protocol.md) fixed shifts 0,1,2,4,6,8 before execution. Reuse all compact crops from 512 four-pair and 512 six-pair parent episodes and all five frozen deep models. No training. These inspected examples are not a new confirmation set. Every crop still contains three pairs and eight tokens.

Global shift: p_i=i+s. Relative distance stays (i+s)-(j+s)=i-j. Pair-only shift: p_i=i+s for i<6; p_6=6 and p_7=7. The verified hook changes only sinusoidal position vectors. Full model outputs need not be invariant to this change; distances being preserved does not make token embeddings plus positional vectors identical.

## Results

Mean accuracy across five seeds. Parent length identifies where crops came from, not their current token count.

| Shift | Four-derived: global | Four-derived: pairs only | Six-derived: global | Six-derived: pairs only |
|---|---:|---:|---:|---:|
| 0 | 100% | 100% | 99.996% | 99.996% |
| 1 | 91.58% | 91.63% | 91.48% | 91.81% |
| 2 | 77.57% | 77.93% | 77.61% | 78.22% |
| 4 | 59.79% | 61.80% | 60.60% | 62.26% |
| 6 | 85.73% | 92.86% | 86.25% | 92.87% |
| 8 | 68.24% | 74.02% | 69.37% | 74.77% |

Global logit invariance failed at every positive shift for both parent lengths. Every seed retained >=90% global-shift accuracy only at shift 1. The criterion across all positive shifts failed. Larger shifts do not cause a monotonic decline; do not infer a simple distance penalty or a proven periodic mechanism.

| Seed | Six-derived global +2 | Global +4 | Global +6 |
|---|---:|---:|---:|
| 137 | 78.75% | 65.74% | 92.81% |
| 163 | 75.74% | 58.61% | 93.91% |
| 191 | 79.79% | 61.07% | 82.62% |
| 223 | 75.68% | 54.98% | 74.02% |
| 257 | 78.09% | 62.60% | 87.91% |

Mean six-derived global cross-entropy rises from 0.0005 nats at zero shift to 2.2016 at +4, then falls to 0.7045 at +6. All per-seed losses, prediction changes, rescued/harmed counts and robust-episode accuracies are in the raw report. Crops share parent episodes; they are not independent trials.

## Evidence and limits

[Script](../../../experiments/position_shifts.py), [raw report](../../../results/position_shifts.json) and [input artifact](../../../results/subset_confirmation_splits.json) preserve provenance. Unshifted metrics reproduced the compact baseline within 1e-12. Zero-shift sham checks passed below 1e-10. Coordinate identities, labels, finite outputs and unchanged checkpoint hashes passed. Source hashes and paired rescue/harm arithmetic were checked separately.

This rules out a purely relative-distance explanation of these predictions. It does not show that relative distances never matter. It does not identify which layer causes sensitivity, establish a universal transformer limitation, or explain every failure on uncropped long inputs. No architecture improvement or paper benchmark is claimed; the [prior literature comparison](../training/depth-comparison-results.md#paper-comparison-and-remaining-question) remains unchanged.

Follow-up completed: [attention/residual/FFN activation replacements](../retrieval/activation-results.md). Second-block attention donors nearly restore accuracy. Reverse and unrelated-donor controls remain necessary before stronger mechanism claims.
