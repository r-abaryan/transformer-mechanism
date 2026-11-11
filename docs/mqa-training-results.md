# MHA versus MQA: trained results

**MQA passes every fixed acceptance criterion.** It halves compact K/V payload and removes 544 parameters (10.69%). Six-pair mean accuracy decreases slightly, and some seeds lose more. Acceptance permits these losses; it does not establish equal quality.

## What we tested

Two query heads with separate K/V (MHA), versus the same query heads reading one shared K/V set (MQA). Both train from scratch. The [learning guide](topics/attention/mqa-training-guide.md) explains the equations, gradient accumulation and diagram; the [frozen protocol](mqa-training-protocol.md) specifies matching and thresholds.

Five paired seeds, RoPE/ReLU/FFN32, width 16, two pre-norm blocks. Q, embeddings, norms, FFN and output projections start identically within pairs; MQA takes the first untrained template K/V head. Both receive identical expanded queries and labels, Adam .001, batch 64, 6,000 updates and 3,456,000 tokens. All ten runs finish without nonfinite failures or replacement. Final tests open after all attempts.

Four-pair families: 800/100/200 train/validation/test; six-pair: 200 test only. These exclude 6,520 previously used families. The three-pair 160/20/40 splits are reused diagnostics. Each family has 36 order/value variants, with all present-key queries. Test answer counts per model: 2,880 / 21,600 / 36,000 at 3/4/6 pairs. Variants and queries within families are dependent.

## Mean accuracy across five seeds

R queries the duplicated key and requires its latest value. U queries a key occurring once. pp means percentage points.

| Pairs | MHA R | MQA R | Difference, pp | MHA U | MQA U | Difference, pp |
|---|---:|---:|---:|---:|---:|---:|
| 3 (reused) | 100.00% | 100.00% | +0.000 | 100.00% | 100.00% | +0.000 |
| 4 | 100.00% | 99.99% | -0.003 | 99.99% | 99.99% | -0.004 |
| 6 | 99.44% | 99.26% | -0.178 | 94.73% | 94.05% | -0.674 |

All-query context accuracy requires every available query to be correct. At six pairs it falls from **78.98% to 76.21%**. Overall answer accuracy falls from 95.67% to 95.09%. These stricter/context-level measures are supplementary; they were not additional adoption thresholds.

## Every seed at six pairs

| Seed | MHA R | MQA R | MHA U | MQA U | U difference, pp |
|---|---:|---:|---:|---:|---:|
| 1411 | 99.46% | 98.82% | 94.10% | 91.77% | -2.333 |
| 1423 | 99.07% | 99.25% | 94.45% | 96.44% | +1.990 |
| 1451 | 99.83% | 99.58% | 96.00% | 96.04% | +0.038 |
| 1481 | 99.40% | 99.49% | 95.69% | 91.58% | -4.111 |
| 1499 | 99.44% | 99.18% | 93.39% | 94.44% | +1.045 |

Three seeds gain unique-query accuracy; two lose. The largest loss is 4.111pp (seed 1481). The lowest MQA unique-query accuracy is 91.58%; all repeated-query scores exceed 98.8%. These results cannot identify a pure sharing-capacity effect separately from this initialization and finite-budget optimization.

## Resources and fixed decision

| Requirement | Observed result |
|---|---|
| Parameters and half compact cache | 5,088 -> 4,544; six-pair batch-one float64 payload 7 KiB -> 3.5 KiB |
| Every MQA R/U score >=99% at 3/4 pairs | Pass; minimum four-pair R 99.97%, U 99.94% |
| Every MQA R/U score >=90% at six pairs | Pass |
| Mean paired losses <=1pp for each type at 4/6 | Pass; largest mean loss 0.674pp (six-pair U) |
| No six-pair seed/type loses >5pp | Pass; worst 4.111pp |

Cache payload is `B L T G(d_k+d_v) s`: reducing G from 2 to 1 halves it at fixed dimensions. Measured batch-one payloads at 3/4/6 pairs are MHA 4/5/7 KiB and MQA 2/2.5/3.5 KiB. These counts exclude allocator overhead and attention/broadcast temporaries. Training uses full causal attention, not a cached decoding benchmark.

Equal inputs and update counts do not imply equal FLOPs: MQA uses fewer K/V projections. Recorded loop totals are 256.25s (MHA) and 254.27s (MQA), including periodic validation and excluding checkpoint writes/final tests. They are not independent timing trials or a decoding speed measurement.

## Cross-entropy and verification

Cross-entropy penalizes low probability on the correct answer; lower is better. Six-pair mean repeated CE rises 0.02354 -> 0.03292, and unique CE rises 0.31746 -> 0.40035. Thus confidence quality also worsens on average. Full metrics, curves and exposure counts remain in the raw report.

Pretraining forward/gradient checks pass, with maximum gradient discrepancy 2.78e-16. The MHA refactor matches the prior implementation, and both causal controls pass. The independent final audit reproduces all **30 evaluations**, maximum logit discrepancy `1.39e-13`. It checks data exclusions/labels, initial matching, regenerated streams/counts, 6,000 Adam steps, cache sizes and provenance hashes.

Evidence: [raw report](../results/mqa_training.json), [data](../results/mqa_training_data.json), [saved scores](../results/mqa_training_scores.pt), [pretraining verification](../results/mqa_training_verification.json), [final audit](../results/mqa_training_audit.json). Code: [training](../experiments/mqa_training.py), [verification](../experiments/verify_mqa_training.py), [auditor](../experiments/audit_mqa_training.py).

```powershell
.venv/Scripts/python.exe experiments/audit_mqa_training.py
```

## Papers, interpretation and closure

[Shazeer (2019)](https://arxiv.org/html/1911.02150v1) defines shared K/V with separate queries; our structure follows that distinction. [GQA (2023 v3)](https://arxiv.org/html/2305.13245v3) pools pretrained heads and adapts them with continued training. Here both models start untrained, with one selected template head. Its language benchmarks and hardware results remain unreplicated. [Source reading depth](reference/sources.md).

**Keep MQA as an accepted option for this task and resource rule; close the comparison.** Half-size cache payload comes with measured mean and seed-specific losses. One split/five seeds do not prove statistical equivalence, general superiority or a language-model speed/quality tradeoff. Other tasks, initialization recipes, group counts and GPU decoding remain unknown. No automatic sweep follows.
