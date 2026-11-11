# Recency helps updates but harms distant unique-key retrieval

Executed 2026-10-04. **Removing only the recency term recovers every tested unique-key error, while sharply reducing repeated-key accuracy.** It is a causal intervention on these fixed checkpoints, not a trained repair or a general verdict on attention.

[Frozen diagnostic](../../protocols/recency-ablation.md) | [Source](../../../experiments/recency_ablation.py) | [Raw results](../../../results/recency_ablation/report.json) | [Parent comparison](delta-retrieval-results.md).

## Method

Three saved softmax checkpoints, seeds 3299/3301/3307. Both native and no-recency modes evaluate the same eight original pools: train/validation 4/8 and test 4/8/16/32. CPU float64, one thread, zero training updates. These pools were already inspected; this is not fresh confirmation.

Native score at position t is `scale*q·k_t + recency*(t-(n-1))`. A separate inference copy sets `exp(log_recency)` to zero. Similarity scale, learned key/value tables, classifier, inputs and labels stay fixed. The intervention applies to every example, with no group-specific switching. Original checkpoints and reports remain unchanged.

## Results

Three-seed mean accuracy, percentages. The no-recency unique score is also 100% in every individual seed/pool/group, including all training and validation pools.

| Test events | Native unique | No-recency unique | Native repeat | No-recency repeat |
|---|---:|---:|---:|---:|
| 4 | 100.00 | 100.00 | 100.00 | 36.46 |
| 8 | 100.00 | 100.00 | 100.00 | 25.87 |
| 16 | 86.46 | 100.00 | 95.83 | 45.66 |
| 32 | 58.33 | 100.00 | 90.63 | 39.06 |

Repeated-key training accuracy falls from 100% to 45.92%/34.72% at lengths 4/8; validation falls to 51.39%/39.93%. Removing recency therefore damages the shorter task too.

| Seed | No-recency repeat, test 4/8/16/32 |
|---|---|
| 3299 | 35.42 / 24.48 / 43.75 / 38.54 |
| 3301 | 37.50 / 23.96 / 47.92 / 38.02 |
| 3307 | 36.46 / 29.17 / 45.31 / 40.62 |

Across all original pools and three checkpoints:

| Query group | Examples per mode | Rescued | Harmed | Both correct | Both wrong |
|---|---:|---:|---:|---:|---:|
| Unique | 5,184 | 318 | 0 | 4,866 | 0 |
| Repeat | 5,184 | 40 | 3,107 | 1,999 | 38 |

Counts are paired sequence evaluations, not independent seeds. Six cyclic value renamings remain within each context family. All declared diagnostic outcomes pass: long unique recovery, short unique preservation and detection of repeat cost. Passing the diagnostic does not mean passing a repair criterion.

## Why both effects make sense

**Unique key:** the matching event may be old. For a newer unrelated key, the score advantage is `scale*(cosine-1) + recency*distance`. The distance term can outweigh the similarity penalty. Removing that term is sufficient to recover the tested unique-key answers with every other learned component fixed. This establishes a contribution of the recency term here; it does not explain all training dynamics or other attention designs.

**Repeated key:** each occurrence has the same non-contextual key vector. With no recency term, its occurrences receive equal attention weights. The reader loses the ordering cue needed to select the latest assignment.

An exact structural witness is `A:1, B:4, A:3` versus `A:3, B:4, A:1`, asking A. Both have the same key/value multiset but different correct labels. Without recency, this reader produces the same weighted value and logits under their permutation. It cannot answer both correctly. The initialized fixture verifies equal logits within 1e-12; the proof follows from permutation-invariant sums. This limit concerns these event embeddings, not contextual Transformer representations.

## Verification and stopping

Three reusable test methods pass: unchanged original tensors, inverse control that incorrectly retains recency, swapped-history ambiguity, supplied scalar/label/criterion errors, absent groups and initialized checkpoint reload through the actual evaluator. Wrong variants fail the same equation checks that the intervention passes.

All 48 evaluation cells and 20,736 predictions match the separate exponential/matrix reference, maximum logit error 4.40e-14 and zero argmax changes. Independent Python-math scalar reductions agree; every native metric reproduces the parent archive. All 117 targeted parent/inherited files remain unchanged. Reference code shares parameters and utilities; it is not an independent software stack. [Verification](../../../results/recency_ablation/verification.json).

Close this diagnostic. Simply deleting recency is unsuitable for the combined task. The original failed reliability/resource gate remains unchanged. No new training, bias grid, group-specific selector or GPU benchmark follows automatically; any repair needs a distinct design and fresh confirmation.
