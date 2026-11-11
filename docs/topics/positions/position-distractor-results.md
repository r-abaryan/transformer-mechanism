# Position changes versus distractor removal

**Position resets alone were insufficient.** Removing distractors helped substantially; combining removal with a tail-position reset reached 90% for every seed at both tested lengths. This is a frozen-model diagnostic, not a proposed architecture repair.

## What was changed, and why?

Follow the [fixed protocol](../../position-distractor-protocol.md): five deep checkpoints, the same 1,024 examples at each length, no training. Hook the embedding output with \(\Delta=PE(p_{desired})-PE(p_{original})\), so the unchanged forward uses the desired sinusoidal positions. This changes position information in all downstream paths, not only attention.

For removal, retain the queried pair plus the first two other pairs, preserving their order. Keeping original position indices holds retained-token positions fixed while removing context. Compacting those same tokens additionally restores indices 0..7. Labels were verified unchanged. Selection uses the queried key and is deliberately diagnostic.

## Findings

Mean accuracy / mean cross-entropy in nats across five seeds. All-seed criteria require each seed to reach 90%, not just the mean.

| Condition | Four pairs | Six pairs | All seeds >=90%, four / six |
|---|---:|---:|---|
| Original | 80.43% / 1.1740 | 64.24% / 2.2840 | No / No |
| Query PE reset to 7 | 81.43% / 1.1426 | 67.30% / 2.0725 | No / No |
| Marker/query PE reset to 6/7 | 81.52% / 1.1400 | 67.64% / 2.0631 | No / No |
| Retain three pairs, original PE | 95.18% / 0.2558 | 89.69% / 0.5826 | Yes / No |
| Retain three pairs, reset tail PE | 96.05% / 0.2344 | 95.20% / 0.2793 | Yes / Yes |
| Retain three pairs, compact PE | 100% / 0.0004 | 100% / 0.0004 | Yes / Yes |

The six-pair removal result depends on seed. These values prevent the mean from hiding a failure:

| Seed | Original | Removal, original PE | Removal + tail reset |
|---|---:|---:|---:|
| 137 | 68.75% | 94.24% | 97.27% |
| 163 | 62.89% | 92.48% | 93.36% |
| 191 | 64.36% | 95.31% | 96.29% |
| 223 | 54.88% | 75.29% | 94.14% |
| 257 | 70.31% | 91.11% | 94.92% |

Paired effects include regressions. Across 5,120 six-pair model/example evaluations, removal with original PE rescued 1,364 original errors but harmed 61 correct predictions. Removal plus tail reset rescued 1,622 and harmed 37. These are paired counts, not 5,120 independent model trials; every seed uses the same examples. All conditions' counts are in the raw report.

## What this establishes and leaves unknown

Removing selected distractors changes behavior even when remaining position vectors are preserved. Query/tail positions alone do not explain the long-input failures. Seed 223's large additional rescue after removal and tail reset suggests interaction under these interventions; it does not identify a unique circuit.

Removal also changes attention normalization and intermediate contextual representations. Position resets can duplicate other tokens' positions. Crops may resemble training cases and use only one deterministic subset rule. Therefore neither “attention dilution is the cause” nor “positions are irrelevant” follows. Perfect compact-crop accuracy checks behavior back at training length, not general length robustness.

This extends the [local depth findings and paper comparison](../training/depth-comparison-results.md#paper-comparison-and-remaining-question). No new paper benchmark or binding mechanism has been reproduced.

## Verification and next step

[Script](../../../experiments/position_distractors.py) and [raw results](../../../results/position_distractors.json) retain source/protocol/checkpoint hashes and all per-seed metrics. Original accuracy/loss matched saved results within 1e-12; all sham-hook logit errors were zero, all outputs finite, and checkpoint hashes unchanged. A separate explicit-position forward calculation checked all six conditions on 16 examples per length and seed (60 fixtures): maximum logit difference 2.34e-14 or less. This verifies hook arithmetic with the same PyTorch blocks, not an independent framework.

Follow-up: [all eligible retained subsets on fresh long examples](../retrieval/subset-confirmation-results.md) failed the generalized rescue criteria. Investigate positional layout with crop content fixed before layer/path attribution. Do not tune or change architecture from this diagnostic alone.
