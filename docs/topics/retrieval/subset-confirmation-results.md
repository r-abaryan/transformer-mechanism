# Rescue depends on the retained subset

**The earlier combined rescue did not generalize across subset choices.** This narrows the previous finding rather than invalidating its recorded measurements.

## Method and meaning

Follow the [predeclared protocol](../../subset-confirmation-protocol.md): 512 fresh episodes per length, excluding both previous long evaluation sets by canonical mapping/query identity, five frozen deep models, no training. Keep the queried pair and enumerate every pair of remaining distractors, preserving order. There are C(n-1,2) = 3 or 10 crops per episode. Reuse the [verified position interventions](../positions/position-distractor-results.md).

For correctness c(i,s), subset-average accuracy is mean(i,s) c(i,s); robust episode accuracy is mean(i) min(s) c(i,s). The latter requires every crop of an episode to succeed. Subsets are dependent observations, not independent trials. Fresh long inputs do not guarantee novel cropped mappings.

## Results

Percentages averaged across five seeds. Original long accuracy was 81.84% at four pairs and 66.76% at six.

| Crop condition | Four: subset average | Four: every subset correct | Six: subset average | Six: every subset correct |
|---|---:|---:|---:|---:|
| Original positions | 86.43 | 75.39 | 84.32 | 50.59 |
| Original pair positions + tail reset | 87.43 | 77.38 | 87.66 | 57.73 |
| Compact positions 0..7 | 100 | 100 | 99.996 | 99.96 |

Both predeclared tail-reset criteria failed at both lengths: neither >=90% subset-average accuracy nor >=90% robust episode accuracy held for every seed.

| Seed | Four: average / robust | Six: average / robust |
|---|---:|---:|
| 137 | 88.41 / 80.27 | 91.88 / 62.89 |
| 163 | 85.94 / 74.02 | 84.69 / 51.37 |
| 191 | 88.54 / 79.30 | 86.54 / 60.16 |
| 223 | 84.83 / 72.07 | 86.25 / 50.59 |
| 257 | 89.45 / 81.25 | 88.96 / 63.67 |

The original first-two-distractor selection, with tail reset, still averaged **96.13% / 94.57%** on these fresh inputs. Thus averaging over different crops exposes sensitivity that the earlier selection concealed; the original rule was fixed before its results, not chosen afterward. Mean tail-reset cross-entropy was 0.7396 / 0.7056 nats; compact crops were about 0.0005 at both lengths.

## Evidence, limits and next step

[Raw report](../../../results/subset_confirmation.json), [exact inputs](../../../results/subset_confirmation_splits.json), and [script](../../../experiments/confirm_subsets.py) preserve all per-seed results, per-episode correct-subset counts, paired rescue/harm counts and provenance. Label preservation, unique subset enumeration, exclusion checks, finite outputs, sham tolerance and unchanged checkpoint hashes passed. Source hashes and aggregation from saved counts were checked separately.

Compacting the same crops almost eliminates errors, supporting an effect of positional layout under this intervention. However, changing subsets also changes content, key locations and gaps. These results do not identify which of those properties causes failure, prove a circuit, or establish a practical repair. This is local evidence, not a paper benchmark replication; the [prior paper comparison](../training/depth-comparison-results.md#paper-comparison-and-remaining-question) remains unchanged.

Follow-up completed: [fixed-content offset/gap/tail comparisons](../positions/layout-results.md). Positional components have conditional, non-additive effects. Preserve the compact control's rare failure as well as the larger negative findings; layer mechanisms remain unresolved.
