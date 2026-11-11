# Which historical updates carry the growing EOS preference?

2026-10-08. **Decoder cross-attention Q/K updates do not carry most of the weak seed's increase between updates 500 and 1,000.** The frozen support criterion fails. Updating the remaining parameters with earlier Q/K already produces EOS on all 128 long examples, but that hybrid also damages short-task fitting.

[Protocol](../../protocols/copy-routing-development.md) · [Report and audit](../../../results/copy_routing_development/report.json) · [Every group and loss](../../../results/copy_routing_development/metrics.csv)

## How to read the intervention

Q/K projections turn the decoder state and encoder memory into attention queries and keys. Their compatibility determines attention weights. However, the input states themselves are learned elsewhere. Copying Q/K parameters does not copy all routing behavior.

Within each run, combine parameters from its saved 500 and 1,000 checkpoints. The four corners are:

| Corner | Cross-attention Q/K weights and biases | Everything else |
|---|---|---|
| 00 | Update 500 | Update 500 |
| 10 | Update 1,000 | Update 500 |
| 01 | Update 500 | Update 1,000 |
| 11 | Update 1,000 | Update 1,000 |

Only the Q/K blocks of the packed decoder cross-attention projections change under the first factor: 144 scalars. The second factor includes V, output projection, encoder/decoder representations, normalization, FFNs and classifier. No cross-seed donors or retraining. All corners receive identical correct prefixes at output position 5, for paired six/ten-symbol sources.

## Weak seed: counts and margin

Counts are out of 128. D is the long minus short mean of `EOS logit - correct-token logit`. It measures source-extension sensitivity; it is not accuracy.

| Corner | Paired-short correct | Paired-long correct | Long EOS | D | Lowest short-training group exact accuracy |
|---|---:|---:|---:|---:|---:|
| 00 | 123 | 49 | 79 | 2.732 | 42.97% |
| 10 | 127 | 54 | 74 | 3.581 | 25.00% |
| 01 | 128 | 0 | 128 | 11.161 | 21.88% |
| 11 | 128 | 2 | 126 | 10.482 | 93.75% |

Q/K-only updating gives 8 long-answer rescues and 3 harms relative to 00. Reverting Q/K at the later rest (01 versus 11) harms the two correct long answers. Both hybrids pass the narrowly declared paired-short preservation rule, but neither preserves short copying generally. Training, validation and test losses/accuracies for every group remain in the CSV. These checkpoints precede full weak-seed fitting.

## Conditional effects and interaction

For D at each corner, Q/K effects are `D10 - D00` and `D11 - D01`. Rest effects are `D01 - D00` and `D11 - D10`. Interaction is `I = D11 - D10 - D01 + D00`; zero would mean additive effects on this particular margin scale.

| Model | Native D increase | Q/K at earlier rest | Q/K at later rest | Rest at earlier Q/K | Rest at later Q/K | Interaction |
|---|---:|---:|---:|---:|---:|---:|
| Cross-only 3767 | +7.750 | +0.849 | -0.679 | +8.429 | +6.901 | -1.528 |
| Cross-only 3779 | -1.084 | -0.333 | -0.831 | -0.253 | -0.751 | -0.497 |
| Cross-only 3793 | -0.488 | -1.054 | -2.883 | +2.395 | +0.566 | -1.829 |
| Combined 3767 | -0.122 | -0.001 | -0.352 | +0.230 | -0.121 | -0.351 |

The weak native increase is positive in all 128 pairs. Its rest effects are positive in all pairs in both contexts. Its Q/K effect is positive in 105/128 pairs at earlier rest, but negative in 101/128 at later rest. Thus the mean sign reversal is accompanied by substantial individual variation. The frozen >=90%-of-native-increase rule fails in both Q/K contexts; paired-short preservation passes separately.

## Successful controls and losing cases

All paired-short group preservation checks pass. This does not hide long-answer harms:

| Model | Long correct: 00 / 10 / 01 / 11 | Hybrid 10 rescues / harms versus 00 | Hybrid 01 rescues / harms versus 11 |
|---|---|---|---|
| Cross-only 3779 | 120 / 127 / 128 / 128 | 7 / 0 | 0 / 0 |
| Cross-only 3793 | 98 / 110 / 100 / 124 | 12 / 0 | 0 / 24 |
| Combined 3767 | 96 / 94 / 128 / 128 | 1 / 3 | 0 / 0 |

The combined 01 hybrid retains all paired answers but its lowest short-training exact group accuracy is only 65.63%, versus 100% in 11. Cross-only 3793's 10 hybrid falls to 95.31%, versus 98.44% in 00. The positive-increase support rule targets the weak seed; controls have decreasing native D and are descriptive comparisons, not failed repair candidates.

## What this narrows

The simple account that these historical Q/K projection updates alone create the growing failure is unsupported. Most of the measured increase is carried by the remaining update package under the tested hybrids. In the later context, Q/K changes actually reduce the extension margin. This does not identify one component inside that package, or imply that attention routing is irrelevant: encoder and decoder states also determine Q/K inputs.

Hybrids are conditional endpoint computations. Their general short-task damage limits interpreting them as intact learned circuits. They do not measure what training would have done with Q/K frozen, reconstruct individual optimizer steps, or demonstrate a reliable repair. The 500-to-1,000 window was selected from opened trajectories, not fresh confirmation. Stop this comparison; no component sweep is queued.

## Verification and sources

16 corners, 4,096 paired selected predictions and 135,168 short teacher-forced target predictions per implementation; zero training. Every seed/group, including absent groups, is preserved. Saved initialized donor/evaluator checks, independent chunk-based parameter checks, identical-donor null, ties, missing groups and deliberate value-block/logit/label errors pass. Native corners reproduce saved historical outputs; reference equations and scalar metrics agree within 1.040e-13 with zero argmax differences. Persisted hybrid logits are reloaded and audited. All 78 protected files are unchanged. Forward and metric helpers are shared, so independence is partial.

[Vaswani et al. (2017), sections 3.2.1–3.2.3](https://arxiv.org/html/1706.03762v7#S3.SS2), inspected, defines query/key weighting, projected values and encoder-decoder attention. The parameter partition, factorial algebra, thresholds and measured historical effects are local work; they are not predictions from that paper.
