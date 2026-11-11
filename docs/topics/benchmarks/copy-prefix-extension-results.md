# Source extension triggers EOS at an unchanged decoder prefix

2026-10-07. **At output position 5, the weak cross-only model changes from correct content on all 128 short sources to EOS on all 128 extended sources.** The decoder prefix and correct target are identical within each pair. All three successful controls remain correct. The frozen primary criterion passes.

[Protocol](../../protocols/copy-prefix-extension.md) · [Report and audit](../../../results/copy_prefix_extension/report.json) · [Every group](../../../results/copy_prefix_extension/metrics.csv)

## Paired inputs, not another component ablation

Take each of the 128 previously opened test10 sources and pair it with its first six symbols. The first six symbols are unchanged; four suffix symbols are added before source EOS. At output position t, both models receive exactly `BOS + source[:t−1]` and must predict the same `source[t−1]`. Positions are 1-based; ordinary RoPE IDs are used throughout.

For example, at position 5 both branches receive BOS and the same first four correct symbols. Only the encoded source differs. This controls decoder position and prefix content, not the entire source representation. It is a supplied-correct-prefix next-token test, not full generated copying.

## Results

Each cell contains 128 paired examples. Margin is `z_EOS − z_correct`; a positive change means extension increases EOS preference relative to the correct token.

| Model | Position | Short correct | Long correct | Correct → EOS | Mean margin change |
|---|---:|---:|---:|---:|---:|
| Cross-only 3767 | 4 | 128 | 121 | 7 | +4.538 |
| Cross-only 3767 | 5 | 128 | 0 | 128 | +17.752 |
| Cross-only 3779 | 4 | 128 | 128 | 0 | +0.138 |
| Cross-only 3779 | 5 | 128 | 128 | 0 | +0.815 |
| Cross-only 3793 | 4 | 128 | 128 | 0 | +0.283 |
| Cross-only 3793 | 5 | 128 | 128 | 0 | +1.618 |
| Combined 3767 | 4 | 128 | 128 | 0 | −0.173 |
| Combined 3767 | 5 | 128 | 128 | 0 | +0.265 |

Weak-seed position-5 mean margin goes from −11.052 to +6.700. Every pair increases the margin and flips to EOS, exceeding both frozen 90% criteria. The successful cross-only models also increase their margins on all position-5 pairs, but remain correct. Margin increase alone does not establish failure.

The position-5 count is 128, while earlier native generation first failed there on 121 examples. The other seven had already stopped at position 4. Here their correct prefixes are supplied, so their position-5 predictions can also be measured; those seven are not naturally reached decisions.

## What this narrows

Decoder position or prefix alone cannot account for the weak model's changed answer in these pairs. The source-extension package triggers its early-EOS preference at a fixed decoder input. This also rules out explaining the original short/long contrast solely by different early source symbols.

It does **not** identify pure sequence length as the cause. Added suffix values, extra attention competitors, contextual encoder representations and relocation of source EOS change together. Combined with [the failed direct end-route removal](copy-end-route-results.md), the evidence favors investigating the extended-source computation as a whole, without attributing the failure to that direct marker contribution. No further test is selected here.

## Verification and scope

Four saved models × two positions × two source lengths ×128 = 2,048 selected predictions per implementation. Initialized saved-reload, correct-label/prefix contracts, deliberate label error, identical-input null, tied-logit and absent-group checks passed before scoring. Explicit reference equations agree with native outputs and archived long logits within 1.089e-13; all argmax predictions agree. Independent scalar counts and margins check the paired summaries. Shared attention/projection utilities limit independence.

The long examples were already opened and selected because the weak model failed on them. Truncation creates paired short inputs, not independent confirmation or new training evidence. Groups are defined by whether the six-symbol prefix repeats symbols; absent groups remain explicit in the CSV. All 128 short prefixes are unique and absent from the prior train6/validation6/test6/uniform6 pools; 10 have distinct symbols and 118 repeat. All 49 protected files remain unchanged. Zero training, no weight changes and no source-length grid.

[Newman et al. (2020), sections 1 and 3](https://arxiv.org/html/2010.07174v1#S3), inspected in the preceding review, motivates separating content from termination. The paired construction, thresholds and measured flips are local, not predictions from that paper. Stop this diagnostic; the internal cause and its learning history remain unresolved.
