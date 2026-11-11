# A supplied head-0 pattern recovers the weak model's fifth token

2026-10-08. **The weak model improves from 2/128 to 128/128 correct long-source answers. The wrong-position control gives 0/128.** Recovery, control preservation and the declared location-specificity criterion all pass. This is a single-decision intervention at update 1,000, not generated full copying.

[Protocol](../../protocols/copy-head-pattern-transfer.md) · [Report and audit](../../../results/copy_head_pattern_transfer/report.json) · [Every group](../../../results/copy_head_pattern_transfer/metrics.csv)

## What was transferred

For each ten-symbol source, run its paired first-six-symbol source with the same correct decoder prefix. Take only head 0's attention weights at output position 5. Transfer weights on positions 0..5 directly; move the short marker weight from index 6 to the real long marker at index 10. Give added content positions 6..9 zero weight.

Use the long source's value vectors. Head 1, encoder computation, decoder residual, output bias and all learned parameters remain intact. Run the actual remaining normalization, FFN and classifier. Thus the intervention changes where one head reads, while the values and subsequent processing belong to the long-source model.

The wrong-position control swaps the transferred marker and position-zero weights. It has exactly the same weight multiset and entropy as the mapped pattern. It changes where the large marker weight goes. Position zero may contain a repeated target symbol, so this is a location control, not a semantic guarantee about that token.

## Results

All counts are out of 128. The other models are the previously successful controls, evaluated here at their earlier update-1,000 checkpoints.

| Model | Native correct | Mapped correct | Wrong-position correct | Mapped rescues / harms |
|---|---:|---:|---:|---:|
| Cross-only 3767 | 2 | 128 | 0 | 126 / 0 |
| Cross-only 3779 | 128 | 128 | 128 | 0 / 0 |
| Cross-only 3793 | 124 | 126 | 126 | 2 / 0 |
| Combined 3767 | 128 | 128 | 128 | 0 / 0 |

Weak-seed recovery includes all 10 distinct-prefix and 118 repeated-prefix examples. Its wrong-position control harms the two originally correct answers. Cross-only 3793 retains two repeated-prefix EOS errors after transfer; neither control arm repairs them. No mapped intervention harms a correct answer in any model/group.

| Weak-seed measure | Native | Mapped | Wrong position |
|---|---:|---:|---:|
| EOS answers | 126 | 0 | 128 |
| Mean EOS-minus-target margin | +4.007 | -6.531 | +5.405 |
| Mean token cross-entropy | 4.079 | 0.029 | 5.421 |

The frozen rules were >=95% weak recovery, <=5 percentage-point loss in every successful-control group, and >=50 percentage-point mapped advantage over the weak wrong-position control. Observed values satisfy all three. These thresholds describe this local diagnostic only.

## Why this is informative

The intact trace showed a sharp short-source marker read in weak head 0, but a mixed long-source read. Replacing that long pattern is sufficient to expose the correct content answer on every tested example with native long values and downstream processing. Merely retaining the pattern's sharpness at the wrong location fails completely in this weak model.

This does not prove the marker weight alone is responsible: the whole head pattern changes, including suffix exclusion and other mixture weights. The earlier direct-marker contribution removal at a later checkpoint was a different operator and cannot serve as its opposite. Historical Q/K parameter swaps also differ: their inputs and downstream parameter compatibility were not held fixed.

For the successful controls, the wrong-position permutation is nearly a null operation. Mean L1 differences from the mapped head pattern are approximately 3.9e-17, 2.0e-9 and 1.1e-12; the weak model's difference is 1.994. Therefore the successful-control results do not establish broad marker-location specificity. They establish absence of measured mapped-answer harms here.

## Math and boundary

For head-0 projected long values `u_j = v_0j W_O,0^T`, the modified cross output is `c' = c + sum_j (a'_0j - a_0j) u_j`. Other heads and output bias cancel in this difference. The producer applies this change before norm2; the reference rebuilds both heads directly and runs explicit downstream equations.

The supplied short pattern and known source correspondence are diagnostic information. We have not trained a repair, tested proper terminal stopping, measured complete generated copying under this intervention, or established why optimization learned the pattern. The 128 examples and chosen checkpoint were already opened. Stop this transfer; no further head, mapping or checkpoint search is queued.

## Verification and source

Twelve model/mode cells, 1,536 selected long predictions per implementation; 512 short-to-self identity examples; zero training. Initialized saved reload, independent mapping checks, unchanged head1, weight-mass checks, control permutation, absent/missing groups and deliberate mapping/logit/label errors pass. Archived native outputs and explicit downstream reference agree with zero argmax differences, maximum discrepancy 6.351e-14. Saved logits are reloaded for scalar audits. All 89 protected files remain unchanged; shared model/trace helpers limit independence.

Preflight caught a reference-code assumption that the final classifier had a bias. The actual classifier is bias-free; its explicit equation was corrected before freezing or evaluating trained checkpoints. No empirical criterion changed.

[Vaswani et al. (2017), section 3.2](https://arxiv.org/html/1706.03762v7#S3.SS2), inspected earlier, defines weighted values and multi-head projection. The transfer, mapping control, thresholds and results are local work. The paper supplies no prediction for this recovery.
