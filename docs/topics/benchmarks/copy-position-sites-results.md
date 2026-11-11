# Which positional-exposure sites repair longer copying?

Completed 2026-10-07. **Only combined self- and cross-attention exposure passes the all-seed reliability rule.** Cross-only succeeds on random long inputs in two seeds but fails at length 10 in the third. Self-only remains weak. All conditions master the short task.

[Guide](copy-position-sites-guide.md) · [Frozen protocol](../../protocols/copy-position-sites.md) · [Report](../../../results/copy_position_sites/report.json) · [Every metric](../../../results/copy_position_sites/metrics.csv) · [Audit](../../../results/copy_position_sites/audit.json)

## Method

Four matched arms apply sorted sampled position IDs to neither site, self-attention only, cross-attention only, or both. Self-attention groups encoder and decoder self-attention. The 1,888-parameter model, token batches, labels, initial weights and 6,000-update budget are matched within each seed. Training content lengths remain 3–6. Evaluation uses ordinary positions and generates its own prefixes, with no supplied alignment.

The RoPE score is `qᵀ R_(r−p) k / √d`. Spacing changes relative offsets without adding tokens. [Ruoss et al. (2023), section 3](https://arxiv.org/html/2305.16843v1#S3) supplies the published sampling method; that section was inspected. The site comparison and scores are local evidence, not novelty or a paper reproduction.

## Full-copy results

Accuracy requires every content token and correct stopping. All nine primary groups are shown; percentages are not independent seed counts.

| Seed | Length / group | Ordinary % | Self-only % | Cross-only % | Both % |
|---|---|---:|---:|---:|---:|
| 3767 | 8 / distinct | 0.00 | 1.56 | 92.19 | 100 |
| 3767 | 8 / repeated | 3.13 | 4.69 | 93.75 | 100 |
| 3767 | 10 / repeated | 0.00 | 0.00 | 0.00 | 100 |
| 3779 | 8 / distinct | 3.13 | 3.13 | 100 | 100 |
| 3779 | 8 / repeated | 9.38 | 3.13 | 100 | 100 |
| 3779 | 10 / repeated | 0.00 | 0.00 | 100 | 100 |
| 3793 | 8 / distinct | 0.00 | 3.13 | 100 | 100 |
| 3793 | 8 / repeated | 6.25 | 6.25 | 100 | 100 |
| 3793 | 10 / repeated | 0.00 | 2.34 | 100 | 100 |

| Arm | Mean copy %, length 8 / 10 | Mean teacher CE, length 8 / 10 |
|---|---:|---:|
| Ordinary | 3.65 / 0.00 | 1.49263 / 2.27037 |
| Self-only | 3.65 / 0.78 | 1.09023 / 1.70794 |
| Cross-only | 97.66 / 66.67 | 0.00933 / 0.20950 |
| Both | 100 / 100 | 0.000178 / 0.000210 |

All confirmation arms reach 100% in every ordinary short train, validation and test group. Across their pooled short seed/length cells, teacher CE ranges are 0.000127–0.000389 for training, 0.000129–0.000565 for validation and 0.000131–0.000367 for test. Every required layout-fit gate passes. Thus cross-only seed 3767's long failure is a transfer failure, not failure to master the declared short training setting.

Uniform long probes also favor both: 100% everywhere. Cross-only length-10 uniform accuracy is 12.5%, 100% and 87.5% across the three seeds. The CSV preserves all uniform and short groups; they are descriptive and not fresh independent confirmation.

## Ranking and interpretation

Combined exposure reduces first-source outranking by at least 10 percentage points in every primary group, satisfying the joint prediction. Neither single-site condition meets the all-seed sufficiency rule.

Ranking is not a complete explanation. Cross-only seed 3767 has an outranking head on every length-8 example, yet copies 92.97% fully correctly. Combined seed 3793 copies every length-10 example despite outranking in 9.38%. The statistic asks whether *either* head prefers another position; it does not require that the whole model answer incorrectly.

Adding self-site exposure to cross-only repairs the observed seed-3767 failure. But cross-only's two successful random-test seeds rule out universal necessity of self-site exposure. The accuracy interaction `A_both − A_self − A_cross + A_ordinary` ranges from −3.13 to +100 percentage points across primary cells; it is not uniformly positive or an additive causal share.

This identifies an effective combination of training interventions. It does not identify where all useful weight changes reside: gradients from either site reach other components. Encoder versus decoder self-attention, exact learned circuits and the optimization trajectory remain unresolved.

## Execution, verification and limits

All four development arms pass at 6,000 updates, so the 12,000 fallback is unused. Sixteen actual runs execute **96,000 updates, 15.312 million sequences and 80.64 million target tokens**, plus the same number of source tokens. CPU float64, one thread. Recorded training/development-evaluation segments total about 23.0 minutes, excluding final scoring and audit. Complete-step profiling measured about 0.017–0.022 seconds after first steps, with 30,340 bytes of Adam tensor state. GPU unused; CPU peak memory unmeasured.

Preflight passes twelve initialized output/gradient cases, positional wiring, supplied metrics, saved two-update runs and deliberate faults. An additional exact step-zero saved-checkpoint audit passed during development, not before training. Final independent checks cover 528 cells/78,336 sequence evaluations, zero prediction differences and maximum logit/score discrepancy 2.682e-13; all 26 protected files remain unchanged. Reference structure/projection helpers are shared; full training gradients are not replayed.

The 256 reserved long sequences exclude previous pools but are shared across three seeds. Short pools are reused. Test positions 0–10 lie within the sampled training range. No arbitrary-length, other-task or universal architecture claim follows. The declared comparison is complete; [the copy assessment](../../assessments/copy-generalization-assessment.md) consolidates the evidence. No further sweep is queued.
