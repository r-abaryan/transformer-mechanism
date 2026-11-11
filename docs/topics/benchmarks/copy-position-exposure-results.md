# Copying after broader positional exposure

Completed 2026-10-07. **All three spaced-position models copy every reserved length-8 and length-10 example correctly.** Ordinary-position controls average 14.06% and 2.60%, respectively. Both arms fully fit the short training task. The architecture, token streams and update budget are matched.

[Educational guide](copy-position-exposure-guide.md) · [Frozen protocol](../../protocols/copy-position-exposure.md) · [Raw report](../../../results/copy_position_exposure/report.json) · [All metrics](../../../results/copy_position_exposure/metrics.csv) · [Independent audit](../../../results/copy_position_exposure/audit.json)

## What changed and why

Training sequences still contain 3–6 content tokens. The treatment samples sorted position IDs from 0–10, once per batch, using corresponding IDs in the encoder and decoder. Ordinary training uses consecutive IDs. Evaluation uses consecutive IDs in both arms and generates its own answer prefixes.

For rotary attention, `(R_p q)ᵀ(R_r k) = qᵀR_(r−p)k`. Spacing positions exposes attention to larger relative displacements without adding content tokens. This tests a learning-exposure explanation suggested by the earlier competing-read diagnostic.

[Ruoss et al. (2023), section 3](https://arxiv.org/html/2305.16843v1#S3) was inspected for ordered positional sampling per batch and its application to RoPE. Our small range, paired encoder/decoder positions and ordinary test positions differ from that paper's setup. This is a local adaptation of an existing method, not a novel architecture. The equation is established mathematics; the scores below are our measurements.

## Confirmation results

Full-copy accuracy requires the entire generated sequence and correct stopping. “Outranked” means another source position scores above the first source token in either cross-attention head at the first decoding step, with tolerance 1e-10.

| Seed | Length / group | Ordinary copy % | Spaced copy % | Ordinary outranked % | Spaced outranked % |
|---|---|---:|---:|---:|---:|
| 3719 | 8 / distinct | 14.06 | 100 | 65.63 | 0 |
| 3719 | 8 / repeated | 25.00 | 100 | 70.31 | 0 |
| 3719 | 10 / repeated | 4.69 | 100 | 69.53 | 0 |
| 3727 | 8 / distinct | 10.94 | 100 | 53.13 | 0 |
| 3727 | 8 / repeated | 10.94 | 100 | 59.38 | 0 |
| 3727 | 10 / repeated | 2.34 | 100 | 44.53 | 0 |
| 3733 | 8 / distinct | 4.69 | 100 | 76.56 | 0 |
| 3733 | 8 / repeated | 18.75 | 100 | 71.88 | 0 |
| 3733 | 10 / repeated | 0.78 | 100 | 84.38 | 0 |

All nine groups pass both declared improvement thresholds: at least 10 percentage points in copying and in outranking reduction. Every short train, validation and test group reaches 100% in both arms. Across the six pooled long seed/length cells, teacher-forced cross-entropy is 0.6646–1.3739 for controls and 0.000145–0.000181 for treatment. The CSV retains every group and loss, including absent groups.

There are 128 fresh examples per long length: length 8 has 64 distinct and 64 repeated cases; length 10 has only repeated cases. Three seeds share this reserved dataset. These are 256 unique examples, not 768 independent datasets. Short pools are reused. Uniform-content challenges also reach 100% under treatment; ordinary controls already reach 100% at both long lengths in seed 3733.

## Training and checks

The 1,888-parameter model uses width 8, two heads, one encoder/decoder layer and a ReLU FFN. Adam uses learning rate 0.001. Both development arms pass ordinary training fit at 6,000 updates; treatment also passes every fixed-layout training group. The control's weakest layout group is 17.19%. Thus the predeclared 12,000-update fallback is not run.

Each of eight runs executes 6,000 updates, 957,000 sequences and 5.04 million target tokens. Total formal training: **48,000 updates and 40.32 million target tokens**. Paired initialization and token streams match. All confirmation checkpoints were saved before long-test scoring. CPU float64, one thread; recorded training/development-evaluation segments total about 11.9 minutes, excluding final evaluation and audit. GPU memory is not applicable; CPU peak memory was not measured.

Three preflight tests exercise initialized gradients, saved reloads, actual evaluators and deliberate errors. The final audit independently recomputes 264 cells and 39,168 sequence evaluations: zero prediction differences, maximum logit/score discrepancy 2.064e-13. All 17 protected files remain unchanged. It checks streams and optimizer counters, not a replay of every training gradient; projection/attention utilities are shared.

## What this establishes

Changing positional exposure produces a learned full-copy repair under this recipe, alongside the predicted ranking improvement. The earlier failure was not an inability of this architecture to solve these tested lengths.

The training intervention affects several attention computations together. It does not prove that ranking alone mediates the repair or explain the complete optimization trajectory. Test positions 0–10 are inside the sampled training range: this is transfer to more content tokens, not to an unseen positional range. Arbitrary lengths, other tasks and novelty remain unestablished. The declared study is complete; no further sweep is queued.
