# Terminal feedback changes the learned failure, but removing it loses stopping

2026-10-08. **Removing terminal-EOS supervision during updates 501–1,000 restores the weak model's fifth-token answers and reduces its marker-attention mismatch. It also destroys proper stopping.** All three diagnostic signatures pass; this is not a complete copying repair.

[Frozen protocol](../../protocols/copy-eos-development.md) · [Report and audit](../../../results/copy_eos_development/report.json) · [All pools, groups and losses](../../../results/copy_eos_development/metrics.csv)

## One training change, from identical saved states

For each of four runs, restore update-500 weights, Adam moments/step and sampling generators. Continue 500 updates with either the original objective or terminal-EOS loss removed. Batches, positions, optimizer settings and content-loss coefficients match. Every native branch exactly reproduces its historical update-1,000 checkpoint, including optimizer and generator states.

For content length n, the changed objective is:

`L_alpha = (sum(content CE) + alpha * sum(terminal EOS CE)) / [B(n+1)]`, with alpha 1 or 0.

The denominator stays fixed. EOS remains an output class: content CE still penalizes premature EOS. Earlier EOS-trained weights and optimizer history remain. This tests dependence on terminal feedback during this interval, not absence of EOS throughout learning.

The older EOS experiment used sinusoidal models trained from initialization and asked about content extrapolation under oracle length. Here we investigate the observed developmental interval in marked RoPE models and evaluate ordinary generation. The historical failure remains valid.

## Weak-seed signatures

Paired answers use the same 128 opened short/long sources and correct prefix at position 5. Attention gap is the absolute difference between mean short and long head-0 marker mass.

| Measure | Native continuation | No terminal target |
|---|---:|---:|
| Long fifth-token correct | 2/128 | 128/128 |
| Long fifth-token EOS | 126/128 | 0/128 |
| Short marker mass | 99.73% | 42.97% |
| Long marker mass | 19.22% | 18.53% |
| Marker-attention gap | 0.805 | 0.244 |
| Fifth-token CE | 4.079 | 0.028 |

The gap shrinks mainly because short-source marker concentration falls, not because long-source marker reading rises. This is a different route from the earlier supplied-pattern intervention, which restored the short pattern on long inputs.

The >=50-point answer gain/EOS reduction rule passes, the gap falls below half its native value, and every ordinary short train/validation/test group preserves content token accuracy within five points. These criteria were frozen before continuation. They do not require preserved stopping, which is reported separately below.

## Ordinary generation exposes the cost

Counts below are out of 128 length-10 test sequences. Content exactness means the first ten generated symbols match; full copying additionally requires the correct EOS. No EOS blocking or supplied content was used. Generation stops at EOS or the fixed n+3 limit.

| Model | Policy | Fifth-token correct | Generated content exact | Full copying | Missing EOS |
|---|---|---:|---:|---:|---:|
| Cross-only 3767 | Native | 2 | 1 | 1 | 0 |
| Cross-only 3767 | No terminal | 128 | 127 | 0 | 128 |
| Cross-only 3779 | Native | 128 | 128 | 128 | 0 |
| Cross-only 3779 | No terminal | 128 | 128 | 0 | 128 |
| Cross-only 3793 | Native | 124 | 124 | 124 | 0 |
| Cross-only 3793 | No terminal | 128 | 127 | 0 | 128 |
| Combined 3767 | Native | 128 | 126 | 126 | 0 |
| Combined 3767 | No terminal | 128 | 118 | 0 | 128 |

The combined control also loses eight correct content sequences in aggregate. Fifth-token success therefore does not imply reliable longer content. Short test6 full copying collapses from 125/128, 128/128, 128/128 and 128/128 to 0/128, 1/128, 0/128 and 0/128 respectively. Missing EOS means none was emitted within the limit, not that EOS can never be emitted.

Weak-seed ordinary test6 evaluation CE increases from 0.033 to 0.830 and test10 CE from 0.500 to 0.562. These common evaluation losses include terminal targets; reduced content errors do not remove the cost of failed stopping. All short training, validation, test, layout and uniform groups remain in the report and CSV.

## What is now supported

Unlike endpoint swaps, this changes one objective term during actual matched learning. Terminal feedback in this interval affects both the weak model's premature-EOS behavior and its head-0 marker pattern, conditional on the shared earlier state. This is evidence of training dependence.

It does not establish that the attention change mediates the answer change: classifier, values and other representations also evolve. Removing the target also reduces active supervision and stopping capability. The three successful controls show that terminal supervision does not universally cause premature EOS. No architecture-level or novel-method claim follows.

The next causal question, if pursued, must preserve stopping while separating the attention-pattern effect from other learned changes. Do not select a new EOS weight or budget from these scores. This bounded continuation is complete.

## Checks, budget and source

Eight runs, 4,000 updates, 638,000 sequences and 3.36M computed target rows. Native branches supervise 1.68M targets; omission branches 1.361M, for 3.041M active targets. Each update computes 840 target rows with length-dependent batch sizes 210/168/140/120. CPU float64, one thread, Adam .001, no accumulation. Original complete-step profiling is reused; initialized continuation steps also record time and optimizer storage. CPU peak memory is unmeasured; GPU allocation/reservation is inapplicable.

Initialized saved continuation/evaluation and independent audit paths pass. Content/EOS gradient decomposition, unchanged content coefficients, zero omitted terminal gradient, wrong-denominator and missing-group checks pass. All four native full checkpoints reproduce exactly; streams match within every pair. Ordinary logits, generated sequences, losses and paired outputs pass reference/scalar checks, maximum discrepancy 1.040e-13. All 92 protected files remain unchanged. Shared forward/optimizer helpers limit independence. Opened seeds, pools and selected interval limit generalization.

[Newman et al. (2020), sections 1 and 3](https://arxiv.org/html/2010.07174v1#S3), inspected in earlier copy work, motivates separating termination and content. The loss weighting follows the preserved earlier local EOS study. The restored-state intervention and results here are local evidence, not predictions or a replication of that paper's scores.
