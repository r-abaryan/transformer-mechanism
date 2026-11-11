# Copying: one paired training-length exposure comparison

Frozen 2026-10-03, after discarded complete-step profiling and before data generation or model evaluation. Question: does replacing some short-copy training with longer examples improve ordinary copying and extrapolation? This changes the training distribution, not the architecture. Previous copy runs remain unchanged.

## Fixed recipe and budget

Six fresh runs: seeds 3061/3067/3079, each with short (cycle 3,4) and mixed (cycle 3,4,5,6) arms. Same initial parameter values within each pair. Reuse the audited 1,888-parameter width-8, two-head, one-layer post-norm encoder/decoder, ReLU FFN16, sinusoidal positions, dropout0 and all 16 output classes. Adam .001, betas .9/.999, epsilon 1e-8, no schedule, clipping or weight decay. CPU float64, one thread, deterministic native Windows; GPU unused, GPU memory unmeasured. Profile includes backward and Adam-state allocation; profile updates are discarded.

Stop at 6,000 updates per run, or preserve a nonfinite failure and stop the comparison. No candidate grid, best-checkpoint selection, replacement seeds or extension. Each update supervises 840 target tokens including EOS; batch sizes at lengths 3/4/5/6 are 210/168/140/120. Both arms see 5,040,000 target tokens. Short sees 1,134,000 sequences; mixed sees 957,000. Longer examples replace short exposure. EOS target frequency and attention work also differ; this does not isolate positional diversity or establish equal compute.

Draw 210 uniform ranks per update from generator seed+1000, using the relevant prefix of ranks for each balanced distinct/repeated half-batch. Both arms share the rank stream. At updates with matching lengths, indices are identical. Record rank/index hashes, sequence/source/target counts, visible attention-pair counts and wall time including declared evaluations/saves.

## Data and evaluation

Generator seed 2026100306. Content symbols 3..10; PAD0/BOS1/EOS2. Correct target is source followed by EOS; decoder input is BOS followed by source. Exclude all previously inspected original copy sequences. New exact sequences are disjoint across all pools; symbols/subsequences are shared. Limit proposals to 100,000 per group, with no changes on exhaustion.

| Length | Training pool | Validation | Test | Groups |
|---|---:|---:|---:|---|
| 3 | 64 | 16 | 16 | Equal distinct/repeated halves |
| 4,5,6 each | 256 | 64 | 128 | Equal distinct/repeated halves |
| 8 | None | None | 128 | Equal distinct/repeated halves |
| 10 | None | None | 128 | Repeated only |

Only short's 3/4 pools are its training support; mixed uses all four. Length-three counts are limited by unused repeated inputs in the eight-symbol vocabulary. Each held-out group there has only eight examples. Ten distinct symbols are impossible with this vocabulary; do not invent that group.

Evaluate full arm-specific training and all validation pools at 1,000/3,000/6,000 updates. No validation-based selection. Save all six final checkpoints and a pretest report before any model test scoring. Then evaluate full pools, saving teacher logits and greedy generations. Generation has no attention override or EOS blocking: first EOS stops output, with source length+3 token cap. The cap bounds evaluation; it does not supply the correct output length. Report exact content/full sequence accuracy, teacher token/content accuracy, CE including EOS, and early/on-time/late/missing EOS per seed/group. Include copy oracle, repeat-first, reverse and sorted baselines.

## Criteria, checks and stopping

Training fit: >=99% exact generation in every arm-specific training group for every seed. Seen-length reliability: fit plus >=95% in every corresponding test group. Broader-support benefit: both fit gates, >=10-point paired balanced test6 gain in every seed, and no >5-point test3/4 group regression. Mixed length8 and length10 extrapolation: mixed seen-length reliability plus >=90% in every eligible seed/group, scored separately. Length6 is seen-length generalization for mixed; never call its improvement extrapolation. Keep descriptive results even if gates fail. These fixed criteria do not select a new architecture or justify tuning.

Independently verify labels/shifting/splits and reverse-label/controller errors. Compare first native gradients/logits with independent model equations; verify finite losses/gradients throughout. Reconstruct every final teacher logit and generated sequence from checkpoints using independent equations/controller: <=1e-9 discrepancies, zero argmax/sequence changes. Recompute scalar metrics, rank/index streams, optimizer counters, exposure counts and criteria. Include a deliberate criterion-error control. Hash relevant dependencies before/after; no all-project hash sweep.

Stop after six runs, the independent audit and one short educational report. An incomplete fit prevents clean generalization attribution. A gain without all-seed reliability remains a measured gain. No new sweep, architecture change or automatic follow-up.
