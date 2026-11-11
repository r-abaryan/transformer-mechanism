# Learning positional robustness in marked copying

Frozen 2026-10-07, after a complete-step CPU profile and initialized preflight. Hypothesis: narrow positional-displacement exposure permits poor source ranking on longer inputs. A matched training intervention should reduce first-source outranking AND improve complete copying. The [saved-score review](../assessments/copy-learning-hypothesis.md) motivates this prediction; it does not establish its cause.

## Intervention and primary source

Keep the 1,888-parameter marked RoPE encoder-decoder, short content lengths 3/4/5/6, labels and all learned operations unchanged. Baseline uses contiguous position IDs. Treatment draws n+1 distinct IDs uniformly from integers 0..10, sorts them, and uses the same IDs for source content+EOS and corresponding decoder rows BOS+content. One draw per batch, independent of the token-sampling generator. All self/cross Q/K rotations use these IDs; causal masking follows sequence order.

Evaluation uses ordinary contiguous positions. Fixed spaced layouts are additional training-fit probes, not the primary test. For generation under these probes, positions after the supplied vector extend by unit increments. Ordinary generation has all 16 classes, own-prefix feedback, EOS stopping and cap n+3. No oracle attention or attention-target loss is used.

[Ruoss et al. (2023), section 3](https://arxiv.org/html/2305.16843v1#S3) was inspected for ordered sampling without replacement, once per batch. This is an adaptation: we use a small range covering the declared long positions, pair source/decoder IDs, and evaluate contiguously; the paper also randomizes test positions and considers a larger range. No reproduction or novelty claim. The intervention changes positional exposure and activations, not just optimization.

## Data and bounded recipe development

Reuse original short train/validation/test pools from `results/copy_marker_training/report.json`; both arms get identical token batches and initial tensors. Original short splits remain opened and are not fresh confirmation. Keep uniform probes at lengths 3/4/5/6/8/10, with their historical overlap limitations.

Reserve new random long test pools before training: 128 sequences each at lengths 8 and 10, generator seed 2026100711, excluding all exact sequences in the marker report and duplicates. Length 8 has 64 distinct and 64 repeated examples; length 10 has only repeated examples. These are fresh long inputs under an already selected task distribution, not an untouched overall benchmark.

One development pair: seed 3711. Candidate budgets are 6,000 and 12,000 updates, with the second continuing BOTH arms only if the first pair fails the training gate. No optimizer, position-range or seed alternatives. Select the first budget at which both arms pass. Adam lr .001, betas .9/.999, epsilon 1e-8, no decay/clipping/schedule. Freeze all candidates now; never select on long tests.

Gate: >=99% ordinary full-copy accuracy in every full train3/4/5/6 distinct/repeated group under contiguous positions, for both arms. Treatment must also pass the same gate on four fixed ordered layouts per length (generator seed 2026100712), applied to every training sequence. This is a declared representative positional subset, not exhaustive evaluation of every possible layout. Baseline spaced-probe failures are reported but do not gate its native training fit. Validation3/4/5/6 losses and accuracies are recorded without selection. Evaluate only these training/validation and layout cells at development endpoints; no long scores.

If both budgets fail, stop with a learning-adequacy failure and leave long tests unused. Otherwise lock the chosen budget and train three new paired seeds, 3719/3727/3733, from scratch. Save all six final checkpoints and a pretest record before any long-test scoring. Do not replace confirmation seeds or extend their budget if they fail fitting.

## Compute and measurements

Native Windows CPU float64, deterministic, one thread. Profiled complete steps include backward and Adam state; 24 discarded profile updates. Steady measured times are approximately .024-.028 s/step; Adam tensor state is 30,340 bytes. GPU unused; CPU peak memory unmeasured, GPU allocated/reserved peaks null. Evaluation batch at most 64. No inference about deployment speed.

Cycle lengths 3/4/5/6 with batches 210/168/140/120, balanced distinct/repeated. Every update has 840 supervised tokens including one EOS per sequence, plus 840 source tokens including EOS. At 6,000 updates: 957,000 sequences and 5,040,000 supervised tokens per run; 12,000 doubles these. Accumulation is one batch. One new position draw does not add examples or labels. Record every development attempt, minibatch loss every 1,000 updates, checkpoints, RNG states and stream hashes. No convergence claim from budget completion.

Final evaluation records full ordinary exactness, generated content exactness, teacher CE/token/content accuracy, EOS rates, first-token accuracy, fraction with position zero outranked in either head, and near-tie fraction (gap tolerance 1e-10). Include every seed/pool/group; absent groups are null. Metrics use all output classes. Uniform probes are descriptive. Full-copy failure and attention rank are distinct quantities.

Primary confirmation criterion: in every seed and each present test8/test10 distinct/repeated group, treatment gains >=10 percentage points of full-copy accuracy AND reduces the either-head outranked fraction by >=10 points. Both training-fit gates must pass in every pair; no short-test group may lose more than 5 points. These are practical thresholds, not significance tests. Record paired differences even when the combined rule fails. Better copying without ranking improvement does not support this particular prediction; better rankings alone do not establish a sufficient repair. A positive outcome still does not prove unique causal mediation or arbitrary-length generalization.

## Verification, counts and stop

Preflight: initialized native identity, sampled-position output/all-parameter-gradient equations, shared translation invariance, positional-order error, label shifting, supplied known logits/ties/absent groups, saved initialized training/evaluator/audit, and deliberate metric/missing-cell/stream/selection errors. Unique fixtures use four updates. Independent audit reconstructs explicit rotation/normalization/softmax paths, greedy outputs and scalar metrics; shared attention/projection utilities limit independence.

Development has 24 cells per run: 8 train/validation pools plus 16 layout probes. Confirmation has 36 per run: 20 ordinary pools plus 16 layout probes. One selected first-stage pair plus six confirmation runs gives 264 final audited cells; a second development stage gives 312. If development stops without confirmation, audit 96 cells. Each cell includes all applicable groups; sequence answers, positions and repeated probes are not independent seeds.

Audit initialization, checkpoint hashes, Adam counters, token/position streams, split disjointness, selection, all final stored logits/rank scores and generated outputs. Require <=1e-9 numerical error and zero prediction differences. This does not replay every training gradient/update. Relevant frozen sources, protocol, data, profile, preflight and historical reports are hashed before/after. Stop after the declared development/confirmation, audit and short educational report. No automatic tuning or architecture change follows failure.
