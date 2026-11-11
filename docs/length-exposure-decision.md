# Next decision: does limited length exposure help?

Status: executed and closed. The [results](topics/training/length-exposure-results.md) show positive transfer gains but failure of the every-seed 90% criterion. The original design below is retained; the [execution protocol](length-exposure-protocol.md) was frozen before generation/training.

## One question and one change

Does training on three **and four** pairs improve recall on **six** pairs, compared with training only on three pairs?

Keep the architecture unchanged. The intervention is exposure to length four, not a new positional encoding or attention mechanism. This is motivated by the [measured length failure](topics/training/final-length-results.md), whose cause remains unresolved. Mixed-length exposure is a hypothesis to test, not an established cure.

| Item | Fixed design |
|---|---|
| Baseline | Three-pair training only |
| Candidate | Alternate three-pair and four-pair minibatches, 50% of updates each |
| Models | Same 2-block pre-norm, width 16, 2 heads, FFN 32, vocabulary 19, final LayerNorm |
| Pairing | Five seeds: 761, 787, 809, 839, 863; identical initial tensors within each pair |
| Optimizer | Adam .001, no warmup, clipping or schedule changes |
| Budget | 6,000 updates per model, batch 64; ten runs total; no extensions |
| Position IDs | Canonical start +4 for every train/evaluation sequence in both conditions |
| Primary evaluation | Six-pair recall on one reserved set of whole mappings |
| Preservation check | Three-pair recall on a reserved set |
| Secondary evaluation | Four-pair recall, loss and queried-rank accuracy |

Same update/example budget is **not equal FLOPs**. Four-pair inputs contain ten tokens rather than eight, increasing per-step attention and FFN work. Record processed tokens, the sum of batch-size times sequence-length squared, and elapsed training time. Treat this as a capped practical recipe comparison, not an equal-compute architecture comparison. No GPU speed claim.

Use the same three-pair training pool and corresponding stream seeds in both arms. At candidate four-pair updates, the baseline instead receives a three-pair batch. This changes length exposure and the proportion of three-pair examples together; the experiment does not separate those effects.

## Data isolation

Reserve the evaluation mappings before training and open predictions only after all ten attempts finish. Split entire mappings, not order/query variants. Exclude earlier evaluated full mappings of the matching length from the new test sets, and exclude test mappings from training. Record all exclusions and overlap checks. Familiar submappings can remain; state that limitation.

Keep twelve keys, six values and distinct values within each example. Six pairs is the largest length supported by this distinct-value grammar. Use balanced targets and query ranks. Report mapping counts and dependence between variants; do not count variants as independent mappings. Do not claim extrapolation beyond six pairs.

## Predeclared decision

Keep the candidate for this task only if all three conditions hold:

1. Every candidate seed reaches at least 90% on six pairs.
2. Mean paired six-pair accuracy improves by at least 5 percentage points, with no seed worse than its matched baseline.
3. Every candidate seed reaches at least 99% on three pairs, with mean preservation loss no larger than 0.5 percentage points against baseline.

Otherwise reject the candidate under this budget. Report useful partial gains, but do not relabel them success. These are practical descriptive thresholds, not statistical significance or universal superiority claims.

## Finish once

Save all final checkpoints, optimizer state, curves and predictions. Record numerical failures as failures. Verify split isolation, paired initialization, implementation controls and reloaded metrics. Use final checkpoints only. Validation diagnoses execution; it does not select recipes or stopping times.

Stop after the ten attempts and audit. No learning-rate search, extra training seeds, larger models, activation rescue, or reopening the test set for tuning. A demonstrated code defect may justify a documented correction; an unfavorable result does not.
