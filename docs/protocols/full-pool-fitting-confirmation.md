# Fresh confirmation of unchanged-reader fitting

Locked settings, 2026-10-05. Formal training starts only after passing the final runtime preflight and complete-step profiles. Question: **does the successful fitting procedure repeat with fresh initialization seeds and new history families?** This confirms finite fitting, not a learning cause or architecture winner.

## Data and warm starts

Initialization seeds 3473/3479/3481 were chosen before scores. Data seed 2026100517 generates one shared fresh dataset. Exclude cyclic-value families from association-load development/confirmation, query-coverage development and query-feedback test data. Save and hash the data before training. Each load 2/8/16/32 has 128/32/64 base histories for train/validation/test, each expanded by six cyclic value shifts: 768/192/384 histories. Family splits are disjoint; shifts are related examples, not independent trials.

Keep the 615-parameter reader, 16-by-8 memory, normal zero-start writes, tied normalized Q/K, scalar beta and six values/classes. Histories have 32 events and constant values per key. Loads 2/8/16 repeat every key; load 32 writes each once. Constant and last-value baselines are recorded. First matching value always succeeds by construction, so this is not a value-changing overwrite or algorithm-identification test.

Build each warm start from the original initialization function and exactly 12,000 exhaustive-query Adam updates. Batch 64; cycle loads 2/8/16/32; lr=.003, betas=.9/.999, eps=1e-8, foreach=False, no decay/schedule/clipping. History/query RNG seeds are initialization seed +10,000/+20,000. Preserve initial/final weights, moments, counters, every gradient and update. Evaluate full training at initialization and warm-up end. Do not stop, skip, select or replace a seed based on warm-up fit.

## Locked fitting comparison

From identical warm-start weights per seed, reuse the [development procedures](full-pool-fitting.md) unchanged. L-BFGS begins with empty curvature history; Adam retains all warm-up moments/counters and the exact continuation streams.

L-BFGS: equal weight for each load's exhaustive training CE, lr=1, max_iter=1, history_size=20, tolerance_grad=1e-10, tolerance_change=1e-12, strong_wolfe. Cap at 128 full-gradient closures, including trials. Set max_eval=min(26,remaining-1); stop when fewer than two evaluations remain. Adam uses exactly 48 added updates per actual closure, matching gradient-target, history and event exposure separately per load. This is not equal FLOPs or an isolated optimizer/batching/history intervention.

One full gradient sees 3,072 histories and 44,544 answers. Twelve four-load Adam cycles see the same counts. Warm-up totals 36,000 updates / 33,408,000 answer targets. Fitting totals at most 34,209,792 targets across six branches. Combined maximum: 67,617,792 targets, 4,663,296 history exposures and 149,225,472 events. Count warm-up once per seed, separately from branched fitting and audit replays.

CPU float64, deterministic, one thread per worker. Three independent seeds may run concurrently in separate native Windows processes, with distinct artifact names. Report workers, wall time and per-run times; no speed-benchmark claim. Profile complete Adam/L-BFGS steps including allocated optimizer state before training. GPU unused, allocated/reserved VRAM null; CPU peak memory unmeasured.

## Criteria and evaluation

Primary confirmation: final accuracy >=99% in every nonempty training load/group in **all three seeds**, separately per procedure. Secondary: 100% full-training accuracy and complete maps in every seed. Record failures and weak groups; an average cannot pass. Gates apply to final checkpoints only, with no early stopping or best-snapshot restoration.

Save full training at warm-up start/end, fitting start, first L-BFGS accepted state crossing each 32-closure milestone, corresponding 1,536 Adam-update milestones, and final states. Report accuracy, CE and complete maps. Save fitting decisions before opening validation/test scores. Then evaluate all held-out pools at each final fitting checkpoint. These histories are fresh but at the same length; no longer-input transfer measurement. Held-out scores are descriptive, excluded from all selection and fitting decisions. Seeds share one dataset, so confirmation samples initialization/stream variability on one fresh pool, not variability across independent datasets.

## Verification and stopping

Exercise the actual saved initialized warm-up/fitting/evaluator/audit path, including three concurrent fixture seeds, supplied ties/absent groups, missing cells/loads and deliberate label/logit/gradient/source-step/count/decision/family-leakage faults. Preserve the earlier passing serial preflight/source snapshot and the parallel preflight blocked by sandbox pipe permissions. Neither was formal training. Run the identical final process preflight with native pipe permission; do not change scores, recipe or tolerances to resolve permission errors.

Reuse independent labels, matrix gradients, scalar Adam updates/moments/streams and L-BFGS two-loop/displacement checks. Verify every warm-up and fitting step, exact initialization and both branches' warm-start lineage, all snapshots/cells/counts and protected evidence. L-BFGS trial replay shares the installed engine with the producer; independently verified gradients/directions do not constitute a second optimizer implementation.

Stop after three warm-ups, six fitting branches, audit and concise documentation. No new settings, extra closures, replacement seeds, architecture changes or automatic causal study. If confirmation fails, retain finite-pool witnesses and report that repeatability failed at this budget. If it passes, a later causal comparison still needs predeclared competing predictions. RL stays parked.

Sources inspected: [PyTorch 2.11 L-BFGS](https://docs.pytorch.org/docs/2.11/generated/torch.optim.LBFGS.html), closure/history/tolerances, plus installed step/strong-Wolfe source already bound by the development audit; [Yang et al. section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2), delta writes/reads. Fresh-family separation, warm-start construction, count matching and confirmation gates are local design choices; neither source predicts local scores.
