# Does independent reset exposure improve reset handling?

Frozen 2026-10-03 after full-step profiling and before training. Six fresh runs: three matched pairs, seeds 3011/3019/3023. This is a bounded exposure diagnostic, not confirmation of a previously eligible recipe. Earlier development selected none.

## Fixed setup and intervention

Unchanged 4,080-parameter MQA Transformer, two blocks, width 16, two Q/one KV head, RoPE, ReLU FFN32, original normalization and final-query CE. CPU float64, one thread. Adam peak .001, 200-update linear warmup then cosine to .0001 at update 6,000. No clipping, weight decay, restarts or intermediate supervision. Batch 64, alternate 8/12 operations. Each run: 384,000 final targets/examples and 4,608,000 input tokens. Profile seed 3001 is discarded; full step allocates 36 Adam states/65,424 tensor bytes. GPU allocated/reserved memory is zero; CPU peak unmeasured.

Both arms sample identical indices from the original 1,920 training examples, with matched full initial tensors and separate identical augmentation coin streams. In the correlated arm keep every input unchanged. In the independent arm, flip the initial bit with probability .5 ONLY for sampled long-tail examples. The first SET makes the answer invariant to this change, so labels and loss weighting are identical. Consume the same coin draws in both arms. Record all initial/reset combination counts. Mixed/no-reset examples stay unchanged. Rotate arm execution order across seeds.

This changes exposure and the available initial-bit variants. Total examples, tokens, suffix templates, reset values and training objective match; it does not hold the count of distinct full inputs fixed.

## Evaluation and splits

Freeze all pools in `results/state_reset_exposure/data.json` before training. Full training evaluation covers original no-reset/mixed examples and BOTH initial values for each original long-tail suffix/reset pair: 2,560 inputs. For the correlated arm, its training-support fit gate excludes disagreeing initial/reset cases it never receives. The independent arm's gate includes them. Report agree/disagree scores separately, not only their mean.

Long-tail validation/test use disjoint suffix families within each length, excluding all training suffixes. At length 8 use 16 validation/32 test families; length 12 uses 64/128; length 24 uses 128 test families. Each suffix has all four initial/reset combinations, two agreeing and two disagreeing. No longer-length training. Validation/test no-reset/mixed pools at lengths 8/12 reuse historical pools as guard measurements. Historical symbols/templates and some held-out inputs have already been inspected in earlier studies. These are held out from these six training runs, not an untouched project-wide confirmation set. No held-out selection or tuning.

Evaluate all training cells every 1,000 steps. Evaluate final training/validation/test accuracy and CE together at step 6,000 only. Report paired initial-bit class-change rate, both-original-and-counterfactual-correct rate, and mean probability difference for every long-tail split/length. Constant classifiers score 50%; the independent oracle scores 100%. A wrong initial-plus-suffix-parity rule must fail on disagreeing reset examples.

## Criteria and stopping rule

For each seed separately require >=99% fit in EVERY training-support cell in BOTH arms before an adequacy-qualified adoption claim. Behavioral criterion: independent exposure gains >=20 percentage points over the correlated arm on twelve-operation test disagree accuracy AND independent both-correct accuracy across initial variants is >=90%. Adoption requires both criteria in every seed. Report behavioral changes even when fit eligibility fails, without claiming complete learning or isolating an internal mechanism. Length-24 results are descriptive transfer evidence and do not select recipes.

No parameter selection, best-checkpoint selection, extra updates, replacement seeds or enlarged grid. Failed/nonfinite attempts remain visible. Stop after six attempts, audit and one short educational report.

## Checks

Verify every label with independent last-SET/parity and recurrent rules. Verify disjoint suffix splits, paired label invariance and balanced groups; detect the wrong initial-base shortcut. Three initialization output/gradient cases must agree within 1e-10, and wrong-label gradients differ >1e-6. Every final logit/argmax must match independent full-model equations; independent Python CE/accuracy and paired-reset metrics agree within 1e-10. Replay index and coin streams, initial tensors, optimizer update counts/final rate and exposure counts. Preserve targeted historical sources/reports/checkpoints. Arithmetic checks establish execution correctness within scope, not general superiority.
