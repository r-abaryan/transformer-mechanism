# Does clearing the saved first moment help continued learning?

Frozen 2026-10-03 after complete-step profiling and before training. This is training-only recipe development following the one-step Adam diagnostic. Six continuations from independent-exposure checkpoints 3011/3019/3023: retain optimizer state versus clear only its first moment once. These are three paired checkpoint-conditioned comparisons, not fresh training seeds or confirmation runs. Preserve every parent artifact.

## Fixed intervention and budget

Unchanged 4,080-parameter two-block Transformer, CPU float64, one thread, batch 64. Both arms inherit identical model weights, second moments, step 6000 and both generator states. Treatment sets exp_avg to zero before its first additional update only. Keep betas .9/.999, epsilon 1e-8, no clipping/decay or supervision changes. Momentum then rebuilds normally. Rotate execution order across seeds.

Add exactly 2,000 updates at constant .0001, the saved final learning rate. Both arms use identical subsequent sampled indices and augmentation coins from the saved generator states. Alternate 8/12 operations, starting with 8. Independently flip sampled long-tail initial bits with probability .5; labels remain unchanged after SET. Reset-free/mixed inputs remain unchanged. No batch/objective/capacity change. The original cosine schedule ends at update 6000; this new continuation policy is explicit, not an implicit extension of that schedule.

Per arm: 128,000 additional examples/final targets and 1,536,000 input tokens; total history 8,000 updates, 512,000 targets and 6,144,000 input tokens. Profile two discarded full optimizer steps on copies before freezing: 36 state entries/65,424 tensor bytes each. CPU peak memory is unmeasured, GPU peak allocated/reserved is zero. Profile timings include startup and are not an arm speed comparison.

## Training measurements and selection

Use only frozen original training-base pools and all eight independent-exposure support pools (2,560 inputs). Evaluate/save full-pool logits at added updates 0/1/50/200/500/1000/2000. These points describe trajectories; only the final point selects. Report every seed/group's accuracy and CE, worst-group accuracy and worst-cell CE, weighted cycle CE, length-12 agree/disagree behavior, paired initial-bit class changes/both-correct/probability sensitivity and exact exposure counts. Constant classifiers score 50%; the label oracle scores 100% on these balanced pools. Historical parent checkpoints are baseline observations; the concurrently continued retained arm is the paired control.

A continuation recipe is eligible only if every training-support cell is >=99% accurate in all three seeds. Among eligible arms, select the smallest worst-cell CE; retained wins an exact tie. If neither qualifies, select none. Report gains/regressions and the failed gate without relaxing thresholds. A selected continuation is a development candidate, not generalization evidence.

Validation/test scores are not evaluated in this stage and cannot select its recipe. If eligibility passes, lock the full parent-plus-continuation recipe and prepare a separate confirmation protocol with fresh initialization seeds/data streams and reserved suffix-disjoint held-out examples. That confirmation must measure full training, short held-out and longer-input accuracy/CE/reset-pair behavior together before any transfer claim. Previously opened pools cannot be called untouched confirmation. No confirmation or new tuning is automatically run by this stage.

## Verification and stopping

Verify labels independently by recurrent and last-SET/parity equations, including replayed augmented batches. Verify matched initial model hashes, retained second moment/step settings, generator replay, all exposure counts, update counters and final LR. On each arm's first actual sampled step, native/reference gradients agree within 1e-10 and wrong-label gradients differ >1e-6. Adam's complete parameter/moment update agrees with independent scalar equations within 1e-12.

Verify every final logit using independent normalization/RoPE/causal SDPA within 1e-10, zero argmax differences. Independently reduce saved logits at all seven observations for CE/accuracy/reset-pair metrics within 1e-10. Intermediate forward equations are not separately replayed from saved weights, so do not claim that coverage. Confirm profile/data/source/checkpoint hashes; write new checkpoints/results separately.

Stop after six fixed-budget attempts, audit and one short educational report. Nonfinite/failed attempts remain recorded; no replacement seed, best-checkpoint selection, schedule change or extra budget. A one-time first-moment intervention can change trajectories through subsequent gradients; after 200 updates the direct old-m coefficient is .9^200, about 7e-10. Any later difference is not evidence that the original moment is still directly present. The result cannot establish convergence, historical causation, architectural impossibility or longer-input repair.
