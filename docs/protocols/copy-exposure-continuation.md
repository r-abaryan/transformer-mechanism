# Can positional exposure repair an existing weak solution?

2026-10-08. One bounded diagnostic continuation, not a recipe search or fresh confirmation.

## Fixed comparison

- Start both branches from seed 3767 cross-only update 6,000 in `results/copy_learning_trajectory`. Restore identical parameters, Adam moments/step and both random generators. Verify the checkpoint against the persisted replay audit.
- Control: continue cross-only positional exposure. Intervention: use the same sampled ordered positions also in encoder and decoder self-attention. Cross-attention positions, examples, full content/EOS supervision, loss coefficients, architecture and optimizer are unchanged. Both self sites change together; they are not individually localized.
- Run updates 6,001–8,000, Adam lr .001, CPU float64, one thread, no dropout or accumulation. Existing cycle through content lengths 3/4/5/6 uses batches 210/168/140/120, 840 target tokens per update. Per branch: 2,000 updates, 319,000 examples, 1,680,000 supervised targets. Sample the same increasing subset of 0..10 in both branches. No optimizer reset or extra long training examples.
- Budget is one third of the original 6,000 updates, a declared local continuation allowance, not a claim of convergence. Save/evaluate at 6,000/6,500/7,000/7,500/8,000. Intermediate scores are observations only; no early selection, extension, replacement seed or tuning.
- Ordinary evaluation on all frozen train/validation/test lengths 3–6 and test lengths 8/10 from `copy_position_sites/data.json`. No uniform/layout pools. Report full generated accuracy, content exactness, teacher-forced CE and token/content accuracy, and early/on-time/late/missing EOS for all/distinct/repeated groups. Missing groups are explicit nulls. Use contiguous native evaluation positions for both branches. Generation uses its own prefix, stops at first EOS or length+3; no oracle suppression or forced stopping.

## Decision at update 8,000

Intervention repair: >=95% full copying and on-time EOS in each long group and >=99% complete copying in every short train/validation/test group. Exposure-specific benefit additionally requires >=10 percentage-point length-10 full-copy gain over matched continuation. Apply the same absolute repair gate to the control. If both recover without a qualifying gap, extra training suffices under this comparison. If neither recovers, report the bounded failure; do not infer that early exposure is necessary. Always report every group and transient short losses.

## Verification and resources

Before formal training: exercise saved initialized models through the actual evaluator and its reference audit, absent/missing groups, supplied correct/wrong logits and corrupted generations. Compare both exposure paths' outputs and parameter gradients to explicit reference equations; dropping self-site positions must change the intended path. Profile a full step with Adam state for both branches. Record parameter/optimizer bytes and process peak working set (CPU process high-water measurement includes imports/evaluation; not isolated tensor memory). GPU allocated/reserved are inapplicable to this CPU experiment. Keep checkpoint hashes, paired data/position stream hashes and generators; reload all saved evaluations for independent scalar/reference checks. Protect all executed dependencies and prior evidence. Nonfinite losses/gradients stop the run as an implementation/numerical failure, without replacement.

## Source and interpretation

[Ruoss et al. (2023), §3](https://arxiv.org/html/2305.16843v1#S3), inspected for this design, samples ordered position subsets to expose short training sequences to a broader positional range. Our RoPE site choice, shared-checkpoint continuation, contiguous evaluation and thresholds are local choices, not the paper's predicted results. Test position IDs remain inside the training sampling range. A positive result identifies an effect of the exposure intervention from this saved state, not an internal mediator, a novel architecture or reliable performance across seeds.
