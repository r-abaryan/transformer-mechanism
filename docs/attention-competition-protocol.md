# Attention competition: bounded mathematical probe

Freeze before execution. Distinguish implementation results from attention-function properties. No retraining, new checkpoints, intervention sweep or architecture proposal. Review existing distractor/subset results; they cannot isolate softmax dilution.

Use the existing standard-library softmax implementation. One visible target and m visible distractors. Test m in {1,4,16,64,256,1024} and equal target-minus-distractor score gaps in {0,2,4,8}: 24 deterministic fixtures. Compare direct softmax with p=1/(1+m*exp(-gap)) at absolute tolerance 1e-12. Check p decreases with m at fixed gap. Target rank alone is not routing fidelity: for positive gap the target is highest-scoring, but its mass can be small.

Test target mass .9 using gap=log(m)+log(9), six fixtures. Add three unequal-score fixtures to verify the exact summed formula, min/max gap bounds and that mean gap alone is insufficient. These are arithmetic checks, not independent statistical trials.

At m=16 and gap=2, hold attention fixed and test three scalar value sets: target and distractors all 1; target 1/distractors 0; target 1/distractors -1. Verify output error equals (1-p)*abs(weighted_distractor_mean-target). Also use half 0/half 2 distractors to test cancellation with distinct values. Test the bound error <= (1-p)*max_j(abs(v_j-v_target)). This concerns one head's output, not class accuracy or a full transformer's residual/readout path.

Independently audit weights/output using decimal arithmetic at 50-digit precision, verify source/protocol/result hashes and predefined fixture coverage. Preserve any failed check; stop after these fixtures. No novelty or general trained-model capacity claim.
