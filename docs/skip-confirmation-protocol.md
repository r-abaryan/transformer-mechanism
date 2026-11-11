# Skip correction confirmation protocol

Fixed before execution: five original checkpoints, lengths 8/10/12/16, 512 unique examples per length, data seeds 20261010+length. Exclude both previously inspected length-ten sets (512 and 1,024 examples). No retraining or tuning. Query position equals symbol length L.

Set delta=PE(6)-PE(L). Compare original, skip-only b=u+delta+f(u), and full residual b=u+delta+f(u+delta). Here u is the original post-attention residual and f is the normalized FFN. Attention remains fixed.

Direction control: generate one Gaussian vector with seed 20261020+L, remove its projection onto delta, normalize, and multiply by norm(delta). Apply it to the skip path only. The direction is shared across models at each length. Check orthogonality and equal norm numerically. One control direction is limited evidence, not a distribution of random interventions.

Primary confirmation: skip-only accuracy at least 95% for every seed at every length. Specificity criterion: at each length, mean paired accuracy gain from the positional correction over the direction control is at least 5 percentage points. Report both criteria separately; failure is retained. Report original, skip-only, full-residual and control accuracy/loss plus corrected/broken counts, all seed/length combinations. These are project thresholds, not significance tests.

Verify captured forward reconstruction within 1e-10, checkpoint hashes, no overlap and perturbation norms before interpreting outcomes. Freeze protocol/source/data hashes and record runtime versions. No post-result donor selection or repeated trial follows a failure.
