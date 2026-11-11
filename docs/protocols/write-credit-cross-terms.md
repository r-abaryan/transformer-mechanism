# Signed temporal cross terms

Supplement declared after opening the primary [write-credit diagnostic](write-credit-assignment.md), before calculating these statistics. Its norm-of-sum ratio conflates directional spread with opposition. Question: do position-pooled value gradients have negative net cross terms?

Read only the audited `results/write_credit_assignment/gradients.pt`: two seeds, correct/error/total, 32 position vectors of 48 shared-value coordinates. No new model evaluation, labels, selection, backward pass or optimizer call. Preserve primary sources and results.

For position vectors D_t, report E=sum_t ||D_t||^2, J=||sum_t D_t||^2 and C=J-E=2 sum_(i<j) D_i dot D_j. Negative C establishes net directional opposition; a small primary ratio alone does not. J/E is descriptive, not an accuracy effect or causal attribution. Also report counts of negative/positive/unresolved pair dots under an absolute 1e-14 guard. Six trajectories contain 2976 correlated pairs, not independent experiments.

Verify the identity independently using explicit nested scalar pair products, absolute+relative 1e-12 tolerance. Exercise opposing, orthogonal and aligned controls before reading formal gradients. Bind input/report/audit/script/protocol hashes; stop after one supplemental reduction. No new training follows automatically.

These statistics and their identity are local linear algebra, not a published trained mechanism. The underlying recurrence source is [Yang et al., section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2); see the primary diagnostic for source inspection and scope.
