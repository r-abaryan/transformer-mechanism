# Saved routing/decoder transport review

Frozen before transport measurements, 2026-10-05. Question: do archived L-BFGS directions contain transport between the routing and decoding blocks, and do the archives support a controlled removal of that transport?

Inspect all six direction-study traces, three seeds 3473/3479/3481, without choosing favorable steps. Metadata already inspected: each L-BFGS trace has 63 optimizer calls; each negative-gradient trace has 39. The archive contains 766 gradient evaluations and 306 calls. No new model forward, gradient, optimizer step, loss probe or held-out selection is allowed.

Use the previous partition: R = write_logit and key.weight (513 parameters, tied queries included); D = value.weight, output.weight and output.bias (102). At each L-BFGS call, apply its stored inverse-curvature operator H separately to the masked gradients. Report native direction, within-block direction and cross-block direction:

$$d=-Hg=d_{within}+d_{cross},$$
$$d_{within}=-(P_RHP_R+P_DHP_D)g,\qquad d_{cross}=-(P_RHP_D+P_DHP_R)g.$$

Report every call's cross/total direction norm ratio, gradient projections and cosine between native and within-block directions. Ratios can exceed one through cancellation. The two cross projections should agree by symmetry; they need not help descent. The first call has H=I and must have zero transport. Negative-gradient directions must match -g; their stored curvature bookkeeping is unused by that direction rule.

These are descriptive measurements of archived updates, not a benefit criterion or evidence that transport caused successful fitting. The operator approximates inverse curvature in raw parameter coordinates; it is not an exact neural Hessian or a scale-free effective decoder.

Reconstruct all 189 L-BFGS directions, use a separate dense inverse-BFGS recurrence to audit the masked contributions, and check all 306 stored displacements/directions and source hashes. Reuse the existing two-loop implementation. Before formal analysis, exercise the saved-trace producer/audit on a small initialized synthetic fixture and deliberate direction, partition, missing-step and metric faults. Keep arithmetic in reusable tests. Stop after review and audit; no training extension.

Inspected [PyTorch 2.11 L-BFGS source](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/lbfgs.py), `step` curvature updates and two-loop recursion, and [Liu/Nocedal (1989), author abstract](https://users.iems.northwestern.edu/~nocedal/Abstracts/limited-memory.html) for limited-memory scope. The block decomposition and possible intervention are local derivations. Published convex convergence results do not establish convergence or predict our reader's scores.
