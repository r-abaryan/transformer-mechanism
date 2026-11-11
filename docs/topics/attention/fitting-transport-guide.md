# Removing transport between parameter blocks

The [experiment](fitting-transport-results.md) changes an optimizer direction in the same reader. The attention/memory architecture stays the same.

## What the two directions do

Let g be the training-loss gradient. L-BFGS builds an inverse-curvature approximation H from recent changes in parameters and gradients, then uses d=-Hg.

Routing R includes keys, tied queries and write strength. Decoding D includes value embeddings and the classifier:

$$d_R=-H_{RR}g_R-H_{RD}g_D,\qquad d_D=-H_{DR}g_R-H_{DD}g_D.$$

The cross terms let one block's gradient alter the other block's update direction. We remove those terms:

$$d_R=-H_{RR}g_R,\qquad d_D=-H_{DD}g_D.$$

The optimizer still remembers global parameter/gradient changes. It keeps the same search policy, but searches along a different direction. Its future curvature and visited parameters can therefore change. The shared loss and global history still connect the blocks; direct cross-block action is what this test removes. This differs from training with two separate optimizers.

## Why cancellation does not settle the question

The [archive review](../../assessments/fitting-transport-review.md) found that cross terms usually opposed local descent. Removing them therefore gives a stronger negative gradient projection at many fixed snapshots. That is a derivative at one point. It does not say which direction fits better after many finite steps.

For example, the first-order change is $t\,g^Td$. A larger negative value describes steeper initial descent, but the useful step length also depends on curvature along that direction. Both directions can descend and reach different parameter settings within the same gradient budget.

## What this test can establish

Strictly more training errors after removal in every seed supports a bounded fitting benefit for full transport. Equal errors limit that claim. CE can change through confidence even when answers agree, so loss cannot replace the error criterion. Held-out scores and complete maps remain separate reliability measurements.

The measured result makes this distinction concrete. Seed 3481's equal-load training CE fell from 0.0035381 to 0.0016338, yet errors rose from 6 to 24. The other two seeds retained zero errors. The all-seed benefit rule failed; the [report](fitting-transport-results.md) retains every pair and its held-out losses.

An exact native replay verifies the control; it does not make reused pools fresh confirmation. No outcome proves that cross terms are universally necessary or explains all earlier learning.

Sources inspected: [PyTorch v2.11.0 L-BFGS implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/lbfgs.py), curvature pairs, direction recursion and line search; [Liu/Nocedal (1989), author abstract](https://users.iems.northwestern.edu/~nocedal/Abstracts/limited-memory.html), method and convex-theory scope. The partition, intervention and derivative interpretation are local reasoning; sources do not predict the experiment's scores.
