# FFN: evidence and stopping boundary

**Keep ReLU/FFN32 as the research baseline. These comparisons are closed.** Retention follows predefined decision rules. It does not establish universal superiority.

An FFN transforms each token's features. Attention exchanges information across tokens. For a normalized row vector z, a ReLU FFN computes `ReLU(zW1+b1)W2+b2`. Hidden width controls the number of intermediate features. An activation or gate changes their values and learning gradients.

| Question | Finding | What it does not prove |
|---|---|---|
| [FFN branches](../topics/ffn/ffn-branch-results.md) | Interventions and derivative checks identify contributions in saved recall models. | Removing a branch is not a general architecture recommendation. |
| [GELU versus ReLU](../topics/ffn/gelu-comparison-results.md) | Six-pair mean accuracy95.85% to96.08%; four seeds gain, one loses. The2-point/no-worse rule fails. | Smoothness does not guarantee better learning. |
| [Hidden width32 versus64](../topics/ffn/ffn-width-results.md) | Mean85.86% to91.44%, with41.51% more parameters; one seed loses7.50 points. Adoption fails. | A larger function class does not guarantee a better trained function. |
| [SwiGLU versus ReLU](../topics/ffn/swiglu-results.md) | Matched projection cost: mean validation PPL2.00% worse with SwiGLU; every pair worse. Reused test1.83% worse. Adoption fails. | This narrow gated package is not evidence that gates are globally inferior. |

The last comparison uses bias-free ReLU33 and SwiGLU22, each1056 FFN projection weights/MACs per token/block. Activations, storage and backward costs differ. It does not isolate gating from hidden width, or replace the biased FFN32 recipe.

All three training comparisons have independent checkpoint audits. They use different seeds, data and tasks; do not pool their percentages or rank activations across unmatched controls. Original [GELU](https://arxiv.org/html/1606.08415v5) and [gated-FFN](https://arxiv.org/html/2002.05202v1) papers motivate alternatives under different experimental conditions. Our results do not reproduce or refute their benchmark findings.

No extra activation, gate-width or seed search follows. The remaining causes of SwiGLU's local loss and performance at other scales are unknown. The [separate normalization transfer comparison](../topics/normalization/wikitext2-norm-results.md) is now closed with ReLU/FFN32 fixed.
