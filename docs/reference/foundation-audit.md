# Foundational claim audit

Audit date: 2026-09-29. This is a focused section audit, not a full proof review or benchmark reproduction. It distinguishes definitions, conditional analysis and observed performance.

## Original transformer

Source: [Vaswani et al., version 7, sections 3-5](https://arxiv.org/html/1706.03762v7).

| Location | Evidence type | Reading and boundary |
|---|---|---|
| 3.1 | Architecture definition | Original blocks use post-norm; our pre-norm decoder is a deliberate variant. |
| 3.2.1, equation 1 and footnote 1 | Definition and conditional calculation | Score scaling follows an independent, standardized component model; learned vectors need not obey it. |
| 3.2.2 | Architecture definition | Heads use projections and an output projection; our shared selectors and identity output were teaching fixtures. |
| 3.2.3 | Visibility contract | Masking and shifted targets work together. Masking alone does not repair incorrect targets. |
| 3.5 | Design motivation | Sinusoidal extrapolation is motivated, not guaranteed for every trained model. |
| 4, Table 1 | Complexity analysis | Attention pairwise work and the complete model's cost are distinct quantities. |

Local action: keep the walkthrough clearly labeled as a variant; independently check its arithmetic and whole-block visibility. Neither a forward trace nor component checks reproduce translation results.

## Normalization placement

Source: [Xiong et al., ICML 2020 proceedings paper, sections 3.1-3.3 and Table 1](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf).

Table 1 places normalization before each pre-norm branch and adds final normalization. Section 3.3 studies initialization using a simplified single-head setting, square parameter matrices and zero query/key initialization, giving uniform initial attention. These conditions differ from arbitrary learned multi-head attention.

Section 3.2 evaluates warm-up settings on IWSLT14 German-English with specified optimizers and learning rates. This is empirical evidence within those settings. It does not establish that warm-up is always necessary for post-norm or always unnecessary for pre-norm.

Local decision: retain pre-norm as a transparent baseline, with no superiority claim. Correct forward/backward implementation is the current question; optimization comparisons require separately controlled training. The appendix proof and full replication setup remain unaudited.

## Implementation verification boundary

The verification uses an explicit block calculation and PyTorch's `TransformerEncoderLayer`, configured with a causal mask, pre-norm and zero dropout. Despite its library class name, the block serves as decoder-only self-attention when future visibility is blocked. There is no encoder cross-attention.

Both paths use PyTorch arithmetic/autograd, so agreement is not complete software independence. Finite differences provide a separate derivative check. The standard-library walkthrough provides an additional forward comparison. Padding tests exclude padded query positions; hiding padded keys does not automatically zero padded query outputs.

No architectural modifications or novel proposals follow from this audit. The remaining source queue includes the attention-explanation debate and more comprehensive assumptions/proof review.
