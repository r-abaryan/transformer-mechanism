# Normalization: evidence and stopping boundary

**The new language-model transfer comparison is also closed.** [Gain-only LN/RMS on WikiText-2](../topics/normalization/wikitext2-norm-results.md) improves RMS validation perplexity395.48 to389.17 (1.60%); every paired seed gains. Only the predeclared2% mean-improvement requirement fails. Reused test improves1.34%. Six finite runs and all2,779,278 independent token-loss checks pass, with zero argmax differences. Keep the original recipe without tuning. This is a modest local RMS gain, not a LayerNorm quality win or general adoption.

Both controls have1,068,752 parameters and gain-only norms at all five sites. They retain the original biased ReLU/FFN32 but remove80 normalization biases in both arms. Historical biased-baseline scores are not paired controls, and bias removal is not adopted. The [guide](../topics/normalization/wikitext2-norm-guide.md) explains the mathematics. This new task question does not reopen the earlier recall thresholds below. Which site explains the gain, larger-scale transfer, convergence and speed remain unmeasured. No further sweep is selected.

**Normalization is closed for this research cycle.** RMSNorm improves mean recall under the controlled recipe but fails its predefined reliability/adoption rule. Retain centered normalization as the baseline; neither method is universally superior.

| Question | Established evidence | Boundary |
|---|---|---|
| [Residual/normalization gradients](../topics/normalization/gradient-flow-results.md) | Local derivative checks and deterministic training replays agree. A skip contribution can be cancelled by its branch. | An identity path alone does not guarantee nonzero total gradients or stable learning. |
| [Pre/post placement and training budget](../topics/normalization/norm-extension-results.md) | All20 runs exceed99% at6,000 updates; two weak3,000-update post-warmup runs recover on fresh mappings. | Placement superiority depends on the recipe/budget. Their internal learning-delay causes remain unknown. |
| [LayerNorm/RMSNorm mathematics](../topics/normalization/rmsnorm-math-results.md) |68 cases and independent Decimal audit pass. Centered-point outputs can agree while gradients differ; epsilon modifies scaling symmetry. | These local properties do not select a training winner. |
| [Matched LN/RMS training](../topics/normalization/rmsnorm-training-results.md) | Ten runs and30 independent evaluations pass. Six-pair unique mean91.71% to94.04%; short-length RMS floors pass. | RMS seed1543 scores89.53%; seeds1511/1567 lose more than1pp unique accuracy. Adoption fails, with gains retained. |

The training comparison matches parameter states, gain-only bias conventions, epsilon, exposure and budget. It changes all five normalization sites together. Its measured differences support that complete replacement within the recipe, but cannot identify the responsible site or establish that centering is unnecessary in general. The LN baseline also has weak seeds; retention under a stopping rule does not certify reliability.

Keep all failed thresholds and numerical-verification failures. No automatic epsilon/bias search, additional seeds or training extension follows. Use the [recipe assessment](recipe-assessment.md) to move toward one different-task check. Other scales, optimizers, site-specific mechanisms and hardware performance remain unmeasured.
