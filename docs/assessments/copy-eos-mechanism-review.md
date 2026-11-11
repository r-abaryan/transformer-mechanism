# Can we identify what makes EOS win?

Review, 2026-10-07. **A targeted direct-route test is feasible, but the proposed “marker reading versus downstream processing” distinction is not exclusive.** No new model evaluation or training was performed for this review.

## What we verified

Inspected the positional-site forward, source-visibility masking, marker-alignment instrumentation and saved failure report. All four selected checkpoint files and four score files match their recorded hashes: cross-only 3767/3779/3793 and combined 3767. The weak seed has 128 premature-EOS first errors on test10; the other three models have none there. Earlier [EOS blocking](../topics/benchmarks/copy-site-failure-results.md) restores 127/128 full copies.

The saved site scores contain only the first decoder row (`scores[:,:,0]`). Failures occur at output positions 4 or 5. Their attention weights and intermediate states must therefore be recomputed from the checkpoints. Existing forward equations support this; the current saved attention scores alone cannot answer the question.

## Why the distinction needs narrowing

| Tempting inference | Why it would be too strong |
|---|---|
| High attention on source EOS causes stopping | The corresponding value vector, output projection and downstream computation determine the effect |
| Masking the marker isolates its content | Softmax renormalizes every other weight, changing the entire mixture |
| Removing the marker position removes all marker information | Encoder self-attention can distribute that information into other positions |
| A readout applied halfway through the decoder reveals its decision | Intermediate activations have not passed the remaining normalization/FFN; such logits are not the model's actual prediction |

## Recommended bounded question

**At the weak model's first failing decision, does the direct value contribution from the source-end position promote EOS over the correct content token?**

For fixed attention weights A, source values v and output projection W, the cross-attention output is

`c = W concat_h(sum_j A[h,t,j] v[h,j]) + b`.

Let `c_end = W concat_h(A[h,t,end] v[h,end])`. Test `c' = c − c_end`, retaining the output bias, all attention weights, other values and decoder residual. Recompute the actual remaining normalization, FFN and readout. This decomposition is local algebra from the inspected implementation, not a measured attribution.

Use the actual EOS-versus-correct margin `m = z_EOS − z_correct`. Define `effect = m_native − m_removed`; positive means the direct end-position route promotes EOS under this intervention. Also count exact correct-token rescues and harms, since reducing EOS can expose another wrong token.

| Outcome | Permitted conclusion |
|---|---|
| Margin falls and correct content replaces EOS | Removing this direct route is sufficient for local recovery on those examples |
| Margin falls but another wrong token wins | The route contributes to EOS preference, but removal is insufficient |
| Margin does not fall | This intervention does not support the proposed direct-route explanation |

None proves that downstream processing is irrelevant or that marker information is absent elsewhere. Zeroing a contribution also changes mixture magnitude; it is a route ablation, not a semantics-preserving replacement. A matched-magnitude contribution from one predeclared non-marker position is a useful specificity control, with zero-norm cases explicitly reported. Without such a control, report only direct-route dependence, not marker-specific dependence.

## Scope before execution

Use all 128 fixed first-failure positions in weak-seed test10, and the same examples/positions in the three successful models. Include genuine terminal EOS and short successful content controls to detect indiscriminate suppression of stopping. Do not choose heads or examples by their intervention scores. Start with both heads' summed end-position contribution; no head sweep.

Before any scoring, freeze the operator, specificity-control construction, exact row counts, metrics and recovery threshold. Verify native reproduction, exact contribution reconstruction, zero-removal identity, supplied synthetic rescue/harm cases, saved initialized evaluation and independent reference calculations. Then execute one diagnostic and stop. This review proposes that design; it does not freeze or launch it. Why training selected the weights remains a separate question.

## Sources

[Newman et al. (2020), sections 1 and 3](https://arxiv.org/html/2010.07174v1#S3), inspected, motivates separating content production from termination. Its hidden-state explanations are not established here. The weighted-value decomposition and caveats above follow from the [local forward implementation](../../experiments/copy_position_sites.py), inspected directly. No novelty or source-backed prediction of local intervention results is claimed.
