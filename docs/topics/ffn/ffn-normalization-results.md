# Next step completed: FFN effects through final normalization

Final normalization changes the measured effect of removing the last FFN. An exact decomposition verifies that its raw output projection is insufficient. This follow-up answers a math question; it is not a normalization-removal proposal.

## The decomposition

Let b=u+f. The original mean and scale are mu_b=mean(b) and sigma_b=sqrt(var(b)+epsilon), over 16 features. LayerNorm uses learned gamma and beta. Let W denote the bias-free vocabulary projection.

| Execution | Logits |
|---|---|
| Original | W[gamma*(u+f-mu_b)/sigma_b+beta] |
| Remove f, freeze original statistics | W[gamma*(u-mu_b)/sigma_b+beta] |
| Remove f, recompute statistics normally | W LayerNorm(u) |

Original minus frozen-removal logits equal **W[gamma*f/sigma_b]** exactly. Original minus normal-removal logits equal that direct term plus **frozen-removal minus normal-removal**. The latter captures changes in normalization statistics. Both terms depend on the example and chosen counterfactual; they are not universal neuron attributions.

## Executed result

Same five frozen models and opened three-pair set as the FFN branch test. No training. [Protocol](../../ffn-normalization-protocol.md).

| Condition | Mean accuracy | Mean loss, nats |
|---|---:|---:|
| Original | 99.978% | .0018 |
| Remove final FFN, normal statistics | 99.09% | .0421 |
| Remove final FFN, frozen statistics | 98.93% | .0782 |

Recomputing statistics gives lower loss in all five seeds than freezing them. Accuracy effects differ: frozen statistics give slightly higher accuracy in seeds 509/571, lower in 541, and equal in 461/487. Thus lower loss and more correct answers need not move together.

Across seeds, the direct term's logit RMS ranges .69 to 1.16; the statistics-change term ranges .36 to .70. RMS is descriptive, not a causal importance fraction. Decomposition and manual LayerNorm identities agree within 7.11e-15. Ordinary and normal-removal metrics match the preceding FFN experiment.

## Concrete answer and boundary

The FFN supplies an example-dependent residual update. Final normalization changes how that update reaches logits. Keeping LayerNorm in the calculation matters even when top-1 accuracy barely changes. Treating Wf as its exact causal contribution fails here.

This closes the local FFN-output/normalization question. It does not prove FFNs store facts, explain length failures, or justify removing either module. The subsequent [training-time gradient study](../normalization/gradient-flow-results.md) is now executed and documented with visualizations; it measures backward propagation without more activation patches in this cycle.

[Code](../../../experiments/ffn_normalization.py), [results](../../../results/ffn_normalization.json), [full logits](../../../results/ffn_normalization_scores.json). Saved logits were reloaded and losses recomputed with logsumexp minus target logit rather than cross_entropy. Provenance and checkpoint hashes passed. Verification shares the implementation; the frozen-statistics condition is an artificial diagnostic and this opened set is exploratory.
