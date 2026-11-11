# Attention competition: score gaps, weights and values

**A target can remain highest-scoring while losing attention mass. That alone does not establish output error.** All 37 controlled fixtures pass. An independent 50-digit decimal audit checks 86 scalar results; maximum disagreement is 1.12e-16 (rounded upward). These are arithmetic checks, not training trials.

## Implementation versus architecture

| Question | Evidence category | Current boundary |
|---|---|---|
| Dense versus efficient/cuDNN execution | Implementation | Same attention equation; measured output/gradient agreement, time and tracked memory. |
| Batch size, precision, run variability | Experimental conditions | Untested settings limit benchmark generalization. |
| Head count | Architectural choice | Earlier head removals test reliance; they do not establish optimal head count after retraining. |
| Whole-model convergence and speed | Joint outcome | Depend on architecture, training and implementation; not established by kernel benchmarks. |
| Competition inside softmax | Attention-function property | Derived below under explicit score assumptions; not a new architecture or a universal capacity limit. |

The [GPU backward section](native-backward-results.md) remains closed. This probe addresses different mathematics.

## Math: why the score gap matters

Let t be the target, with m other visible tokens. Scores s already include dot-product scaling and any temperature. For equal distractor scores, define the gap Delta = s_t - s_d:

`p_t = exp(s_t)/(exp(s_t) + m*exp(s_d)) = 1/(1 + m*exp(-Delta))`.

To keep target weight p_t = rho, where 0 < rho < 1:

`Delta = log(m) + log(rho/(1-rho))`.

Thus doubling m requires an additional log(2) score gap in this equal-score setting. A positive gap guarantees first rank, not most of the total mass. My earlier phrase "routing is correct" was too strong: this probe holds target rank correct, not retrieval fidelity.

| Distractors m | Target weight at gap 4 | Gap needed for weight .9 |
|---|---:|---:|
| 1 | .982014 | 2.197225 |
| 16 | .773365 | 4.969813 |
| 64 | .460363 | 6.356108 |
| 1,024 | .050620 | 9.128696 |

We tested six counts and four gaps: 24 cases. Six more cases maintained .9 weight. These are deterministic score fixtures, not learned queries/keys.

With unequal gaps Delta_j, the exact expression is `p_t = 1/(1 + sum_j exp(-Delta_j))`. Replacing the gaps by their mean is generally wrong. Four gaps [2,2,2,2] give .648786 weight; [.5,.5,3.5,3.5] give .439859 despite the same mean. Three unequal-score fixtures checked the exact formula and bounds obtained from minimum/maximum gap. Future masked tokens do not count as competitors; changing real context can also change all the scores.

## Why values matter

Let v_bar be the attention-weighted mean of distractor values, normalized within the distractors. Then:

`o - v_t = (1-p_t)*(v_bar-v_t)`.

For any norm, `||o-v_t|| <= (1-p_t)*max_j ||v_j-v_t||`. Low target weight can therefore cause large error, small error, or no error.

Hold m=16, gap=2 and target value=1 fixed. Target weight is .315919 in every row:

| Distractor values | Output | Absolute error from target value |
|---|---:|---:|
| All 1 | 1.000000 | 0 |
| All 0 | .315919 | .684081 |
| All -1 | -.368161 | 1.368161 |
| Eight 0, eight 2 | 1.000000 | 0 |

The last case cancels despite distinct values. These scalar examples verify the identity and bound; they are not classifier accuracy. Residuals, other heads and the readout can further change final predictions.

## What our trained-model evidence says

Earlier [distractor removal](../positions/position-distractor-results.md) improved six-pair accuracy from 64.24% to 89.69% with retained positions unchanged. However, removal changed contextual representations as well as attention normalization. The [fresh subset study](../retrieval/subset-confirmation-results.md) then failed its general rescue criteria. Neither isolates softmax dilution as the cause. We reviewed those findings; no trained model was rerun here.

## Updated research comparison

Focused primary-source check on September 30, 2026; not an exhaustive survey or a paper replication.

| Source | Relevant finding and reading depth | Relation to our evidence |
|---|---|---|
| [Velickovic et al., ICML 2025](https://proceedings.mlr.press/v267/velickovic25a.html), [v3 full text, section 2](https://arxiv.org/html/2410.01104v3) | Lemma 2.1 inspected: with uniformly bounded logits and fixed positive temperature, individual weights scale as Theta(1/n). Also proposes adaptive temperature. | Our fixed-gap example illustrates dispersion. It does not reproduce their transformer theorem or imply exactly uniform weights. Growing the gap changes the bounded-score assumption. |
| [Golowich et al., ICML 2025](https://proceedings.mlr.press/v267/golowich25a.html) | Abstract checked: length generalization under sparse dependencies in an idealized transformer framework. | Counterweight to universal failure claims. Task assumptions and model definitions matter; no local reproduction. |
| [Gao et al., ICML 2026](https://proceedings.mlr.press/v306/gao26y.html) | Abstract checked: Softplus with L1 normalization, dynamic scaling and re-weighting; authors report improved length extrapolation. | Recent alternative already exists. We have not audited its proofs/code or reproduced its gains; it is not a recommendation or our novel proposal. |

## Decision and evidence

Close this mathematical probe: fixed score gaps permit dilution, mean gap is insufficient, and values determine whether dilution corrupts an isolated head output. Whether this explains our trained models' failures remains unproven. No architecture change follows from these examples alone.

[Frozen protocol](../../attention-competition-protocol.md), [runner](../../../experiments/attention_competition.py), [raw results](../../../results/attention_competition.json), [independent decimal audit](../../../experiments/audit_attention_competition.py), [audit result](../../../results/attention_competition_audit.json). Standard-library CPU arithmetic only. Reproduce using the project Python and each script; the runner protects existing results from overwriting.

Follow-up completed: [trained weight interventions with fixed V](trained-competition-results.md). Uniform sharpening fails; oracle target-directed weights largely rescue. This narrows the explanation without proving distractor-count dilution as the original cause.
