# Explaining learned failures

Latest result: [the four-condition copy comparison](../topics/benchmarks/copy-position-sites-results.md) confirms combined exposure in every fresh seed. Cross-only succeeds in two seeds but fails at length 10 in the third; self-only remains weak. All short-task fit gates pass. The [copy assessment](copy-generalization-assessment.md) now consolidates the bounded learning account. No experiment is running or queued; exact mediation and broader transfer remain open.

Active direction, 2026-10-07. Prioritize understanding learned mechanisms and identifying a genuine research gap before selecting another experiment. The user parked RL. Completed experiments remain evidence; their stopping rules do not settle their underlying questions. This page records priorities and the initial review; the separate [endpoint-gradient diagnostic](../topics/attention/objective-beta-pressure-results.md) is executed and audited.

Preceding diagnostic: [copy source visibility](../topics/benchmarks/copy-source-visibility-results.md) recovered all 351 random-long first-token errors by decoder-only exclusion with the full encoder unchanged. That endpoint intervention motivated the now-completed learning study; it was not itself a learned repair.

The [retention learning strategy](retention-learning-strategy.md) closes direction variants, fitting extensions, the trajectory diagnostic and this fixed map-replacement policy. Incomplete active fitting blocks a stronger exposure conclusion; withdrawing old feedback is coupled to adding new maps. Future comparisons need bounded training-only adequacy before locked confirmation, not tuning on these opened scores. The [causal synthesis](retention-learning-causal-account.md) preserves earlier effects and limits. No retention experiment is running or queued. RL stays parked.

## Remaining questions and stopping boundary

### Vision and next milestone

| Goal | Where we are | What would count as progress |
|---|---|---|
| Understand computation | Equations, implementation checks and component interventions provide substantial groundwork | Resolve a specific uncertainty, rather than repeat a known identity |
| Understand the black box | Some local prediction effects are explained; why training selects unreliable solutions remains partial | A discriminating explanation of a learned failure, with its scope and counterexamples |
| Find gaps and possible novel methods | No field-level novelty is established | Identify what predecessors leave unanswered, then test a mechanism-based improvement against them |

Here, "black box" means incomplete knowledge of learned representations and learning dynamics. The forward equations are known. An intervention that changes predictions does not by itself explain why training learned those weights.

**The gap review, first-token diagnostic, positional-exposure repair and site comparison are complete.** The joint training intervention is reliable in these confirmations; cross-only is sometimes sufficient. This partially localizes the effective exposure intervention without proving a unique circuit or universal site necessity. No position-range or head sweep is queued. The method is published; novelty is not established. Retention variants remain deferred.

For retention, the deferred question is: **does unreliable unseen-map retrieval follow from restrictions of the reader, or from the solution training selects?** Separate representability, reachability through normal writes, learning and transfer. Existing finite-pool fits establish useful witnesses, not a solution for every possible map. The tested 615-parameter standalone delta reader is not a full Transformer.

The suggested combined old/new-map fitting comparison is withdrawn as the selected next study. It could be a supporting control for a justified exposure hypothesis; it is not itself the research direction. Do not resume optimizer, beta, donor or exposure grids without a new discriminating question.

**Literature boundary.** [Schlag, Irie and Schmidhuber (2021)](https://proceedings.mlr.press/v139/schlag21a.html) already discuss finite-memory capacity and delta-rule correction in linear attention. The proceedings abstract was checked for this planning decision; it does not establish the cause of our classification failures. [DeltaNet (2024), section 2.2](https://arxiv.org/html/2406.06484v2) supplies a related update rule, with input-dependent write strength unlike our scalar beta. These predecessors rule out treating the update rule or generic memory interference as a new contribution. A specific learning explanation or repair still needs a targeted literature comparison; no exhaustive novelty search is claimed.

### Review outcome: choose copying

2026-10-07. This is a focused assessment of two branches, not a repository-wide code audit.

| Selection criterion | Compact retention | Marked copying |
|---|---|---|
| Strongest evidence | Finite pools can be fitted; saved-read conflicts and endpoint interactions are established | Short copying fits; supplied alignment recovers every tested longer group |
| Main uncertainty | Universal representability, normal-write reachability and learning remain entangled | Why the learned source read changes when the source grows |
| Next useful contrast | Needs a task-wide capacity argument or a new learning hypothesis; another parameter variant is weak | Added attention competitors versus changed encoder representations, including their interaction |
| Decision | Defer further empirical variants | Prioritize one causal diagnostic, then decide whether a learning intervention is justified |

The old [shared-head solver failure](../topics/attention/shared-readout-feasibility-results.md) is **resolved by the later exact certificates** in [numerical recovery](../topics/attention/shared-readout-recovery-results.md). Reopening the original solver question would repeat completed work. Conversely, those certificates constrain fixed reads, not every parameter setting. Follow-up count does not strengthen independent replication when starts and data are reused.

**Exact selected question:** why can the marked RoPE encoder-decoder copy short sources but mispredict the first token of longer ones? At lengths 8/10, existing evidence records 177/384 and 174/384 first-token errors. Full-copy means are 10.42%/2.34%; supplied alignment gives 100% in every tested group. These are measurements from [the executed alignment study](../topics/benchmarks/copy-marker-alignment-results.md), not new evaluation scores.

**A useful implementation constraint:** this model has one decoder block. Before cross-attention, its first query depends on BOS, causal decoder self-attention and fixed parameters, not the source. The source enters through encoder-derived K/V. There is no earlier generated mistake at this position. This does not make learned Q weights irrelevant; it rules out source-dependent changes to that first query within a fixed checkpoint. Deeper decoders need not satisfy this restriction.

For a head, with first query fixed,

\[
p_0=\frac{e^{s_0}}{\sum_j e^{s_j}},\qquad s_j=q_0^\top k_j/\sqrt{d}.
\]

Adding keys while holding existing keys fixed changes normalization but preserves existing pairwise odds: \(p_i/p_j=e^{s_i-s_j}\). Re-encoding a longer source can instead change existing keys, values and their relative scores. This local algebra motivates two explanations; it does not identify the observed cause.

| Candidate explanation | Discriminating evidence to seek | What it would not establish |
|---|---|---|
| Additional competing values disrupt a previously useful read | Reveal added positions to cross-attention while encoder outputs stay fixed; measure answer margin and errors | Why training chose that sensitivity |
| Encoder context changes the useful source representation | Change encoder visibility while keeping decoder-memory visibility fixed | That all effects belong to K rather than V |
| Both interact | A crossed intervention differs from either intervention alone | A unique additive attribution or a general Transformer limit |

The design above was subsequently [frozen](../protocols/copy-source-visibility.md) and [executed](../topics/benchmarks/copy-source-visibility-results.md): one 2-by-2 visibility contrast, unchanged EOS positions, ordinary short and aligned controls, three saved seeds and all declared examples. Decoder-only and joint recovery pass in all nine random-long seed/groups; encoder-only recovery fails. Both masked and ordinary short baselines have 100% first-token accuracy, with different margins. No examples were selected by patch success.

**Decision boundary:** if ordinary competition explains the local failure, report a local instance of an existing mechanism and stop; no novelty claim or temperature sweep. If encoder context or an interaction is necessary in this contrast, it can motivate one training hypothesis with a predicted representation change. If neither contrast discriminates, retain the uncertainty and stop this design. Even a successful patch only localizes the endpoint failure; a learning explanation needs a controlled training intervention and fresh confirmation. Full-sequence copying remains a later validation target, not something first-token recovery proves.

### Primary-source comparison and review scope

| Source inspected | Relevant prior knowledge | Consequence for this project |
|---|---|---|
| [Chiang and Cholak, ACL 2022, sections 5.1-5.3](https://aclanthology.org/2022.acl-long.527.pdf) | A learned first-symbol reader can fail on longer strings through competing values; log-length attention scaling is already proposed | Dilution and simple sharpening are not novel. Their FIRST setup does not establish the cause in our RoPE encoder-decoder |
| [Zhou et al., 2023, abstract](https://arxiv.org/abs/2310.16028) | Proposes a conjecture relating simple Transformer programs and length generalization | Representing an algorithm and learning it are separate questions; this is a conjecture, not a guarantee for our model |
| [Phan et al., 2025, abstract](https://arxiv.org/abs/2510.00258) | Reports shortcut learning and delayed-attention training in Transformer-RNN hybrids | A generic training schedule change is not automatically new; different task and architecture, results not reproduced here |

This targeted search supports question selection, not an exhaustive novelty claim. Older directly relevant analysis takes priority over a newer abstract. The local question remains unanswered by our evidence; that alone does not make it a field-level research gap.

Inspected the copy producer, source-marker adapter, RoPE forward path, parent architecture and evaluator; retention reader and linked causal/feasibility reports; current failure register; five report/audit bindings. A separate [review check](../../results/gap_review_20261007/verification.json) recomputes the cited copy summary and checks first-query invariance through the real model path using 18 forward calls across three saved seeds. This checks an implementation constraint, not the proposed causal hypotheses. Zero training updates. Historical audits were not rerun, and the entire repository was not certified.

This is a prioritized list of unknowns, not authorization to execute every item. Mathematical verification and bounded component coverage already provide enough background; we do not need to repeat them.

| Priority | Remaining question | Evidence needed | Boundary |
|---|---|---|---|
| Completed: copy endpoint | Which source-side change causes first-token failure when the source grows? | Decoder-only exclusion recovers every tested error without changing the full encoder | Stop subdivisions; the mask is task-informed and does not explain learning |
| Deferred exposure hypothesis | Does finite-pool reuse contribute to gains and harms on unseen maps? | Adequate active fitting and a contrast separating additional exposure from withdrawal of old feedback, followed by fresh confirmation | [Fixed replacement](../topics/attention/map-exposure-results.md) fails its criteria; this is not the selected next study |
| Exception to retain | What accounts for answers already wrong after their own write, including six last-position errors per seed? | Preserve these cases in future successful/failing trajectory comparisons with correct controls | Later writes cannot explain last-position errors. These cases do not automatically justify another endpoint partition |
| Conditional on a useful intervention | Can a learned change improve reliable shared maps and longer-input/held-out behavior? | Adequate full-group fitting followed by fresh paired seeds/data; report CE, accuracy, maps and compute | None of the present hybrids is a net endpoint repair. Reused training cohorts cannot establish generalization |
| Conditional: copy learning | Why does training miss transferable source alignment even when supplied routing works? | A bounded training intervention that predicts a measured representation change, with full fit and fresh confirmation | Oracle routing is task knowledge; do not jump from a patch to a learning cause |
| Deferred: state tracking | Why do trained state representations remain unreliable? | Groupwise fit and a specific state/credit-assignment intervention | Existing GRU and Transformer results are task-specific; no architecture impossibility claim |
| Parked: RL | PPO, GRPO variants, RLHF and related objectives | Resume the existing [RL roadmap](../topics/rl/rl-roadmap.md) when requested | Not part of the active retention diagnosis |

For the active branch, stop checkpoint-only exploration once it supports a falsifiable learning test. Freeze a finite recipe-development budget separately from confirmation, record every attempt and stop at its limit. If evidence remains inconclusive, retain the unknown rather than calling the topic solved. Larger-scale adoption, novelty, full historical optimization reconstruction and nonlinear-readout benefits remain outside demonstrated scope; they are not mandatory prerequisites to close a bounded diagnostic.

## Failure versus explanation

| Priority | Observed behavior | Established | Still unresolved |
|---|---|---|---|
| Deferred: [compact retention](compact-retention-assessment.md) | Fresh L-BFGS fits two seeds perfectly and leaves six errors in the third; held-out load-32 accuracy is about 87–89% with CE above Adam's | Finite-pool normal-write feasibility, confirmed 99% fitting under both procedures; earlier interference/pressure/transport findings remain valid | Individual procedure causes, universal/dataset-wide capacity and why learned solutions have weak unseen-map reliability |
| First: [marked copying](copy-generalization-assessment.md) | Short training is fitted; native longer copying fails; supplied source alignment recovers it | The saved downstream decoder can produce the answers under that supplied route | Why training fails to learn transferable routing; which role belongs to route position versus mixture sharpness |
| Later: [state tracking](../topics/benchmarks/state-momentum-training-results.md) | Training groups remain unreliable despite development and continuation | A one-time momentum intervention has small mixed effects; extra training helps the weak seed | How representations and gradients produce the remaining group failures; adequate full-group fitting is still absent |

These tasks require different explanations. Do not combine their accuracy gaps into one architecture diagnosis. Use the [failure register](failure-register.md) to avoid repeating completed interventions.

## Initial diagnostic, completed

**At the same saved weights, do assigned-query and all-query training objectives exert different pressure on beta?**

The earlier swap asks what changing beta does to predictions. This question asks what the loss rewards locally. It is a first step toward explaining training, not a full explanation of its history.

For H histories containing m keys each, compare $L_{assigned}=H^{-1}\sum_h\ell(h,q_h)$ with $L_{all}=H^{-1}\sum_h m^{-1}\sum_q\ell(h,q)$. The second is the expected loss under uniform query resampling at those weights. A particular sampled minibatch can have a different gradient. Equal weighting across the four loads matches the original sampling objective.

Let alpha be the learned write logit and beta = sigmoid(alpha). For a loss L,

$$\frac{\partial L}{\partial\alpha}=\frac{\partial L}{\partial\beta}\,\beta(1-\beta).$$

For plain gradient descent, a positive derivative pushes beta down; a negative derivative pushes it up. Near an endpoint, the sigmoid factor can reduce alpha's gradient. Adam also uses stored moments, so a loss derivative alone does not establish the next optimizer update or the actual historical trajectory. These are local chain-rule interpretations, not measured gradient findings.

The [frozen diagnostic](../protocols/objective-beta-pressure.md) used four existing checkpoints and original training histories at each load, comparing assigned-query versus exhaustive all-query mean CE at the same weights. Beta derivatives and parameter-block gradients passed independent checks and finite differences. The four loads receive their original equal weights. No new held-out selection, beta grid, optimizer step or training continuation occurred. See the [guide](../topics/attention/objective-beta-pressure-guide.md) for the mathematics and interpretation.

A disagreement would identify local objective tension. Agreement or tiny gradients would leave that explanation unsupported at the tested endpoints. Neither result alone establishes why training followed its particular path. Report both seeds and all loads, including contradicting cells.

## Original artifact inventory

A live read checked all four checkpoint hashes, their model/optimizer payloads and step counters against the completed report and its passing audit. All checkpoints are final step 3000. Each run records only three minibatch losses at steps 1000/2000/3000. The training loop logs every 1000 updates; because loads cycle over four values, those logged losses are all load-32 batches. They are not full-pool convergence measurements.

The inspected study folder has final weights, final Adam states, exposure counts and prediction scores. It has no intermediate training-weight checkpoints or logged beta/gradient trajectory. Final optimizer states cannot reconstruct the complete sequence of earlier updates. See [artifact inventory](../../results/learning_mechanism_review/inventory.json).

The separately [frozen replay](../protocols/retention-learning-replay.md) has now reproduced the original initialization, streams and budget with tensor-exact endpoint checks. It adds intermediate checkpoints and observations; it does not add updates to the saved model or fresh confirmation. The original study's final-only artifacts remain unchanged. Any later causal training intervention needs its own matched control and budget.

## Source and decision boundary

[Yang et al., DeltaNet 2024, section 2.2](https://arxiv.org/html/2406.06484v2) was reopened for the correction/read rule and input-dependent write strength. Our reader uses one scalar beta, tied normalized query/key vectors and a separate classifier. The paper does not explain our learned scalar, predict our scores or make this a full DeltaNet comparison.

Initial review scope: retention producer, reports, final checkpoint contents and audit binding; linked copy/state reports for prioritization. That initial review executed no new gradients or predictions. Later diagnostics, reproductions and training interventions are separately reported and audited. No reliable held-out shared-map repair or exhaustive repository audit has been established. Keep the other questions deferred and advance only when a result distinguishes explanations. RL remains parked until the user resumes it.
