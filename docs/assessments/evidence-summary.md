# Research evidence summary

Latest external FFN comparison: [SwiGLU versus ReLU](../topics/ffn/swiglu-results.md), bias-free hidden22/33 at equal projection weights/MACs, with MQA fixed. Six runs and 2,779,278 independent validation/test loss checks pass; no argmax differences. SwiGLU mean validation PPL392.25 to400.10 (+2.00%), every pair worse; adoption fails. Reused test mean365.35 to372.03 (+1.83%) is descriptive. Thirty scalar/matrix math cases and six full-model preflights pass. Gate arithmetic is valid; quality improvement, global inferiority, convergence and speed gains are not established. Study closed without tuning.

Previous external attention comparison: [WikiText-2](../topics/benchmarks/wikitext2-results.md), three paired MHA/MQA seeds and 6,000 updates each. All six runs and the 1,473,408-token independent loss audit pass. Geometric mean perplexity worsens 357.92 to 364.95 (1.96%), with every MQA seed worse, yet stays within the declared tradeoff tolerance. Potential compact K/V payload halves; total parameters fall 0.051%. One argmax differs across precision paths; loss tolerance passes. No convergence or speed claim follows. Study closed without tuning.

Latest different-task finding: [majority aggregation](../topics/benchmarks/majority-transfer-results.md). All five seeds score100% at9/11/15 bits. At21 bits, mean97.18% hides seed1627 at86.25%, failing overall and boundary reliability floors. All113 errors occur on one-bit majorities with label1. Twenty independent evaluations pass; exact arithmetic witnesses pass. This supports bounded recipe transfer beyond recall, not a learned counting mechanism or general extrapolation. Study closed without tuning.

Latest trained finding: [gain-only LN versus RMSNorm](../topics/normalization/rmsnorm-training-results.md). Five paired seeds yield91.71% to94.04% mean six-pair unique accuracy; RMS short-length floors pass. Seed1543 scores89.53%, while1511/1567 lose5.98/1.17pp unique accuracy. Mean gain is sensitive to weak LN seed1597. All30 independent evaluations pass; adoption fails. [Normalization closure](normalization-assessment.md) and [recipe consolidation](recipe-assessment.md) set the stopping boundary and next task question.

Latest: [LayerNorm versus RMSNorm](../topics/normalization/rmsnorm-math-results.md),68 checked cases and30,464 audited scalar comparisons. Common-shift behavior and local gradient directions differ; positive epsilon qualifies idealized scale invariance. Constant/zero inputs can have zero output and nonzero derivatives. A failed finite-difference step is archived and corrected without changing the threshold. No training or runtime superiority is established.

Latest: [cache precision](../topics/attention/cache-precision-results.md),180 independently audited cells. Of14,400 reused final decisions per mode, FP32 changes0, FP16 changes1, INT8 K/V changes9 (5 damaged,4 repaired). INT8's four-byte scales reduce its savings:25% versus FP16 here. Exact decision preservation fails for FP16/INT8 on this subset; no universal deployment rejection follows. Study closed.

Latest execution comparison: [MQA GPU cached decoding](../topics/attention/mqa-decoding-results.md). All36 numerical checks and the independent audit pass across three processes. Compact cache halves, but zero of six shapes passes the repeatable 10% speedup rule. Batch16 MQA has higher temporary tracked allocation. Timing variability and its causes remain unresolved; no general performance ranking follows. Study closed.

Latest trained comparison: [MHA versus MQA](../mqa-training-results.md). Five paired seeds pass the fixed resource/quality rule: compact K/V payload halves and parameters fall 10.69%. Six-pair mean unique accuracy falls 94.73% to 94.05%; worst seed loss is 4.111pp. Every MQA query-type score stays above90%. All30 independent evaluations pass. Accept MQA for this recipe/tolerance, with measured losses retained. Comparison closed.

Latest mathematical probe: [shared KV heads](../shared-kv-results.md). Four-head payload is 80/40/20 KiB for MHA/GQA/MQA at fixed dimensions. Already-shared controls match; mean pooling different heads changes outputs. Shared keys constrain joint log odds within each group. All four probes and the 1,969-scalar independent audit pass. This establishes scoped arithmetic properties, not trained quality or latency. Study closed.

Previous implementation check: [KV-cache equivalence](../topics/attention/kv-cache-results.md). All 120 cases and the independent audit pass, with 4.75e-14 maximum cached/full error and no argmax changes. Both intentional error controls are detected. This is reused-fixture inference agreement, not new learning or speed evidence. Study closed.

Previous comparison: [broader training queries](../topics/training/query-exposure-results.md). With architecture and context streams fixed, unique-key mean accuracy rises 74.32% to 99.89% at four pairs and 59.53% to 94.83% at six. Every paired seed improves. One six-pair seed scores 87.77%, failing the fixed all-seed reliability rule. All 30 independent evaluations pass. This supports a training-coverage benefit, not an architecture change. Comparison closed.

Previous diagnostic: [fixed-context query switching](../topics/training/query-switch-results.md). Five saved models achieve81.59% mean on previously untrained unique-key queries; only two seeds reach90%. The models are not completely query-blind on these inputs, but reliable transfer fails. All predictions checked against independent attention; no retraining. Diagnostic closed.

Previous comparison: [FFN hidden width32 versus64](../topics/ffn/ffn-width-results.md), with RoPE/ReLU fixed. Mean six-pair accuracy rises 85.86% to 91.44% (+5.59pp), with41.51% more parameters. Four seeds improve; one loses7.50pp and scores85.90%, failing the all-seed reliability/no-worsening rules. Median gain is0.68pp. Retain width32 under the stopping rule, not as a claim of superior accuracy. All30 independent evaluations pass. Four/six-pair data is fresh; three-pair data is reused diagnostic evidence. Comparison closed.

Previous comparison: [ReLU versus GELU with RoPE fixed](../topics/ffn/gelu-comparison-results.md). Mean six-pair accuracy 95.85% to 96.08% (+0.225pp); four seeds improve and one worsens. Both arms clear90% in all five seeds, but GELU fails the fixed >=2pp/no-worse-seed rule. Retain ReLU for this recipe. All30 independent checkpoint evaluations pass; comparison closed. The pretraining data-capacity correction is archived, and the three-pair test has only40 families.

Previous comparison: [repeated-key latest-value recall](../topics/training/repeated-key-results.md). After retraining, RoPE improves six-pair mean 57.18% to 93.51% and every paired seed gains. Two seeds miss the fixed 90% threshold (89.875% / 89.9028%), so the reliability rule fails. All 30 independent checkpoint evaluations pass audit. Comparison closed; no tuning or automatic follow-up. The earlier unique-key success remains valid.

Previous comparison: [sinusoidal versus RoPE](../positional-comparison-results.md). Five paired seeds improve mean six-pair accuracy from 81.91% to 99.86%, with 100% three/four-pair test accuracy in both arms. All frozen keep criteria and the independent audit pass. Keep RoPE for this task/budget; comparison closed. General superiority and the mechanism of the gain remain unproven.

Updated 2026-10-01. This is the current overview; linked documents hold detailed explanations and raw evidence. **Derived** means justified under stated assumptions; **observed** means measured locally; **reported** means a paper's finding, not our reproduction.

Previous mathematical measurement: [causal head-width constraints](../topics/attention/causal-width-results.md). A jointly visible log-odds rectangle proves one causal pattern requires width two; width two constructs it. Width one still matches outputs for scalar values, but not identifying values. Independent decimal audit passes 108 scalar checks. This necessary constraint is not a complete characterization of causal attention or a trained-model improvement. Bounded probe closed.

Earlier CPU measurement: [attention runtime and memory](../topics/attention/attention-cost-results.md), sixteen isolated CPU processes. Dense/tiled outputs agree within 1e-15. At2,048 tokens median forwards are92.60/33.86 ms; process peaks are358.88-360.59/304.70-309.22 MiB. Causal future-tile skipping contributes to execution differences. Process peaks are not tensor allocation; setup/hash peaks can conceal additional allocations. Exact tiling changes workspace, not quadratic arithmetic. Fixed benchmark closed.

Research decision: [length exposure completed and closed](../topics/training/length-exposure-results.md). Mixed three/four-pair training improves six-pair mean accuracy from 58.11% to 81.65%; all seeds gain and three-pair accuracy remains 100%. Four seeds miss 90%, so the fixed keep rule fails. All thirty checkpoint evaluations pass audit. This is a training-recipe result with additional compute, not an architecture improvement. No further experiment is selected; [closed-branch evidence map](research-checkpoint.md).

Earlier mechanism observation: [matched key-binding transfer](../topics/retrieval/key-content-results.md), 35 conditions. Donor key swaps with fixed values/positions redirect recipient answers in 99.997% of cases via two block-1 value-row outputs. Reverse transfer restores 99.828%; answer-preserving control retains 99.992%. This supports key-dependent binding information, not pure key identity. Derived mappings include training overlap; the experiment is exploratory. Saved metrics and controls pass audit.

Previous row diagnostic: [head-output row localization](../topics/attention/head-row-results.md), 120 conditions. Both-head removal at block-1 value rows leaves 24.60% mean accuracy versus 99.75% at its final-query row. Value-only and all-row errors differ; removal effects are not monotonic. Block-2 non-final removals leave final predictions unchanged within tolerance, as its architecture requires. All saved summaries pass audit. This is dependency localization on reused examples, not proof of preceding-key content transmission.

Previous head diagnostic: [head-output removals](../topics/attention/head-ablation-results.md), 35 conditions in five frozen pre-norm models. Both-head removal leaves 27.81%/18.35% mean accuracy in blocks 1/2. Important head indices differ across seeds. Seed 683 supplies a counterexample to ranking removal impact by target attention alone. Explicit and library interventions agree within 7.91e-14; saved metrics reproduce exactly. These opened-data diagnostics do not establish safe pruning or fewer-head training limits.

Normalization-budget conclusion: [normalization-budget extension](../topics/normalization/norm-extension-results.md), all twenty runs pass 99% after 6,000 updates. On the same fresh mappings, post-warmup seeds 659/683 improve from 97.88%/50.92% to 99.89%/100%. All twenty replays match their original 3,000-update weights; forty checkpoint evaluations reproduce exactly. This establishes budget-dependent recovery in the tested recipe, not its internal cause or universal architecture superiority. The extension is closed.

## 1. Attention mathematics

Matched normalization comparison: [twenty pre/post-norm x warmup runs](../topics/normalization/norm-placement-results.md) give mean test accuracies 99.981%/99.958% for pre constant/warmup and 99.911%/89.356% for post constant/warmup. Only post warmup fails every-seed 99%, driven by seeds 659/683. All gradients/losses remain finite; all 20 checkpoints reproduce exactly. Same final normalization retained in both architectures; lower integrated LR under warmup and the two-block task limit interpretation. No general superiority claim or retuning.

Gradient-flow study: [five exact training replays with diagrams and plots](../topics/normalization/gradient-flow-results.md) record 20 states and reproduce all probe metrics after checkpoint reload. Residual gradients equal skip plus branch; opposing components and two local norm contractions are observed. LayerNorm VJP and directional finite differences agree. Mean probe loss falls from 3.115 to .001197; small late gradients do not by themselves establish a vanishing-gradient problem. No post-norm/deep-model comparison or general stability claim follows.

FFN cycle completed: [branch tests](../topics/ffn/ffn-branch-results.md) give 85.36% accuracy after zeroing block 1, 99.09% after zeroing block 2, and 91.52% after permuting block 2, versus 99.978% ordinary. All 35 summaries reproduce from saved logits. Sum-of-directions and local derivative checks pass; the raw W_out f logit-effect shortcut fails. The [exact normalization decomposition](../topics/ffn/ffn-normalization-results.md) separates direct removal from changed normalization statistics. This is local exploratory evidence, not fact storage or length-failure attribution. The subsequent training-time gradient study is linked above.

Final length boundary: [current frozen recipe at canonical +4](../topics/training/final-length-results.md) gives three/four/six-pair means 99.98%/92.20%/65.38%. Baseline criterion passes; every-seed 90% fails both longer lengths. All 15 summaries and rank breakdowns reproduce from saved logits. This is not a controlled comparison with earlier 80.43%/64.24% transfer scores because models/data differ. Close the branch and move to the [FFN learning guide and scoped investigation](../topics/ffn/ffn-investigation.md), without assuming an FFN cause.

Scoped resolution: [canonical position IDs](../topics/positions/canonical-position-results.md) give 99.978% mean accuracy on 400 further fresh mappings with one ordinary forward and no donor. Setting positions to 4..11 removes irrelevant global offsets by construction. Every seed passes 99%; all 40 summaries reproduce from saved logits. This resolves input handling for independent consecutive eight-token recall, not learned invariance, meaningful absolute positions or longer sequences. The global-offset cycle is closed.

Fresh joint-routing confirmation: [400 mappings absent from both prior matching-task corpora](../topics/retrieval/joint-routing-confirmation-results.md), same five frozen varied models, give ordinary/joint accuracy 77.51%/99.31% at +8 and 60.78%/96.99% at +12. Both predeclared donor-assisted criteria pass; ordinary robustness fails both. All 25 summaries reproduce from saved logits. This closes the diagnostic cycle, not the problem of generating reliable routing without a privileged donor.

Query/key localization: [five frozen varied models](../topics/attention/query-key-results.md) reach 97.49% +8 accuracy with +4 K-only, versus 79.86% with Q-only and 99.44% jointly. Reverse controls show greater harm from K. This supports stronger intervention sensitivity at block-2 keys, not a unique upstream cause. All 40 summaries reproduce from saved logits. The [joint-routing assessment](joint-routing-assessment.md) closes finer patches and links the completed fresh-mapping confirmation.

Position-training follow-up: [within-model routing interventions](../topics/positions/position-routing-results.md) raise varied +8 mean accuracy from 77.65% to 99.44% using +4 QK; V alone gives 78.29%. Reverse QK replacement lowers +4 accuracy to 78.36%. The [slot/answer control](../topics/attention/donor-slot-results.md) shows same-slot/different-answer rescue at 97.81%, whereas different-slot donors give about 96.6% agreement with recipient values at the donor-selected slot. This identifies routing transfer under the patch, not independent recipient retrieval. All 50 intervention summaries reproduce from saved logits. Upstream block-1 causation remains unproven.

Attention compares queries with keys and aggregates values. Understanding this operation separates implementation facts from claims about learned behavior.

\[
A=\operatorname{softmax}(QK^T/\sqrt{k}+M),\qquad Y=AV.
\]

Here k is key width; M blocks forbidden positions. Eight probes checked scaling, masking, permutation equivariance, derivatives, non-unique attention distributions, streaming accumulation, reassociation and dense storage. All passed. Their assumptions and numerical results are in [the concise explanations](../topics/foundations/test-clarifications.md) and [probe report](../topics/foundations/results.md).

**Limit:** arithmetic checks do not show learning, interpretation quality or architectural superiority.

## 2. Forward and backward verification

A correct forward pass computes predictions; a correct backward pass supplies derivatives for learning. We compared explicit block equations with a library block across three seeds and checked numerical derivatives:

\[
\frac{\partial f}{\partial\theta_j}\approx
\frac{f(\theta+\epsilon e_j)-f(\theta-\epsilon e_j)}{2\epsilon},\quad \epsilon=10^{-6}.
\]

Maximum output difference: **4.441e-16**. Maximum numerical block-parameter gradient error: **2.254e-9**. Causal and padding checks passed. See [methods and results](../topics/foundations/decoder-verification.md).

**Limit:** both tensor paths share PyTorch; finite differences supply an additional check. These tests cover selected small configurations, not every model setting.

## 3. Controlled learning

First-token recall asks whether a model can retrieve earlier information: `[2,0,3,1,0,1,query] -> 2`. Only the query prediction receives loss:

\[
L=-\frac1B\sum_{b=1}^{B}\log p_\theta(x_{b,0}\mid x_{b,0:n},\text{query}),
\qquad \theta\leftarrow\theta-0.1\nabla_\theta L.
\]

The prefix notation here denotes all n symbols. The one-block pre-norm decoder has **2,416 parameters**, width 16, two heads and FFN width 32. Five seeds each ran 600 SGD updates, batch 32, CPU float64. Train/validation/test sizes were 2,048/512/512 disjoint length-six sequences; a separate 512-example test used length ten. Criteria were fixed before training; no tuning followed.

| Outcome | Length six | Length ten |
|---|---:|---:|
| Mean accuracy | 100% | 95.27% |
| Seed range | 100%-100% | 82.03%-100% |
| Sample SD across seeds | 0 points | 7.52 points |
| Mean cross-entropy, nats | 0.001690 | 0.149264 |
| Every seed at least 95% | Passed | Failed |
| Constant training-majority baseline | 25.78% | 21.29% |
| Copy-last-symbol baseline | 27.54% | 24.41% |

Uniform guessing has expected accuracy 25%; exact first-symbol lookup achieves 100% by definition. The neural model is not necessary to solve this synthetic task. See [protocol](../first-token-protocol.md), [per-seed results](../topics/retrieval/first-token-results.md) and [raw measurements](../../results/first_token_training.json).

**Conclusion:** the model learned this task at the training length; reliable transfer to the tested longer length was not established. The cause of seed differences is unknown.

## 4. Comparison with papers

These comparisons concern scope and mechanisms. Different tasks, scales and training recipes prevent a meaningful leaderboard comparison.

| Primary source | Connection to our evidence | What we cannot claim |
|---|---|---|
| [Vaswani et al., sections 3-4](https://arxiv.org/html/1706.03762v7) | Our probes examine defined attention operations; our decoder differs from the original encoder-decoder. | Reproduced translation performance or a test of every length-generalization claim. |
| [Xiong et al., section 3](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf) | Motivates examining normalization placement; our baseline uses pre-norm. | Confirmed pre-norm advantage: we did not compare post-norm or warm-up schedules. |
| [Jain and Wallace](https://arxiv.org/abs/1902.10186); [Wiegreffe and Pinter](https://arxiv.org/abs/1908.04626) | Our weighted-sum counterexample illustrates why attention alone needs careful interpretation. | Reproduction or resolution of their empirical explanation debate. |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | Our streaming probe checks an accumulation identity relevant to memory-efficient attention. | Measured GPU speed, memory savings or reproduced its kernel. |

Reading depth varies: see [source register](../reference/sources.md) and [focused audit](../reference/foundation-audit.md). No paper benchmark has been reproduced.

## 5. Next evidence required

Inspection describes a pattern; an intervention tests whether altering it changes behavior. [Frozen-model interventions](../topics/retrieval/first-token-interventions.md) now show that seed 47's errors can be rescued by forcing head 0 to read the first symbol or restoring the query's training-position vector. Its head 1 attends strongly to the first symbol even on failures. Full causal decomposition and fresh-data confirmation remain pending.

Fresh-data update: [separate pathway interventions](../topics/positions/query-position-effects-on-attention-versus-the-residual-path.md) confirmed combined rescue on 1,024 new examples. Seed 47 reached 100% with residual-only replacement, versus 78.22% with attention-only replacement. Several donor positions rescued it; the effect is not unique to the training position. Its downstream mechanism remains unresolved.

Architecture changes were deferred during this first-token investigation; the later depth comparison is recorded below. Existing source, protocol, split and checkpoint hashes provide traceability; they do not replace an independent rerun.

Downstream decomposition: [skip-path, FFN and normalization tests](../topics/normalization/residual-effects-through-ffn-normalization-and-logits.md) show that seed 47's skip-only patch rescues all 210 errors while retaining original attention and FFN outputs. FFN-only rescues 127. Full rescue also survives using original final-normalization statistics. This decomposition reuses the pathway dataset and needs independent confirmation.

Confirmation update: [new examples at lengths 8/10/12/16](../topics/retrieval/skip-confirmation-results.md) preserve seed 47's 100% patched accuracy, but the all-model rescue criterion and the per-length control-advantage criterion both failed. The correction is not a reliable general repair across our tested models. Stop generalizing this mechanism; retain the seed-specific evidence and negative outcomes.

Independent task: [associative recall](../topics/retrieval/associative-recall-results.md) failed the 90% criterion across all five seeds. Mean three-pair accuracy was 54.16%, below a post-hoc frequency baseline of 55.57%; untouched-complement testing retained the negative result. Training accuracy was also low. Query/binding probes showed limited sensitivity but included overlapping training episodes, so they are exploratory rather than confirmation evidence. No architecture change was made.

Positive controls: [single-pair learning and 32-episode fitting](../topics/training/positive-controls-results.md) both reached 100% for all five seeds. The small-set models scored only 40.92%-44.04% on held-out three-pair episodes. Basic learning and small-set fitting work; the large-task failure remains unresolved between optimization, capacity and representation explanations.

Optimizer comparison: [ten matched SGD/Adam runs](../topics/training/optimizer-comparison-results.md) reproduced prior SGD metrics exactly. Adam's mean validation gain was 1.54 points; neither recipe approached 90% training accuracy. This particular optimizer change did not solve the task; optimization in general remains unresolved.

Depth comparison: [five two-block and five wider one-block runs](../topics/training/depth-comparison-results.md), with the prior Adam baseline reused, reached mean validation accuracy of 100%, 55.78% and 55.90%, respectively. Every deep seed also reached 100% training accuracy. Both predeclared depth criteria passed; this wider one-block configuration did not resolve the failure. Sequential depth helps under this recipe, with approximate parameter control but unmatched compute. Validation was already inspected; independent confirmation, length transfer and the learned mechanism remain open. Consult the [failure register](failure-register.md) before repeating any failed recipe.

Replication: [ten new runs on regenerated splits](../topics/training/depth-confirmation-results.md) confirmed 100% deep versus 54.16% wide three-pair test accuracy. Deep transfer means fell to 80.43% at four pairs and 64.24% at six; both 90% transfer criteria failed. Longer-sequence log loss was worse for deep than wide models despite better accuracy. New splits are internally disjoint by mapping/query, but overlap earlier research and share mappings across queries. This is local replication, not independent reproduction or proof of unseen-mapping generalization.

Frozen-model diagnostics: [position resets and distractor removal](../topics/positions/position-distractor-results.md) show that tail-position resets alone fail 90% rescue. Removing distractors while preserving remaining positions gives 95.18% / 89.69% mean accuracy at four/six pairs; adding tail reset gives 96.05% / 95.20% and passes every seed for the specified first-two crop. These inspected-data interventions support interacting context/position effects, not a unique causal mechanism.

Subset confirmation: [every eligible crop on fresh long episodes](../topics/retrieval/subset-confirmation-results.md) reduced removal-plus-tail-reset mean accuracy to 87.43% / 87.66%; both predeclared robustness criteria failed at both lengths. The earlier crop rule still performs well, so its rescue must not be generalized across subsets. Compacting the same crops gives 100% / 99.996%.

Fixed-content layouts: [eight offset/gap/tail combinations](../topics/positions/layout-results.md) show six-pair-derived crop accuracy of 97.50% with only original tail positions, 88.01% with only pair offsets, and 92.17% with only gaps, versus 99.996% compact. Effects are conditional, not additive.

Global shifts: [moving every position together](../topics/positions/shift-results.md) preserves relative distances but reduces six-derived mean accuracy to 60.60% at +4 and 86.25% at +6. Logit invariance failed at every positive shift; all-seed 90% passed only at +1. Absolute-position sensitivity is established under this intervention; relative distances may still matter.

Activation replacements: [compact donors inserted into shifted runs](../topics/retrieval/activation-results.md) show 99.984% mean accuracy for block-2 attention replacement, versus 60.60% shifted baseline. Every seed passes 90%, but four correct predictions are harmed. Block-1 attention, residual and FFN replacements fail that criterion. Donors contain upstream computation, so this identifies a sufficient rescue point, not a unique error source.

Donor controls: [reverse and unrelated replacements](../topics/retrieval/donor-controls-results.md) reduce compact accuracy to 58.71% in reverse, but same-answer unrelated donors still give 99.36% accuracy. Different-answer donors produce 99.34% donor-answer agreement and only 0.18% recipient accuracy. Matching-example specificity failed (0.625-point advantage). Whole-output rescue may transfer an answer rather than restore recipient retrieval.

Q/K versus V: [15 interventions per seed](../topics/attention/qkv-results.md) give 99.15% for matched compact QK with shifted V, versus 61.13% for matched V alone and 60.60% baseline. Every seed passes 90% only for matched QK/QKV; K alone fails one seed. Unrelated QK does not rescue, even with the same answer. Shifted V supports near-correct predictions under matched routing, but upstream origin and head/token specificity remain unresolved. The report includes a short learning guide and equations.

Head/token localization: [13 conditions per seed](../topics/attention/head-token-results.md) show final Q plus K at all value-token slots reaches 98.66%, every seed >=90%. Either head alone and target-value-only K replacement fail that criterion. Non-final Q replacements have no final-logit effect, as expected from the final block's structure.

Source competition: [target/other value K replacements with fixed compact Q](../topics/attention/target-distractor-results.md) give 89.96% target-only, 70.70% others-only and 98.66% combined, versus 63.57% Q-only reference. Reverse target corruption drops compact accuracy to 70.91%; reverse others to 89.98%. Both groups influence scores and predictions under these interventions. Only combined restoration meets every-seed 90%.

Block-1 construction: [mixed residual/attention/FFN contributions projected into value-position K](../topics/attention/block1-key-results.md) show 94.75% with compact attention alone and saved shifted FFN, or 95.60% with FFN recomputed. Every seed passes 90%; residual/FFN-only replacements fail. Final Q is compact and other recipient paths stay shifted.

Source interventions: [paired-key forcing at value rows 3/5](../topics/attention/block1-source-results.md) raises shifted-source construction accuracy from 63.57% to 97.68%; self/other-key controls fail. Compact paired-key removal drops 94.75% to 53.95%. Observed attention shifts toward nonpaired earlier keys under +4 positions. This supports source involvement under intervention but retains compact final Q and artificial downstream construction.

Direct forcing: [normal downstream execution without compact donors](../topics/retrieval/direct-pair-results.md) reaches 93.48% mean shifted accuracy from 60.60%. Seed 223 reaches 82.79%, failing every-seed 90%. Control advantages and compact-preservation tolerance passed, but 505 shifted correct predictions were harmed. This is partial rescue, not a reliable general repair.

First-value diagnostic: [forcing all three value positions](../topics/retrieval/first-value-results.md) gives 93.31% shifted mean and 81.09% for seed 223. The every-seed criterion still fails. Compact-preservation tolerance passes, while self forcing harms performance. The omitted row is not a sufficient explanation.

Fixed-rule confirmation: [fresh parents across six shifts](../topics/retrieval/pair-confirmation-results.md) retains every-seed 90% at +1/+2 but fails at +4/+6/+8. Mean forced accuracy is 92.48%, 91.00%, 85.97% at those failing shifts. Source-control advantages and compact-preservation criteria pass. Parents are new, but 1,956/5,120 crops overlap earlier diagnostic identities and 3,890 overlap training identities. Gains are repeatable within this scope; general shift robustness remains unestablished.

Error strata: [query ranks and value patterns](../topics/retrieval/error-strata-results.md) show failures across all ranks, particularly seeds 223/257. At +8, accuracy is 82.53% for distinct values, 90.00% with target twice, 96.68% for all equal. Repeated answers can mask incorrect retrieval; this is correlation, not improved binding demonstrated. Of 3,592 errors, 705 predict absent values and none invalid tokens.

Matched value edits: [balanced counterfactual families](../topics/retrieval/value-counterfactual-results.md) raise +8 forced accuracy from 81.98% distinct to 95.63% all equal, while target-slot attention slightly decreases on average. Wrong-value repetition gives 81.01%. Target-top frequency rises modestly, so routing is not unchanged. The all-seed >=5-point gain criterion fails (seed 163: +2.78). Answer redundancy helps under these edits but does not prove better retrieval or complete mediation. Consolidate before further patching.

Cycle assessment: [mechanism conclusions](mechanism-assessment.md) are consolidated. [Strict distinct-value evaluation](../topics/retrieval/strict-retrieval-results.md) uses 16 balanced mappings wholly absent from training, with all 21 eligible mappings supplementary. The fixed intervention passes every-seed 90% at +1/+2 but fails +4/+6/+8; +8 mean is 82.50% versus 60.49% unchanged. Benefit survives distinct values and training-mapping exclusion, but broader robustness does not. Most eligible mappings appeared in earlier diagnostics. Close this cycle without a universal repair claim; reserve a new distribution before fresh confirmation.

New reserved cycle: [five fresh training runs on twelve keys/six values](../topics/training/reserved-cycle-results.md) achieve 99.986% mean unshifted test accuracy on 400 mappings reserved before training. The fixed intervention gives +8 accuracy 77.71% versus 55.33%, but all-seed 90% still fails at +4/+6/+8. Compact preservation also fails for seed 433 (1.71-point loss). Test evaluation began only after all final checkpoints were saved; no tuning followed. Benefit and unreliability both persist under the new combined data conditions.
