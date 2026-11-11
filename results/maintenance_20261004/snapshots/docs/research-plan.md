# Research protocol and decision tree

Current branch closure: [marked RoPE alignment diagnostic](topics/benchmarks/copy-marker-alignment-results.md). Saved models, zero updates. Supplied matching reads recover 100% longer copying and stopping in all tested seeds/groups; shifted reads give 0%. Native transfer remains weak. All 72 cells pass independent equations. [Eight-study synthesis](assessments/copy-generalization-assessment.md) consolidates the structural boundary, exposure/objective comparisons, marked learning and conditional recovery. End copy investigations here without more training or routing subdivisions. Training's routing cause and a deployable repair remain unknown. No new study selected automatically; historical plans below are not an active queue.

Current learning closure: [source-marker training](topics/benchmarks/copy-marker-training-results.md), six completed matched 6,000-update runs. Every training group is 100%. RoPE also achieves 100% random and uniform copying at lengths3/4/5/6; ordinary length8/10 means are only 10.42%/2.34%, versus sinusoidal 0.78%/0%. Both extrapolation rules and the frozen long-gain rule fail. All 120 cells pass independent equations with zero prediction differences. This answers learning with a marker separately from the prior inference-only diagnostic. Both arms are marked on opened pools; historical scores cannot isolate its training effect. Close the fixed budget without tuning, replacement seeds or automatic extension. [Seven-study assessment](assessments/copy-generalization-assessment.md). Historical entries below are an archive, not an execution queue.

Current closure: [copy positional representation](topics/benchmarks/copy-rope-results.md), six completed fresh paired runs. Both training-fit gates pass; RoPE gives only small long-copy gains and fails benefit, seen-length preservation and extrapolation rules. All 84 cells pass independent equations with zero prediction differences. A separate posthoc uniform-source witness proves a length ambiguity for this local Q/K-only positional variant; it does not explain all varied-content failures. [Learning guide](topics/benchmarks/copy-rope-guide.md), [six-study assessment](assessments/copy-generalization-assessment.md). No next experiment selected or tuning authorized automatically. Earlier objective-weight, routing and exposure branches remain closed. The [direction review](assessments/research-direction-review.md) favors explanatory questions over coverage. Historical entries below are an archive, not an active queue.

Latest closure: [MoE load balancing](topics/ffn/moe-balancing-results.md). Arithmetic and counterexamples checked; no training or next experiment selected.

Current closure: [MoE routing arithmetic](topics/ffn/moe-routing-results.md), with [educational guide](topics/ffn/moe-routing-guide.md). The recommended audit below is now complete. No next experiment selected.

Current planning status: [coverage review](assessments/architecture-coverage-review.md). Memory studies remain closed. Recommended next is a bounded MoE routing mathematical audit; no new experiment is launched by this review. Older cycle plans below are historical, not an active execution queue.

Latest closure: [checkpoint-only reset diagnostic](topics/attention/memory-reset-diagnostic-results.md). State dependence survives; class answers remain correct with forgetting disabled. No retraining or automatic extension.

[Learned global relay](topics/attention/global-retrieval-results.md), executed and closed: nine matched6000-update full/local/global runs. Every arm passes eligible training/validation/test17 gates. Global test21 distant accuracy is100%/100%/83.33% across seeds; the all-seed reliability and combined dependence criteria fail. Removing global edges reduces distant accuracy to1/6 in every test cell. All48 native/ablated evaluations and80,640 predictions pass independent audit with zero argmax changes. Useful relay and a remaining transfer gap coexist; cause unisolated. No tuning or automatic extension selected.

[Global-token audit](topics/attention/global-attention-results.md), executed and closed:96 Q/K/V numerical cases,96 reach cases and64 counts pass. Bidirectional global connections give two-layer all-pairs paths. Causal global routes require source<=global<=query; a first global token cannot summarize later inputs. An intentionally unmasked global query leaks future information to an earlier ordinary query after two layers; the correct mask prevents it. Shared Q/K/V only, no training or speed claim. No automatic extension selected.

[Learned local retrieval](topics/attention/window-retrieval-results.md), executed and closed: three matched full/local pairs,4544 parameters,6000 updates and360,000 examples each. Every seed scores100% at distances1/3/4 across training, validation and length9/13 tests. Full attention also scores100% at5/8; local scores exactly1/6 with invariant six-answer-family logits. All24 pools and40,320 predictions pass independent audit with zero argmax changes. Local relay is learned for this task; the longer test preserves relay distances. Dense masking gives no sparse-speed claim. No tuning or automatic extension selected.

[Sliding-window arithmetic](topics/attention/sliding-window-results.md), executed and closed:48 Q/K/V output/gradient cases,18 reachability cases and40 pair counts pass. Missing window, future-key and boundary-key errors are detected. At depth L and causal radius r, possible lookback is Lr under the declared local-only operations. Paths permit influence but do not establish learned use. Pair counts are not measured speed or memory savings. No training or automatic extension selected.

[Copy stopping diagnostic](topics/benchmarks/copy-stopping-results.md), closed: saved checkpoints, no training. Blocking early EOS gives six-symbol content/full accuracy1.56%/0.52%, and eight-symbol0%/0%. Correct-prefix predictions also fail. All24 pools and15,798 intervention tokens pass independent audit. Early stopping alone is insufficient to explain away the content failure; its cause remains open. No tuning or next experiment selected.

## Current direction

Latest learning study: [encoder-decoder copying](topics/benchmarks/encoder-decoder-copy-results.md), executed and closed. Three6,000-update CPU float64 runs fit all training examples exactly. Exact test means96.88%/90.63% at3/4 symbols;0% at6/8. Early EOS dominates long outputs. The combined95% training/validation gate fails because validation groups are weak; training fitting succeeds. All24 pools,14,976 teacher-forced targets and12,686 generated tokens pass independent audit. [Guide](topics/benchmarks/encoder-decoder-copy-guide.md), [protocol](protocols/encoder-decoder-copy.md). Causes, learned mechanism and broader generation remain open. No extension or decoding repair selected.

Latest correctness study: [complete encoder-decoder contract](topics/foundations/encoder-decoder-results.md), executed and closed. All18 CPU float64 cases pass native/independent logits, masked loss, gradients and135 identical-prefix decoding comparisons. Source padding through the encoder and decoder, target-loss masking, shifting and error controls pass. This is one initialized encoder/decoder layer, not learned or free-running generation. [Guide](topics/foundations/encoder-decoder-guide.md), [protocol](protocols/encoder-decoder-contract.md). No training study selected.

Latest component study: [cross-attention correctness](topics/attention/cross-attention-results.md), executed and closed. All36 numerical cases pass; source padding and target causality hold within the tested memory-interface/two-attention assembly. Deliberate mask and shifting errors are detected. The first one-sided padding error fixture was ineffective in one case; its source/report are preserved and both perturbation signs now expose the error. [Guide](topics/attention/cross-attention-guide.md), [protocol](protocols/cross-attention.md). Full encoder-decoder training and target-loss masking remain outside scope. No new study selected.

Latest completed diagnostic: [GPU learning adequacy](topics/benchmarks/composition-learning-adequacy-results.md), three new pairs of approximately60K parameters, batch1,024,3,000 updates/3.072M examples per run. Both one-step fit gates fail despite stronger observed lookup. Six-edge validation means91.15% Transformer/35.11% GRU. All42 pools and626,688 predictions pass independent audit; three precision changes leave decisions stable. [Guide](topics/benchmarks/composition-learning-adequacy-guide.md), [protocol](protocols/composition-learning-adequacy.md). Reused data and several joint recipe changes limit causality. Closed at its budget; no extension or new experiment selected.

Latest completed study: [one-step/two-step graph lookup](topics/benchmarks/recurrent-composition-results.md), three fresh pairs,6,000 updates each. Both short fitting and reliability fail; two-step validation means22.43% Transformer/26.95% GRU, with every paired gain below the frozen10-point rule. All42 full pools and626,688 final predictions pass independent audit. One-step control failure prevents a composition-specific explanation. [Guide](topics/benchmarks/recurrent-composition-guide.md), [protocol](protocols/recurrent-composition.md). [Five-task consolidation](assessments/capability-summary.md) is complete. Both recipes still improve at the budget limit; study closed without tuning. No further experiment selected.

Latest completed study: [GRU versus Transformer language prediction](topics/benchmarks/recurrent-language-results.md), three fresh pairs,6,000 updates each. Transformer validation PPL396.64 versus GRU597.34,33.60% lower, with every pair improving. All models beat unigram; all24 pools and4,645,902 losses pass independent audit, with four cross-precision top1 differences recorded. The validation margin and frozen baseline guard pass. Training fit uses a fixed subset; losses still fell at the final interval, so convergence and eventual ranking remain open. Study closed; no tuning or broader reasoning study selected. [Guide](topics/benchmarks/recurrent-language-guide.md), [protocol](protocols/recurrent-language.md).

Latest completed comparison: [GRU versus Transformer retrieval](topics/benchmarks/recurrent-retrieval-results.md), three fresh seed pairs,6,000 updates. Transformer six-pair R/U-balanced mean97.82%, GRU23.36%. Transformer reliability and consistent advantage pass; GRU short training fit fails. All42 full-pool evaluations and1,023,840 predictions pass independent audit. Study closed; no language/reasoning comparison or tuning follows automatically. [Educational guide](topics/benchmarks/recurrent-retrieval-guide.md), [frozen protocol](protocols/recurrent-retrieval.md).

The [GRU/Transformer comparison](topics/benchmarks/recurrent-state-results.md) is executed and closed. All three GRU seeds reach100% through24 operations; the scoped performance rule passes. Transformer short fitting remains incomplete, so the gain does not isolate extrapolation or recurrence alone. All144 final-checkpoint cells pass independent verification. [Educational guide](topics/benchmarks/recurrent-state-guide.md), [frozen protocol](protocols/recurrent-state.md). No additional experiment is selected; prior studies remain valid.

Completed matched comparison: [intermediate-state supervision](topics/benchmarks/state-supervision-results.md), six fresh runs with identical paired initial states and input streams. All144 independently checked final-checkpoint cells pass. Training/short-length means improve, but gains vary across seeds and length transfer fails. The fixed validation decision fails; study closed. [Guide](topics/benchmarks/state-supervision-guide.md). Reused splits, additional labels and changed objective scale limit attribution. Prior studies stay closed; no new comparison is selected.

The authorized [state-tracking training-fit diagnostic](topics/benchmarks/state-tracking-results.md#follow-up-did-the-models-fit-their-training-examples) is complete. Saved weights were evaluated without updates: training performance is also low for no-reset/long-tail groups, especially at12 operations. This narrows the finding but does not identify its cause. No new training comparison is selected.

The component-comparison phase is closed. The bounded [state-tracking study](topics/benchmarks/state-tracking-results.md) is also executed and closed: three fixed seeds,3,000 updates each,13,824 audited predictions. Reliability fails already at seen lengths without resets. [Educational guide](topics/benchmarks/state-tracking-guide.md), [frozen protocol](protocols/state-tracking.md). No component sweep, extra seeds or automatic training extension follows. No new experiment is currently selected.

## Historical research log

The entries below preserve earlier decisions and their evidence. Their words "latest", "next", "pending" and "unverified" describe their original stage, not today's work queue. Current hardware evidence includes native Windows GPU attention; associative recall and key-overwrite studies have already run. Only the current-direction section selects new work. Frozen protocols remain unchanged.

Latest completed study: [weight tying](topics/embeddings/weight-tying-results.md), under [Decision0006](decisions/0006-wikitext2-weight-tying.md), saves 49.82% parameters but worsens validation PPL 6.41%, exceeding the 5% average tolerance; every paired seed loses. Reused test worsens 7.12%. All six runs and 2,779,278 independently recomputed token losses pass; one CPU/GPU argmax differs descriptively. Initial tensors/predictions match, with embedding initialization changed in both controls. [Educational guide](topics/embeddings/weight-tying-guide.md). Study closed, original recipe retained; no tuning or automatic follow-up.

Latest completed transfer study: [gain-only LN/RMS on WikiText-2](topics/normalization/wikitext2-norm-results.md), under [Decision0005](decisions/0005-wikitext2-normalization.md), with ReLU/FFN32 fixed. All six finite runs and2,779,278 independently recomputed validation/test token losses pass, with zero argmax differences. RMS improves validation PPL1.60% in aggregate and every pair gains, but misses the predefined2% adoption margin; reused test improves1.34%. Study closed, original recipe retained, no tuning or automatic follow-up. The [FFN assessment](assessments/ffn-assessment.md) closes prior activation/width/gating comparisons without pooling unmatched cohorts.

Latest completed comparison: [SwiGLU versus ReLU](topics/ffn/swiglu-results.md). [Decision0004](decisions/0004-swiglu-matched-ffn.md) matches bias-free FFN projection cost and fixes MQA attention. Six runs, thirty scalar/matrix math cases, six preflights and 2,779,278 independently audited validation/test losses pass. SwiGLU worsens geometric mean validation PPL2.00% with every pair worse; adoption fails. Reused test worsens1.83%, descriptively. Original recipe retained; study closed without tuning or automatic follow-up.

Previous external comparison: [WikiText-2 with matched MHA/MQA](topics/benchmarks/wikitext2-results.md). All six frozen attempts and 1,473,408 independent test-token loss checks pass. MQA meets [Decision0003](decisions/0003-wikitext2-matched-baseline.md)'s tradeoff rule, with 1.96% worse geometric mean perplexity and half potential compact K/V payload. Every paired seed loses quality. Study closed; no tuning, budget extension or automatic follow-up.

Previous different-task check: [binary majority](topics/benchmarks/majority-transfer-results.md), five seeds and20 independent evaluations. All score100% at9/11/15 bits; at21 bits seed1627 scores86.25%, failing the fixed transfer rule. Errors are confined to label1 one-bit majorities. The architecture learns aggregation beyond recall, with a bounded reliability failure. Study closed; no local tuning. The external comparison above supports a scoped tradeoff, not broader architecture superiority.

Previous trained comparison: [LayerNorm versus RMSNorm](topics/normalization/rmsnorm-training-results.md). Ten attempts and30 independent evaluations pass. RMS six-pair unique mean91.71% to94.04%, but seed1543 misses90% and two paired seeds lose more than1pp. The fixed adoption rule fails. [Normalization is closed](assessments/normalization-assessment.md); [consolidated recipe choices](assessments/recipe-assessment.md) include the completed different-task check. No further normalization tuning is selected.

Latest mathematical study: [LayerNorm versus RMSNorm](topics/normalization/rmsnorm-math-results.md). All68 cases and the independent Decimal audit pass. Centering removes a common-offset direction locally; equal outputs at a zero-mean point do not imply equal gradients. Positive epsilon breaks exact scale invariance. The initial finite-difference failure and corrected step rule are archived. Study closed; training/speed superiority remains untested.

Latest numerical study: [cache precision](topics/attention/cache-precision-results.md). Fixed reused subset, ten models,180 audited cells. FP32 preserves all final predictions; FP16 changes1 and symmetric INT8 K/V changes9 of14,400 decisions per mode. All changes occur at six pairs. Storage savings include scale metadata; no speed or fresh generalization claim. Study closed.

Latest execution comparison: [MQA GPU cached decoding](topics/attention/mqa-decoding-results.md). Three fresh processes give 36 passing numerical checks and an independent audit. Compact cache halves, but none of six shapes passes the fixed repeatable 10% speedup rule. Temporary tracked allocation is higher for MQA at batch16. Causes of timing variability remain unisolated. Study closed; no automatic sweep.

Latest trained comparison: [MHA versus MQA](mqa-training-results.md). Five paired seeds pass the fixed resource/quality rule: compact K/V payload halves and parameters fall 10.69%. Six-pair mean unique accuracy falls 94.73% to 94.05%; worst seed loss is 4.111pp. Every MQA query-type score stays above90%. All30 independent evaluations pass. Accept MQA for this recipe/tolerance, with measured losses retained. Comparison closed.

Latest mathematical probe: [shared KV heads](shared-kv-results.md). Four-head payload is 80/40/20 KiB for MHA/GQA/MQA at fixed dimensions. Already-shared controls match; mean pooling different heads changes outputs. Shared keys constrain joint log odds within each group. All four probes and the 1,969-scalar independent audit pass. This establishes scoped arithmetic properties, not trained quality or latency. Study closed.

Previous implementation check: [KV-cache equivalence](topics/attention/kv-cache-results.md). All 120 cases and the independent audit pass; maximum cached/full logit error is 4.75e-14. Reset-position and omitted-entry controls are detected in every case. This verifies inference correctness and K/V payload under the fixed conditions; speed and model quality were not compared. Study closed; no automatic follow-up selected.

Previous comparison: [broader training queries](topics/training/query-exposure-results.md). With architecture and context streams fixed, unique-key mean accuracy rises 74.32% to 99.89% at four pairs and 59.53% to 94.83% at six. Every paired seed improves. One six-pair seed scores 87.77%, failing the fixed all-seed reliability rule. All 30 independent evaluations pass. This supports a training-coverage benefit, not an architecture change. Comparison closed.

Previous diagnostic: [fixed-context query switching](topics/training/query-switch-results.md). Five saved models achieve81.59% mean on previously untrained unique-key queries; only two seeds reach90%. The models are not completely query-blind on these inputs, but reliable transfer fails. All predictions checked against independent attention; no retraining. Diagnostic closed.

Previous comparison: [FFN hidden width32 versus64](topics/ffn/ffn-width-results.md), with RoPE/ReLU fixed. Mean six-pair accuracy rises 85.86% to 91.44% (+5.59pp), with41.51% more parameters. Four seeds improve; one loses7.50pp and scores85.90%, failing the all-seed reliability/no-worsening rules. Median gain is0.68pp. Retain width32 under the stopping rule, not as a claim of superior accuracy. All30 independent evaluations pass. Four/six-pair data is fresh; three-pair data is reused diagnostic evidence. Comparison closed.

Interpretation of the request: build a continuing, falsifiable research programme. The first cycle establishes sources, definitions and mathematical probes. “All evidence” cannot mean a finite proof about every architecture, dataset and training scale; each subsequent cycle must name the question and its stopping criterion.

```text
Candidate claim
|-- Is it mathematically precise?
|   |-- No -> define variables, scope and observable outcome
|   `-- Yes -> derive or locate proof; inspect assumptions
|-- Is it an empirical claim?
|   |-- No -> check derivation with independent numerical witnesses
|   `-- Yes -> pin paper/code/data; predefine comparison and failure criterion
|-- Does a controlled experiment support it?
|   |-- No -> diagnose implementation, task, tuning and claim separately
|   `-- Yes -> repeat seeds, test counterconditions and held-out tasks
`-- Is an improvement supported?
    |-- No -> retain negative result and refine question
    `-- Yes -> independent reproduction and prior-art search before novelty claim
```

## Cycle 1: mathematical foundations

S01-S08 in `experiments/attention_spikes.py` use deterministic synthetic fixtures and seeded random sampling. Thresholds are specified in code before execution. Controls check that removing the relevant condition actually changes the outcome. These are demonstrations and implementation checks, not full paper replications.

## Cycle 2: baseline verified; first controlled task complete

First controlled training completed: [five-seed first-token retrieval](topics/retrieval/first-token-results.md). All seeds passed the same-length criterion; the longer-length criterion failed because one seed scored 82.03%. No tuning followed. Next investigate the saved models' behavior, then expand to the remaining task families. This does not close the broader training programme.

Completed foundations: [numerical walkthrough](topics/foundations/numerical-walkthrough.md), [focused paper audit](reference/foundation-audit.md), and [forward/backward verification](topics/foundations/decoder-verification.md). The [evidence summary](assessments/evidence-summary.md) records the scope and limitations in one place.

Frozen-model inspection and initial interventions are [complete](topics/retrieval/first-token-interventions.md). [Pathway separation on fresh data](topics/positions/query-position-effects-on-attention-versus-the-residual-path.md) now supports a residual-only rescue for seed 47. Next trace downstream FFN/normalization effects before architecture changes. GPU runtime, performance and broader configurations remain unverified; current experiments use the pinned CPU environment.

Remaining task families: associative recall, key overwrite, full sequence copy and state tracking. For each, fix disjoint splits, baselines, ambiguity rules and success criteria before training. Test same-length performance and length transfer separately. Synthetic task results must not be relabeled as language understanding.

Associative-recall update: [five runs, diagnostics and untouched confirmation](topics/retrieval/associative-recall-results.md) failed the learning criterion. [Positive controls](topics/training/positive-controls-results.md) passed simple learning and tiny-set fitting. A [matched SGD/Adam comparison](topics/training/optimizer-comparison-results.md) did not solve the plateau. The subsequent [fixed depth comparison](topics/training/depth-comparison-results.md) reached 100% train/validation for all five deep seeds, versus 55.78% mean validation for the wider one-block control. Both depth criteria passed; this is reused-validation evidence with unmatched compute, not a mechanism proof. Preserve the [failure register](assessments/failure-register.md). Key overwrite, full copy and state tracking remain pending.

The [new reserved cycle](topics/training/reserved-cycle-results.md) is complete: five new models, expanded vocabulary, distinct values and whole-mapping reservation before training. The fixed forcing rule retains mean shift gains, fails larger-shift all-seed robustness and newly fails compact-preservation tolerance. No tuning followed. Keep the rule as a diagnostic and stop expanding patches. Any next cycle needs a distinct bounded question and newly reserved evaluation; these results do not establish architecture novelty or general superiority.

## Cycle 3: controlled architecture challenges, depth comparison complete

| Intervention | Question | Required control |
|---|---|---|
| Remove sqrt(k) or learn a temperature | When does fixed scaling help optimization? | Logit variance, entropy, gradient norms; equal tuning budget |
| Pre-norm versus post-norm | Does placement affect stability at this depth? | Same initialization family, depth, data and optimizer sweep |
| Head count / shared KV heads | What capacity and cache cost are traded? | Report actual parameter counts, memory and quality |
| Additive versus rotary positions | Which lengths and tasks generalize? | Same train lengths; unseen lengths and positions |
| Dense attention versus recurrent memory | Where does compressed state lose information? | Sweep state capacity, collisions, overwrite frequency and query distance |
| Hybrid attention-memory | Can explicit retrieval and compressed state complement each other? | Pure attention, pure recurrence, fixed hybrid and gating ablations |

Proposed initial training evaluation: five seeds per configuration; disclose every seed, failure and tuning trial. Select hyperparameters on validation only. Primary outcomes are held-out accuracy and cross-entropy per task. Report mean, standard deviation and paired differences across seeds, with uncertainty intervals justified by the sampling unit. Avoid treating individual tokens as independent training runs.

Compare both parameter-matched and compute-budget-matched configurations; these answer different questions and cannot generally be satisfied simultaneously. Keep data/tokenizer and tuning budgets fixed. Log tokens processed, estimated FLOPs, wall-clock time, peak allocated memory, dtype, kernel, software versions and hardware. Separate training, prefill and token-by-token decoding. Warm up and synchronize GPU measurements; do not infer device performance from Python loops.

Predeclare a meaningful improvement margin before each training run, after the task metric and deployment constraint are selected. Use an untouched confirmation set after exploratory tuning. A single winning seed, unreported baseline tuning, or a quality gain from extra compute does not support an unqualified architecture claim.

## Cycle 4: interpretation and candidate improvements, pending

Measure attention/value contributions and intervention effects on trained tasks. Try to predict the effect of an intervention before observing it; compare against random and magnitude-matched controls. Separate correlation, local causal effects and a transferable mechanism.

No candidate architecture is selected. Establish and understand the baseline before formulating an improvement hypothesis. Existing hybrid and memory papers remain literature to audit, not a proposal for this project.

Other unresolved questions: when learned score normalization beats fixed scaling; how state capacity interacts with interference; whether interpretability constraints cost predictive quality; and which kernel costs erase theoretical savings. Their answers depend on data and hardware and are unknown here.

Before calling a solution novel: review direct predecessors and citation trails, compare equations and algorithms rather than names, document the exact difference, and seek an independent reproduction. Before calling it better: name the tasks, resource regime, uncertainty and losing cases. Retain failed hypotheses.

## Evidence closure for one question

A question is ready for a bounded answer when its terms are fixed, sources and counterarguments reviewed, implementation independently checked, relevant controls and repeated runs recorded, and limitations explicit. If evidence conflicts, the answer is unresolved. This rule closes individual questions without claiming the whole field is settled.


Latest educational follow-up: [trained Q/K/V walkthrough](topics/retrieval/trained-recall-walkthrough.md) is complete and verified against archived predictions and independent attention. It traces two already-evaluated examples; no new training comparison or causal intervention was performed.
