# Learning guides

Recent reports with integrated explanations and equations:

- [How exposure can improve an existing weak solution](../topics/benchmarks/copy-exposure-continuation-results.md)
- [Why whole cross-attention outputs may not transfer between models](../topics/benchmarks/copy-cross-output-transfer-results.md)
- [What a within-model attention-pattern intervention establishes](../topics/benchmarks/copy-head-pattern-transfer-results.md)

Dedicated guides and earlier explanations:

- [Separating positional exposure in self-attention and cross-attention](../topics/benchmarks/copy-position-sites-guide.md)
- [Why spaced positions help longer copying](../topics/benchmarks/copy-position-exposure-guide.md)
- [Training-map reuse, new associations and forgetting](../topics/attention/map-exposure-guide.md)
- [Why accuracy can improve while mean margins fall](../topics/attention/fitting-trajectory-guide.md)
- [What fitting, routing/decoding and logit scale actually explain](../assessments/retention-learning-causal-account.md)
- [Cross-block action: changing directions while retaining global curvature](../topics/attention/fitting-transport-guide.md)
- [How curvature transports gradients between routing and decoding](../assessments/fitting-transport-review.md)
- [Decoder scale versus non-scalar change: same answers, different loss](../topics/attention/decoder-scale-shape-guide.md)
- [An exact split between routing and value/class decoding](../topics/attention/fitting-representation-guide.md)
- [Gradient, curvature direction and line-search length](../topics/attention/fitting-direction-guide.md)
- [Full-pool L-BFGS, curvature history and exposure-matched Adam](../topics/attention/full-pool-fitting-guide.md)

- [Classifier norm constraints, unchanged instant winners and changed future gradients](../topics/attention/readout-norm-cap-training-guide.md)

- [How beta and learned key geometry affect different parts of retrieval](../topics/attention/beta-key-transport-guide.md)

- [Target-write transport, competing content and temporary answer damage](../topics/attention/write-interference-stages-guide.md)

- [Prediction accuracy versus perfect affine decodability during learning](../topics/attention/readout-conflict-stages-guide.md)

- [Coordinate conditioning and the limits of changing only the affine head](../topics/attention/shared-readout-recovery-guide.md)

- [Fixed memories, a different classifier: positive witnesses and exact certificates](../topics/attention/shared-readout-feasibility-guide.md)

- [Single-query feedback: signed coefficients and shared-value gradients](../topics/attention/single-query-write-opposition-guide.md)

- [Shared value embeddings: within-history versus between-history opposition](../topics/attention/write-history-opposition-guide.md)

- [Write credit assignment: state transport and shared-value aggregation](../topics/attention/write-credit-assignment-guide.md)

- [Shared-state preservation: local value correction and other-key damage](../topics/attention/shared-state-preservation-guide.md)

- [Shared-state feasibility: decoding, writing and learning are separate questions](../topics/attention/shared-state-feasibility-guide.md)

- [One-time first-moment removal: what changes and what it proves](../topics/attention/memory-first-moment-reset-guide.md)

- [Adam memory movement: current gradients, stored moments and actual error loss](../topics/attention/adam-memory-error-transport-guide.md)

- [Gradient opposition versus net harm in memory parameters](../topics/attention/memory-gradient-competition-guide.md)

- [Target-only histories: separating clean decoding from competing writes](../topics/attention/target-only-history-guide.md)

- [How Adam carries classifier expansion across minibatches](../topics/attention/readout-adam-transport-guide.md)

- [Scale derivatives: correct answers, errors and their gradient balance](../topics/attention/readout-scale-pressure-guide.md)

- [Why classifier scale changes confidence without changing answers](../topics/attention/readout-scale-control-guide.md)

- [Accuracy, true-class probability, wrong-winner confidence and loss contributions](../topics/attention/feedback-loss-decomposition-guide.md)

- [Why resume weights, Adam moments and data streams together?](../topics/attention/query-feedback-continuation-guide.md)

- [Same expected objective and answer count, different query feedback](../topics/attention/query-feedback-training-guide.md)

- [Why one query gives a noisy beta-control gradient](../topics/attention/query-gradient-variability-guide.md)

- [Retention learning: objective pressure, chronology and Adam](../topics/attention/retention-learning-replay-guide.md)

- [Training-objective pressure, gradients and retention](../topics/attention/objective-beta-pressure-guide.md)

- [Active learning-mechanism questions and how to interpret gradients](../assessments/learning-mechanism-agenda.md)
- [RL foundations, PPO, RLHF and GRPO roadmap, parked](../topics/rl/rl-roadmap.md)
- [Saved language models: earlier context and recent order](../topics/attention/language-context-use-guide.md)

- [Beta, cross-key corrections and immediate versus final answers](../topics/attention/beta-swap-guide.md)

- [One training query, broader exposure and complete-map accuracy](../topics/attention/delta-query-coverage-guide.md)

- [Squared key overlap, frame potential and its nonzero floor](../topics/attention/overlap-regularization-guide.md)

- [Orthogonal keys, zero padding and one state answering every query](../topics/attention/key-geometry-retention-guide.md)

- [How a later key's write can change a previously correct answer](../topics/attention/query-write-order-guide.md)

- [Why whole-history repetition cancels in softmax but changes delta memory](../topics/attention/history-repetition-guide.md)

- [More events versus more associations, and why a write can affect another read](../topics/attention/association-load-guide.md)

- [Why inference state, query cost and training memory differ](../topics/attention/retrieval-gpu-resources-results.md#what-each-measurement-means)

- [Why recency helps latest-value updates but can favor unrelated recent keys](../topics/attention/recency-ablation-results.md#why-both-effects-make-sense)

- [How delta memory and a softmax cache retrieve the latest value](../topics/attention/delta-retrieval-guide.md)

- [How supplied alignment selects learned values and differs from learned transfer](../topics/benchmarks/copy-marker-alignment-results.md#what-changes-and-what-stays-learned)

- [Why training with a source marker differs from inserting one afterward](../topics/benchmarks/copy-marker-training-results.md#what-changed-and-why)

- [Why a distinct source marker changes an identical-value average](../topics/benchmarks/copy-source-end-marker.md#why-the-marker-matters)

- [RoPE copying: relative positions, cross-attention and length transfer](../topics/benchmarks/copy-rope-guide.md)

- [EOS training: content gradients, terminal feedback and oracle length](../topics/benchmarks/copy-eos-training-guide.md)

- [Copying: token budgets, seen lengths and extrapolation](../topics/benchmarks/copy-length-exposure-guide.md)

- [How selecting a source value differs from inserting an answer](../topics/benchmarks/copy-aligned-routing-results.md#what-changes-and-why)

- [One-time momentum reset, rebuilding history and matched training controls](../topics/benchmarks/state-momentum-training-results.md#what-changed-and-why-this-comparison-matters)

- [How Adam's two moment estimates affect an update](../topics/benchmarks/state-adam-momentum-results.md#how-adam-uses-history)

- [Why lower CE and reset invariance can disagree](../topics/benchmarks/state-training-gradients-results.md#what-a-gradient-tells-us)

- [How K changes routing and V carries content](../topics/benchmarks/state-block1-transport-results.md#what-k-and-v-change)

- [Residual transport versus attention: a matched activation diagram](../topics/benchmarks/state-initial-paths-results.md#what-gets-replaced)

- [Reading four-input reset response grids](../topics/benchmarks/state-reset-response-results.md#why-four-inputs-help)

- [Why a reset should remove initial-bit dependence](../topics/benchmarks/state-reset-exposure-results.md#why-the-initial-bit-should-not-matter)

- [Recipe development versus confirmation](../topics/benchmarks/state-recipe-development-results.md#why-this-phase-is-different)

- [Does balancing change learned routing?](../topics/ffn/moe-balance-training-guide.md)

- [MoE load balancing](../topics/ffn/moe-balancing-guide.md)

- [MoE routing: choosing an FFN](../topics/ffn/moe-routing-guide.md)

- [State erasure versus reset-aware answers](../topics/attention/memory-reset-diagnostic-results.md)

- [How memory control is learned](../topics/attention/learned-memory-control-guide.md)

- [Gated delta memory: forget, then correct](../topics/attention/gated-delta-memory-guide.md)

- [Delta memory: key-directed correction](../topics/attention/delta-memory-guide.md)

- [Gated memory: retention and overwrite](../topics/attention/gated-memory-guide.md)

- [What changes when linear attention learns retrieval?](../topics/attention/linear-retrieval-guide.md)

- [Linear attention and accumulated state](../topics/attention/linear-attention-guide.md)

- [Learning a causal global relay](../topics/attention/global-retrieval-guide.md)

- [Global tokens and causal leakage](../topics/attention/global-attention-guide.md)

- [Learning to relay through local attention](../topics/attention/window-retrieval-guide.md)

- [Sliding-window attention](../topics/attention/sliding-window-guide.md)

This index lists learning documents across the [subject folders](../topics/README.md).

- [Learning sequence copying: teacher forcing and free-running outputs](../topics/benchmarks/encoder-decoder-copy-guide.md)
- [Encoder-decoder: shifting, padding and teacher forcing](../topics/foundations/encoder-decoder-guide.md)
- [Cross-attention: reading a separate sequence](../topics/attention/cross-attention-guide.md)
- [Learning adequacy: fit lookup before interpreting composition](../topics/benchmarks/composition-learning-adequacy-guide.md)
- [Combining two relationships: graph lookup](../topics/benchmarks/recurrent-composition-guide.md)
- [GRU versus Transformer: predicting the next word](../topics/benchmarks/recurrent-language-guide.md)
- [GRU versus Transformer: retrieving associations](../topics/benchmarks/recurrent-retrieval-guide.md)
- [GRU versus Transformer: recurrent state and gates](../topics/benchmarks/recurrent-state-guide.md)
- [Intermediate-state supervision: targets and gradients](../topics/benchmarks/state-supervision-guide.md)
- [State tracking: ordered updates and reset shortcuts](../topics/benchmarks/state-tracking-guide.md)
- [Architecture and mathematical tree](../topics/foundations/architecture.md)
- [FFN activation: ReLU versus GELU](../ffn-activation-guide.md)
- [Next subject: the feed-forward network](../topics/ffn/ffn-investigation.md)
- [FFN width: what32 versus64 changes](../ffn-width-guide.md)
- [What does “black box” mean here?](../topics/foundations/interpretability.md)
- [Learning with shared K/V](../topics/attention/mqa-training-guide.md)
- [One complete transformer forward pass](../topics/foundations/numerical-walkthrough.md)
- [Positional representations: one next comparison](../positional-comparison-guide.md)
- [Query exposure: what changes during learning?](../topics/training/query-exposure-guide.md)
- [SwiGLU: what the gate actually does](../topics/ffn/swiglu-guide.md)
- [LayerNorm and RMSNorm: a short learning guide](../topics/normalization/wikitext2-norm-guide.md)
- [Weight tying: one word matrix, two roles](../topics/embeddings/weight-tying-guide.md)
- [What the eight attention tests mean](../topics/foundations/test-clarifications.md)
- [From input pairs to a trained answer](../topics/retrieval/trained-recall-walkthrough.md)
- [From toy tasks to WikiText-2](../topics/benchmarks/wikitext2-comparison-guide.md)
