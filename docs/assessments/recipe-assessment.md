# Recall recipe: supported choices and remaining decisions

Latest vocabulary constraint check: [weight tying on WikiText-2](../topics/embeddings/weight-tying-results.md) saves 49.82% parameters for 6.41% worse validation PPL, exceeding the permitted 5% loss. Six runs and the all-token audit pass. The explicit shared-start controls differ from earlier initialization; their failed tradeoff does not change the original recipe. Study closed.

Latest external normalization check: [gain-only LN/RMS on WikiText-2](../topics/normalization/wikitext2-norm-results.md). Every RMS validation pair improves, with1.60% aggregate gain, below the2% adoption margin. Reused test improves1.34%; six runs and2,779,278 independent loss checks pass. Both controls remove80 norm biases; their comparison does not adopt that removal or overwrite the original recipe. Study closed without tuning.

This records choices under their original decision rules. Their evidence comes from different cohorts; percentages cannot be combined into a single benchmark score.

| Part | Current research choice | Evidence and limit |
|---|---|---|
| Attention | Causal softmax; two query heads | The implementation passes output/gradient checks. Mathematical rank limits do not establish that this task exhausts head capacity. |
| K/V heads | One shared head, MQA | [Matched training](../mqa-training-results.md) accepts half-size cache with tolerated accuracy losses. Sharing has not demonstrated repeatable GPU speedup in the [small eager decoder](../topics/attention/mqa-decoding-results.md). |
| Position | RoPE, fixed offset4 | [Unique-key comparison](../positional-comparison-results.md) passes; [repeated-key comparison](../topics/training/repeated-key-results.md) gains substantially but misses its all-seed reliability rule. |
| FFN | ReLU, hidden width32 | [GELU](../topics/ffn/gelu-comparison-results.md), [width64](../topics/ffn/ffn-width-results.md) and the separate [external SwiGLU comparison](../topics/ffn/swiglu-results.md) fail adoption. SwiGLU's bias-free22/33 controls do not replace this recipe's biased width32 FFN. Retention does not establish global optimality. |
| Residual placement | Two pre-norm blocks | The current recipe uses this arrangement. [Placement/budget comparisons](../topics/normalization/norm-extension-results.md) show that earlier post-norm failures can recover with more updates; permanent inferiority is unsupported. |
| Query exposure | Train queries for every present key | [Coverage comparison](../topics/training/query-exposure-results.md) improves transfer, with one seed still below its threshold. Task coverage is a training choice. |
| Normalization | Retain the LayerNorm baseline; RMS adoption fails | [Matched gain-only comparison](../topics/normalization/rmsnorm-training-results.md) gains2.34pp mean six-pair unique accuracy but loses in two seeds and misses one90% floor. Both arms have4,464 parameters. |
| Numerical execution | Float64 research reference | [FP32 cache storage](../topics/attention/cache-precision-results.md) preserves the tested final decisions; FP16/INT8 change some. Whole-model low-precision accuracy remains unmeasured. |

The MQA study jointly tests RoPE, ReLU, FFN32, pre-norm and broad queries. It supports that complete recipe within its quality-loss tolerance. It does not prove that each part is individually necessary or the joint recipe is optimal. That existing recipe learns LayerNorm biases and has4,544 parameters. The centering comparison uses fresh four/six-pair data and gain-only norms; use its paired baseline when interpreting it. Bias removal itself was not independently compared or adopted.

The first **task-transfer check is closed**: [binary majority](../topics/benchmarks/majority-transfer-results.md) trains the MQA structure from scratch with task-specific embeddings/readout, retaining learned LayerNorm biases. It has4,016 parameters. All five seeds score100% at9/11/15 bits;21-bit mean97.18% hides seed1627 at86.25%, failing the predeclared reliability rule. This supports aggregation beyond recall within tested conditions, not zero-shot weight transfer, exact counting or universal architecture superiority.

The [external WikiText-2 comparison](../topics/benchmarks/wikitext2-results.md) is also closed under [Decision0003](../decisions/0003-wikitext2-matched-baseline.md). Six runs and independent checks of 1,473,408 test-token losses pass. MQA meets the declared tradeoff tolerance, but geometric mean perplexity worsens 357.92 to 364.95 (1.96%) and every paired seed loses quality. Potential compact K/V payload halves; total parameters fall only 0.051%. No further sweep is selected. Convergence, speed and broad architectural or novelty claims remain unsupported.

Do not start another local attention, epsilon, bias or seed search without a new decision requiring that evidence. One additional toy task does not justify a general architectural claim or novel proposal.

The [matched-cost gated FFN comparison](../topics/ffn/swiglu-results.md) is closed. SwiGLU worsens mean validation PPL2.00% with all three pairs worse; reused test worsens1.83%. All six runs and 2,779,278 independent token-loss checks pass. Equal projection cost does not ensure equal quality or runtime. The original recipe remains unchanged; no additional sweep is selected.
