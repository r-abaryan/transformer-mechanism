# Architecture coverage and next priority

Latest authorized follow-up: [MoE balancing training](../topics/ffn/moe-balance-training-results.md) is closed. This adds small-classifier empirical evidence: less overflow, adequate fit, failed groupwise CE tolerance. It does not establish language quality or specialization.

Latest follow-up: [MoE load balancing](../topics/ffn/moe-balancing-results.md) is also executed and closed. Loss/gradient checks and counterexamples add arithmetic evidence, not training or specialization evidence. No next experiment selected.

Follow-up status: [MoE routing audit](../topics/ffn/moe-routing-results.md) is now executed and closed. [Educational guide](../topics/ffn/moe-routing-guide.md). No next experiment selected. The review and proposal below record the prior decision.

Reviewed 2026-10-03. Documentation review only; no new experiments. This is a navigation and stopping decision, not another results report.

**Established** means supported within the linked derivation or tested conditions. **Conditional** means the result depends on the task, recipe or hardware. Neither means universal superiority.

| Area and evidence | Established | Conditional | Unknown | Next priority |
|---|---|---|---|---|
| [Core architecture](architecture-assessment.md) | Attention, masking, residual, FFN and encoder-decoder contracts have numerical checks | Correct operations do not guarantee a learned algorithm | Complete explanations of trained behavior | Retain evidence; no repeat audit |
| [Positions](position-representation-assessment.md), [FFNs](ffn-assessment.md), [normalization](normalization-assessment.md) | Components change functions and gradients; matched comparisons are recorded | Preferred recipes vary across tasks and acceptance margins | Rankings at larger scale or after convergence | Keep bounded conclusions |
| [Caching and execution](../topics/attention/mqa-decoding-results.md) | Compact MQA caches reduce stored K/V | None of six shapes passed the repeatable speedup criterion | Broader end-to-end speed/quality tradeoffs | Defer until a deployment decision needs them |
| [Sparse attention](sparse-attention-assessment.md) | Masks constrain paths; local/global relay can be learned | Global relay failed longer-input reliability in one seed | Cause of that transfer gap; sparse-kernel savings | Closed |
| [Memory updates](memory-update-assessment.md) | Additive, gated and correction rules have distinct storage behavior | Both trained delta readers reached 100%; added gating gained no accuracy | Broader-task quality and full learned mechanism | Closed |
| [Reset diagnostic](../topics/attention/memory-reset-diagnostic-results.md) | All 3,456 paired histories retain state differences despite correct answers | Disabling forgetting changes no tested classes, but changes loss | How the complete reader suppresses irrelevant history | Closed; no further mechanism probes selected |
| [Task capabilities](capability-summary.md) | State, retrieval, language and other task results are independently checked | Winners differ; some failures include incomplete training fit | An overall winner or reliable broad reasoning | Do not combine different tasks into a ranking |
| Mixture-of-experts FFNs | No local study found in this review | No local quality or speed evidence | Routing arithmetic, gradients and capacity behavior | Recommended next: one mathematical audit |

## Why mixture of experts next?

A dense FFN applies the same network to every token. A mixture of experts (MoE) chooses among several FFNs. This adds a distinct architectural question: **which parameters process each token?**

For a Switch-style top-one router, before capacity filtering:

\[
p=\operatorname{softmax}(Wx),\qquad e=\arg\max_i p_i,\qquad y=p_e E_e(x).
\]

Here, each expert \(E_e\) is an FFN. The selected probability scales its output. The discrete choice and the differentiable probability are different parts of the computation. Expert capacity limits how many tokens an expert accepts. These definitions come from [Switch Transformers, sections 2.1–2.2](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf), a primary reference chosen for its explicit routing design, not its recency.

**Proposed scope, not yet executed:** freeze a small CPU float64 audit. Compare explicit per-token equations against dispatched expert computation. Check input, router and expert gradients away from selection ties. Declare tie handling, capacity rounding, overflow behavior and padding before execution. Include intentional routing and gate-weight errors. Count total versus activated parameters separately.

**Stop after the declared arithmetic checks and one short educational report.** No training, expert-count sweep, GPU speed claim or automatic follow-up. Numerical routing correctness would not prove learned specialization or better language modeling.

Tokenization, large-scale optimization and distributed execution remain outside this review's coverage. They are deferred questions, not a queue of authorized experiments. Existing studies and their stopping decisions remain unchanged.
