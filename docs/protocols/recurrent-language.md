# GRU versus Transformer on WikiText-2

Frozen2026-10-02 before new-model evaluation. One bounded comparison, six attempts.

| Item | Fixed choice |
|---|---|
| Question | Which complete model/training package yields lower next-word loss within this budget? |
| Corpus | Existing audited word-level WikiText-2 streams, publisher revision b08601e04326c79dfdd32d625aee71d232d685c3. Training-only vocabulary33,278; unknown ID0, row-end ID1, including empty rows. Reused/opened dataset, not fresh confirmation or a published benchmark reproduction. |
| Labels | Input token at index i predicts token i+1. No same-token labels, masking of unknown/row-end targets, or padding. |
| Transformer | Existing two-block width16 MQA, two Q heads/one KV head, RoPE offset4, biased LayerNorm/ReLU32 FFN, untied bias-free output.1,068,832 parameters; processing core3,936. |
| GRU | Two layers, hidden16, embedding16, zero initial hidden state per window, dropout0, untied bias-free output.1,068,160 parameters; processing core3,264. Same total within0.063%, but GRU core17.07% smaller. Vocabulary layers dominate total count. |
| Pairing | Seeds2183/2203/2213. Copy Transformer initial embedding and output values into GRU. Other native initializations differ. Identical sampled window starts within each pair; alternate arm order across pairs. |
| Training |6,000 Adam0.001 updates, batch8, context64, all512 next-word targets per update.3,072,000 targets per attempt. Native Windows RTX4060, FP32, no TF32/AMP, deterministic algorithms, one CPU thread. No clipping, dropout or weight tying. |
| Evaluation | Reset context/GRU state at each window. Validation and test: every next-word target exactly once in consecutive non-overlapping64-token windows plus tail. Descriptive test128 uses the same target tokens but fewer context resets. It is a context-window check, not a separate held-out dataset or pure length-transfer isolation. |
| Training fit | Predeclared subset of1,024 complete64-token windows: block indices floor(j*(32634-1)/1023), j=0..1023;65,536 targets. Even coverage of the training stream, not full training-pool loss or guaranteed distribution representativeness. |
| Metrics | Loss, perplexity=exp(loss), top1 accuracy, target count; per seed and early/middle/late positions0..7/8..31/32 onward and target groups unknown/row-end/ordinary. Train subset/validation every1,000 updates and at initialization. Final checkpoint only. |
| Baseline | Training-count add-one unigram probability (count+1)/(training_tokens+vocabulary), on every evaluation pool. |

Decision: all six attempts finish finitely and all models beat the unigram by at least0.05 nats on final training-subset, validation and context64 test. A scoped validation winner additionally requires at least2% improvement in geometric mean perplexity and improvement in every paired seed. Otherwise report mixed/small evidence without choosing a winner. Context64 test and context128 are descriptive; neither selects a model or changes the rule. No convergence or overall superiority inference.

Before training: verify existing data-audit hashes; independently check shifted labels on corpus windows and detect intentionally unshifted labels; manually unroll both GRU layers and use existing independent attention forward; check all parameter gradients in CPU double and GPU FP32. CPU forward/gradient absolute tolerance1e-10; GPU output maximum1e-4/relative L2 1e-5 and gradient maximum1e-5/relative L2 1e-4. Check causality, batch isolation and zero-state behavior; reset-at-every-token GRU error must be detected.

After training: freeze all final train-subset/validation metrics and checkpoint hashes before test evaluation. Independent CPU-double audit of every retained token loss/prediction and initial/tail logit fixtures: token-loss/fixture error<=1e-4, mean-loss error<=1e-5, PPL relative error<=2e-5. Report cross-precision argmax differences without assuming bit identity. Recompute streams, optimizer steps, per-group metrics and decision. Keep failures; no replacement seeds or automatic budget/tuning extension. Sources/protocol and prior artifacts remain unchanged.
