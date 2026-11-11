# Softmax versus positive-kernel retrieval

Frozen2026-10-03 after complete CPU optimizer-step profiling, before training or final evaluation. Six attempts only; no replacement seeds, tuning or automatic extension.

| Item | Fixed choice |
|---|---|
| Question | How does replacing softmax similarity with normalized ELU+1 feature similarity affect learned latest-value retrieval? |
| Data | Reuse archived MQA family splits, expanded to all query keys. Previously examined data, not fresh confirmation. Exactly one repeated key, distinct values, twelve keys, six values, separator18. Label is latest value for query. |
| Lengths | Train/validation3 and4 pairs; test3/4/6 pairs. Sequence length2n+2. Full pools evaluated, repeated and unique queries separately. |
| Model | Both4,544 parameters, width16, two pre-norm blocks, two query heads sharing one KV head, head width8, ReLU FFN32, LayerNorm, RoPE offset4,19-class output. |
| Change | Softmax(qk^T/sqrt(8)) versus normalized phi(q)^Tphi(k), phi=ELU+1. Both causal inclusive. Apply RoPE before phi in linear arm. Positivity avoids negative kernel weights; this nonlinear feature map does not preserve RoPE's usual dot-product relative-position identity. Compare this exact recipe, not all linear attention. |
| Execution | Dense pairwise implementation for both during training. Linear independent reference uses recurrent S/z. No sparse/linear-runtime claim. |
| Pairing | Fresh seeds2591/2609/2621; complete identical initial state and sampled minibatch stream per pair. Rotate execution order. |
| Budget | CPUfloat64, one thread, deterministic;6,000 Adam updates, lr0.001, batch64, alternate3/4 pairs.384,000 examples/final-query targets and3,456,000 input tokens per arm. Same examples/updates, not equal compute. |
| Profile | Separate discarded seed2587, full forward/backward/Adam step including state allocation. Both candidates execute. GPU unused; allocated/reserved GPU bytes0. CPU peak memory unmeasured. Cold single-step timings are not a benchmark. |
| Measurements | Minibatch loss every1,000 steps; final full training/validation accuracy and loss. Test only after all attempts and saved pretest snapshot. Save checkpoint/optimizer/predictions and hashes. |
| Checks | Independent labels/families, oracle/first-match/last-value/constant/uniform baselines; all-parameter gradients and outputs against manual model plus recurrent kernel at initialization, causal prefixes and batch isolation; missing-denominator and shifted-label controls. Full-pool final independent logits/metrics, tolerance1e-10, zero argmax changes. |

Each arm passes short fit only if every seed's repeated/unique groups reach99% on training and validation at both lengths. Reliability additionally requires99% on test3/4 and90% on test6 for every group/seed. Report each six-pair balanced repeated/unique mean. A consistent advantage needs at least10 percentage points in the same direction in all three pairs. A transfer-specific comparison is eligible only if both short-fit gates pass. Failure to fit prevents attributing the long-input gap solely to transfer. Accuracy cannot isolate state collisions as the learned failure mechanism.

Stop after six attempts and independent audit. Nonfinite training or a rejected denominator is a failed attempt. Preserve failures and earlier evidence. Main reference: [Katharopoulos et al. (2020)](https://proceedings.mlr.press/v119/katharopoulos20a.html), sections3.2-3.4; our [math audit](../topics/attention/linear-attention-results.md) verifies the kernel identity, not trained quality.
