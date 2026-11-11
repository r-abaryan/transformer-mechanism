# Recency removal: single saved-checkpoint diagnostic

2026-10-04. Distinct question: does removing this reader's recency term recover unique-key retrieval while changing repeated-key updates? The parent study remains closed and unchanged. This is a posthoc diagnostic on already inspected data, not fresh confirmation or parameter selection.

Three saved softmax checkpoints, seeds 3299/3301/3307. Reuse all eight confirmation pools: training 4/8, validation 4/8, tests 4/8/16/32. Native versus one intervention, applied to every sequence and both query groups. No group-specific switching, new data, optimization, sweep or GPU benchmark.

Native logits use `scale*q·k_t + recency*(t-(n-1))`. The intervention sets only the positive recency coefficient to zero by using a separate inference copy with `log_recency=-infinity`. Similarity scale, learned embeddings, classifier, query and event sequence remain fixed. Original parameter tensors/checkpoints are unchanged. No backward pass through the inference copy is intended.

Reuse the parent forward path. Independently check zero-recency results with the parent's explicit exponential/matrix reference, which omits that term. Reuse the separate Python-math loss reducer and forward-dictionary labels. Scalar metrics include count/correct/accuracy/CE, per seed/pool/group. Record rescued/harmed/both-correct/both-wrong predictions. Native metrics must reproduce the saved report.

Before evaluating saved models: test three initialized seeds, inverse control that leaves recency enabled, unchanged original tensors, saved-initialization reload through the actual evaluator, absent groups, wrong labels/scalars/criteria, and a swapped-history ambiguity witness. Without position information, swapping two assignments to the same key preserves the key/value multiset but changes the latest-value label. This is a structural counterexample for this content-only reader, not every Transformer.

Expected evaluation cells derive from 3 seeds × 8 pools × 2 modes = 48. Each checkpoint has 3,456 sequences; two modes and three checkpoints give 20,736 prediction evaluations. Training updates: zero. CPU float64, one thread. Output audit tolerance <1e-9 with zero argmax differences; scalar discrepancy <1e-10. Preserve all parent artifacts and inherited protected files with targeted hashes.

Declare outcomes before evaluation: long unique recovery means >=95% in every seed at test16/32; short unique preservation means >=95% in each other pool/seed. Repeat cost is a decline exceeding two percentage points in any pool/seed versus native. Publish full tables and paired counts even if these criteria fail. These describe intervention effects, not repair adoption.

Stop after the fixed comparison and audit. Removing a single term can establish its effect under the fixed checkpoints; it does not explain every training cause, make a trained repair, reopen resource benchmarking or justify a general architecture ranking. No automatic follow-up. [Parent design](delta-retrieval.md) and [reader equations](../topics/attention/delta-retrieval-guide.md) supply the mathematical conventions.
