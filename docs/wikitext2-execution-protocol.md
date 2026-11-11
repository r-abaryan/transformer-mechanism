# WikiText-2 execution details, frozen before training

[Decision0003](decisions/0003-wikitext2-matched-baseline.md) fixes the comparison and adoption rule. This records parsing and numerical details after inspecting publisher bytes, before model evaluation on test. It does not change that decision.

## Encoding and windows

PyArrow25.0.1 is isolated in `.parquet-reader`; it decodes the publisher's pinned single string column. Nonempty rows have exactly one trailing newline, empty rows have none. Encode each row as its whitespace-separated words followed by one `<eos>`, including blank rows. This preserves row boundaries and yields the independently published token totals2,088,628 /217,646 /245,569. Record actual counts, never force them by dropping/adding arbitrary content.

Vocabulary: `<unk>`, `<eos>`, then sorted distinct training words excluding those two. Its observed size is33,278. Preserve case/punctuation; no tokenizer optimization. Unknown validation/test words map to `<unk>`; report mappings separately from literal `<unk>` occurrences. Store little-endian uint32 streams and hashes. Independently reconstruct using regex whitespace matching, not the producer's tokenizer.

For stream t of M tokens, evaluation starts at0,64,..., while start<M-1. Input=t[start:start+count], target=t[start+1:start+count+1], count=min(64,M-1-start). Adjacent blocks share one boundary token as previous target/next input; every target index1..M-1 is scored once. No padding or discarded tail. Batch8 complete blocks; the final partial block is evaluated separately. Context and RoPE reset at each block. Training draws batch8 starts uniformly from0..M-65 inclusive with a CPU generator seeded seed+1000. Both paired arms replay the same starts.

## Numerical checks and evidence

Training uses explicit causal dot-product/softmax attention, expanding shared K/V logically to the two query heads. This is a device-capable counterpart of the earlier CPU implementation, which hardcodes CPU masks/positions. It retains the same parameters and math. No cache is used during training; claimed cache payload is a shape calculation.

Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing torch; deterministic algorithms, one CPU thread, TF32 off. FP32 weights/compute on RTX4060. The preflight compares actual vocabulary and all three paired seeds with the independent CPU float64 forward using FP32-rounded weights: maximum output error<=1e-4, relative L2<=1e-5; gradient absolute error<=1e-5 and relative L2<=1e-4; causal prefix error<=1e-6. CPU-double device-capable outputs/gradients must match the independent implementation within1e-10. Capture executed versions/device.

Check finite losses, every parameter gradient and updated parameters at every optimizer step. Save resumable model/Adam/generator state each1,000 updates, training index hashes, target/token counters, validation curve and elapsed loop time. Use final checkpoints only. All attempts finish before test scoring. No speed comparison is selected.

Save every test token's FP32 negative log-likelihood, argmax and correctness; total vocabulary logits are too large to retain. Keep a fixed fixture at the first two complete test blocks plus the final partial block. The independent CPU-double audit recomputes **all** test token losses with FP32-rounded final weights, tolerance1e-4 per token and1e-5 mean loss. Fixture logits must agree within1e-4. Recompute unigram loss from training counts. Independently verify initialization, optimizer counters, matching streams, source/data hashes and the adoption rule. Count argmax differences descriptively; near ties can differ under FP32 without violating loss tolerance.

The tolerances are numerical audit limits, not quality adoption thresholds. Corpus structure/OOV inspection is allowed before training; no model or unigram test quality is evaluated before all six attempts finish. Every failed training attempt is retained and fails the decision. A failed numerical preflight or audit is an implementation issue to resolve and archive, not evidence against MQA.

## Sources and limits

[Merity et al.2016 Table1](https://arxiv.org/html/1609.07843v1) supplies independent reference counts. [Apache Arrow25.0.1 reader documentation](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html) defines decoding; [PyTorch2.11 reproducibility notes](https://docs.pytorch.org/docs/2.11/notes/randomness.html) cover deterministic CUDA requirements. Fixed64-position context and tiny capacity prevent direct paper/leaderboard comparisons.
