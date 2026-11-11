# KV-cache equivalence: execution protocol

Frozen before execution. The [decision and mathematical guide](decisions/0002-kv-cache-equivalence.md) defines the question and boundaries.

Use all ten final checkpoints recorded in `query_exposure.json`. Load their original weights in evaluation mode, CPU float64, one thread, deterministic operations and no gradients. Verify every checkpoint, source and input hash. No training or data generation.

For each model, use original test rows 0, 1, 36 and 37 at 3/4/6 pairs, batched together. Test position offsets 4 and 17. Compare all-token logits from the existing full RoPE forward against two schedules: all single tokens; or floor(sequence_length/2) tokens followed by single tokens. This gives 120 cases and 480 sequences, with repeated reuse across models/conditions.

Each layer stores its own rotated K and unchanged V. Chunk attention uses the global token-index causal mask: a new query at index i can see key indices <=i. Preserve all learned biases, normalization, output projection, residuals, ReLU and final readout. Record cache shape and actual tensor bytes after every chunk. Expected K/V payload: `B L h T (d_k+d_v) element_size`.

Run two negative controls under both schedules. Reset-position control uses `offset + local_chunk_index` for new Q/K rotations, retaining previously cached entries. Omitted-entry control leaves the cache intact but excludes its oldest past entry from attention in each layer whenever a past prefix exists. Each control is an intentional error, not an eviction policy or architecture candidate.

Save full, cached and both control logits plus final correct K/V tensors in a tensor artifact. Store every case's maximum absolute logit discrepancy, argmax differences, minimum reference top-two gap, count of gaps <=1e-10, per-chunk lengths/payload and final task predictions. Reference task correctness is diagnostic only; intermediate-token logits were not trained as task answers.

Pass requires all 120 correct-cache errors <=1e-10 and every length/payload check correct. Each negative control must exceed 1e-6 discrepancy in at least one case. Disclose any argmax differences even when logits pass tolerance. Never replace a checkpoint or tune a threshold.

One separate auditor reloads artifacts and checkpoints, checks hashes and fixture membership, and recomputes full/chunked outputs using explicit rotation matrices, separate Q/K/V projections, explicit layer-normalization arithmetic and PyTorch SDPA. It independently checks saved metrics, final K/V and control activation. This second implementation exists only to corroborate the numerical contract.

Write one results document and audit. Close after these checks. Preserve a failure and diagnose it if the contract fails. Do not add seeds, training, latency benchmarking, padding, cache eviction or quantization to this study.
