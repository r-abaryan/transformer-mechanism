# Graph lookup: GPU learning adequacy

## Question and stopping rule

Can a larger, fixed GPU recipe reliably fit one-step lookup on the existing graph task? Two-step composition is secondary. This is a new feasibility study, not an extension of the [closed tiny-model comparison](../topics/benchmarks/recurrent-composition-results.md).

Stop after three fresh seed pairs and the declared example budget. No replacement seeds, score-based selection, tuning or automatic extension. A failed fit gate is a result.

## Frozen task and data

Reuse `results/recurrent_composition_data.json` exactly. These pools were previously evaluated; this is diagnostic evidence, not untouched confirmation. Retain the independent dictionary-walk label audit and baselines.

Each context contains a shuffled directed cycle with4 or6 edges during training. All node queries,12 cyclic symbol renamings, one order and its reverse, and both hop labels are included. Input ends with `[separator, query, hop-marker]`; supervise only the final answer with15-class cross-entropy. Sample equal numbers of one-step and two-step queries. Graph-family split policy and the8-edge test stay unchanged. The latter changes distractor count and positions, not reasoning depth. Families do not represent different graph topologies.

| Pool | Families | Expanded examples |
|---|---:|---:|
| Train4 / train6 |128 /128 |24,576 /36,864 |
| Validation4 / validation6 |32 /32 |6,144 /9,216 |
| Test4 / test6 / test8 |32 /32 /32 |6,144 /9,216 /12,288 |

## Models and optimization

| Setting | Transformer | GRU |
|---|---|---|
| Embedding |15 symbols ×64 |Same initial values |
| Core |2 pre-norm blocks; width64 |1 layer; hidden110 |
| Attention |2 query heads,1 KV head; head width32 |None |
| FFN / normalization |ReLU hidden128; biased LayerNorm, epsilon1e-5 |GRU reset/update gates; no LayerNorm |
| Position |RoPE, offset4, base10,000 |Sequential processing |
| Output |Bias-free64-to15 |Bias-free110-to15 |
| Expected parameters, verified before training |60,672 |60,690 |

Seeds:2267,2273,2281. Adam: learning rate0.001, betas0.9/0.999, epsilon1e-8, no weight decay or clipping. Standard PyTorch Linear/Embedding/GRU initialization; share embedding values within each pair. Other initial values cannot be matched across different structures. Alternate arm order across pairs. Use identical sampled indices within each pair, with CPU generator seed `seed+1000`; alternate4/6 edges by update.

Native Windows CUDA FP32, deterministic algorithms, TF32 off, no AMP. Zero GRU initial state for each sequence. There is no dropout. The generalized Transformer implementation is separate because the archived implementation hardcodes width16/head8. Preserve it and verify the new arithmetic independently.

## Hardware-only batch selection

Before evaluating task models, profile throwaway models initialized with seed2026100202. Use synthetic random15-symbol rows and labels, never task scores. Profile batches256,512,1024 for both arms at11/15/19 tokens. Each case includes three complete forward/backward/Adam updates, including optimizer states. Record peak tensor allocation and allocator reservation after clearing unused cached blocks between cases.

Choose the largest candidate whose every case completes and whose peak allocated **and** reserved memory are below6GiB. If no candidate qualifies, stop before task training. An OOM is recorded; only these three candidates are authorized. Write the selected configuration and hashes before preflight/training. Memory totals and [allocator counters](https://docs.pytorch.org/docs/2.11/notes/cuda.html#memory-management) have different meanings; filling all8GB is not a goal.

Fixed exposure:3,072,000 supervised examples per arm and seed. Updates = exposure / selected batch; candidates divide this exactly. No accumulation. Half the examples have each hop count; half have each context size. At batch1024 this is3,000 updates,8× the old384,000-example exposure. Capacity, initialization, precision, batch and update/exposure schedule change together. Neither a pure capacity effect nor a pure GPU effect can be inferred.

## Measurements and numerical controls

Retain every attempt and final checkpoint/optimizer state. Full training and validation pools are evaluated at one-sixth, one-third, two-thirds and all of the update budget; log minibatch losses at these points too. Record training/evaluation elapsed time and peak allocated/reserved memory. Timing is descriptive, not a controlled throughput benchmark.

Finish all six attempts and save a pretest snapshot before any model test scoring. Then evaluate all seven full pools. Save15-class logits and per-seed accuracy/loss for one-step, two-step, and both two-step edge orders. Report all groups even if the primary gate fails.

Independent preflight: CPU float64 native versus separately expressed attention/GRU equations, outputs and all parameter gradients within1e-10. CUDA FP32 versus that reference: output absolute error ≤1e-4 and norm-relative error ≤1e-5; gradient absolute ≤1e-5 and norm-relative ≤1e-4. Check causal prefixes, changed-future invariance, batch isolation and zero-state repeatability within1e-6. Intentionally remove the attention causal mask and reset GRU state every token; both must be detected (>1e-6). Intentionally one-step labels must fail every two-step row.

Final audit: independently recompute every retained logit in CPU float64. Maximum absolute logit error ≤1e-3; per-example loss error ≤1e-3. Report every cross-precision argmax difference. Recompute metrics and the decision independently. For each accuracy threshold, require its pass/fail classification to hold even if every differing argmax were unfavorable. A classification crossing this bound remains unresolved. Validate stream hashes, optimizer update counts, shared initial embeddings, checkpoint hashes, frozen sources and preserved old reports/checkpoints.

## Decision

1. **One-step fit gate, per architecture:** every seed achieves ≥99% one-step accuracy on both full training pools and both validation pools.
2. **One-step transfer:** only for an architecture passing that gate, every seed reaches ≥99% on test4/6 and ≥90% on test8.
3. **Two-step fitting:** only interpret as a composition-specific comparison if both architectures pass the one-step gate. Require ≥99% two-step train/validation accuracy, including both edge-order groups, for every seed. Otherwise retain descriptive measurements and the fitting failure.
4. **Two-step transfer:** only for an architecture passing its two-step fitting rule, require ≥99% test4/6 and ≥90% test8 in both edge-order groups.

No overall architecture winner or adoption rule is selected. Full training fit plus poor held-out performance establishes a gap; low fitting does not isolate the cause. Larger batches and synthetic contexts do not establish real-world capability.
