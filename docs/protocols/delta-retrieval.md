# Learned delta versus softmax retrieval: frozen design

2026-10-04. Question: can a compact learned delta reader preserve softmax retrieval quality with less persistent decoding state? This is a new matched structured-event comparison, not a rerun of the closed ELU+1 or reset-reader studies. No full Transformer or published-model reproduction is claimed.

## Task and models

Eight key symbols, six value symbols. Each event supplies a key/value pair to both readers; the final query names a present key. Predict its latest value. Unique queries occur once; repeat queries occur 2 through min(6,n) times, with different first/last values. No reset or absent-key task. Queries and relevant event positions vary. Structured records remove token-binding requirements; symbols/templates remain shared across splits.

Both learn a shared-start 8-by-16 key embedding, 6-by-8 value embedding and 8-to-6 linear classifier. Queries use the same unit-normalized key table. Softmax learns positive similarity scale (initial 16) and positive recency slope (initial .5). Delta learns global write strength sigmoid(2). There are 232/231 parameters respectively. The softmax positional bias is part of its declared design, not an isolated normalization substitution. No ideal orthogonal keys or manual key-addressed slots are supplied.

Softmax final-query weights: `softmax(scale*q·k_t + recency*(t-(n-1)))`. Delta starts S=0 and updates `S <- S + beta*k_t*(v_t-S^T*k_t)^T`; final read is `S^T*q`. Both predict with the learned classifier. Key normalization epsilon is 1e-12. No FFN, residual block, convolution or published parallel kernel.

## Frozen pools and budget

| Phase | Settings | Data |
|---|---|---|
| Development | Seeds 3271/3277; constant .003 or cosine .003 to .0003; both arms | Generator 2026100407; train/validation lengths 4/8 |
| Conditional confirmation | Fresh seeds 3299/3301/3307; each arm's locked eligible recipe | Generator 2026100413; new train/validation 4/8 and tests 4/8/16/32 |

Per length/group: training 64 base contexts, validation 16, test 32. Each base generates all six cyclic value renamings, balancing labels. All renamings and queries of a context are forbidden across pools; canonicalization removes cyclic value renaming, not key renaming or shared templates. Confirmation rejects all development families. Development has 1,920 examples; confirmation 3,456. Test data are generated only after both development arms qualify. Confirmation tests are evaluated only after all confirmation attempts finish; no selection uses them.

Every formal run receives 3,000 Adam updates, batch 64, alternating training lengths 4/8: 192,000 final-query targets and 1,152,000 event records. CPU float64, one thread, deterministic algorithms; no dropout, clipping or other schedule. Cosine uses the declared endpoint-inclusive formula in source. Identical sampled indices and common initial tensors are paired by seed. Equal exposure and near-equal parameter counts do not establish equal compute. Record every attempt, stream hash, initialized/checkpoint hashes and 1,000-step losses. Exceptions remain failed attempts, with no restarts or replacement seeds.

## Gates and stopping

1. Preflight: independent labels/families; baseline and wrong-label controls; 18 initialized output/all-parameter-gradient cases; six directional checks; wrong delta/recency controls; batch isolation; supplied logits/missing groups; saved initialization reload through the actual final audit entry; criterion errors. Profile a complete optimizer step for both readers before formal training. Test fixtures use fresh retained folders.
2. Evaluate full training and validation pools for every development run. A recipe qualifies only if **every training length/group in both development seeds reaches >=99%**. Select each arm's qualifying recipe by lowest worst training-group CE; declaration order breaks ties. If either arm has no eligible recipe, stop: no confirmation or performance comparison.
3. Fresh confirmation requires >=99% in every training cell and >=95% in every validation/test cell for both arms/all seeds. Delta test accuracy may lose at most two percentage points to its paired softmax control in every group/length. CE is reported but not used to change these criteria.
4. Only passing confirmation quality permits performance measurement. The conditional accuracy-memory claim additionally requires >=50% reduction in logical persistent decoding tensor payload at n=32 for batches 1/32 in every pair. Speed is a separate descriptive measurement, not an adoption requirement. Any gate failure closes this budget without tuning.

## Audit and resource method

Reconstruct labels by a forward dictionary. Audit delta with ordered `(I-beta*k*k^T)` transitions; audit softmax with explicit stable exponentials and matrix reads. Check full final scores, scalar metrics, zero argmax changes, streams and initialization. Tolerances: final CPU logits <1e-9, CE discrepancies <1e-10. Reference code uses shared learned parameters/data, not a wholly independent software stack.

Conditional resource tests: native RTX 4060 Laptop GPU, FP32, all six confirmation checkpoints. First check GPU versus CPU predictions on every pool; any argmax change fails the GPU measurement gate. Measure batches 1/32, lengths 8/32/128. Synthetic cost inputs are not additional accuracy/generalization evidence.

Persistent payload counts stored K/V or S tensors, excluding weights, input events, scalar sequence counters and ephemeral query buffers. Predicted bytes: `4*B*n*(16+8)` versus `4*B*16*8`. Also measure incremental CUDA allocated bytes of retained state; allocator rounding can differ from payload. Verify full/read-from-state equivalence. Profile one complete training step including first Adam-state creation at batch 64, n=8; report peak allocated/reserved bytes separately. These throwaway profiles are not confirmation training updates.

Time prefill and query separately using wall clock with CUDA synchronization. Each case has two sessions, each with three warmups/five timings; publish all samples and medians. Sessions are repeated measurements, not independent training experiments. Eager sequential delta prefill is not the paper's parallel kernel. No throughput or constant-training-memory inference from state size alone.

Sources: [Vaswani et al. 2017](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html); [Yang et al. 2024, sections 2.2/3](https://arxiv.org/html/2406.06484v2); [Yang et al. 2025, section 3.1](https://arxiv.org/html/2412.06464v3); [installed-version CUDA semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html). The exact local code/protocol/test hashes bind preflight and training. Preserve all earlier study files.
