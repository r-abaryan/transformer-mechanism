# Compact reader GPU resources: separate frozen diagnostic

2026-10-04. Question: what storage and execution costs accompany the six saved delta/softmax readers? User authorized this resource-only diagnostic after the recency intervention. The parent study's failed reliability and skipped conditional benchmark remain unchanged. These costs cannot establish accuracy-preserving adoption.

## Reuse and scope

Use the exact `Reader.prefill/read/forward` from [delta_retrieval.py](../../experiments/delta_retrieval.py), unchanged. Confirmation seeds 3299/3301/3307, both arms. Load saved float64-trained weights, round to float32, native Windows CUDA, TF32 disabled, deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, one CPU thread. No compilation, fused delta kernel, architecture change or quality retraining.

The reader embeds structured key/value records. Prefill builds K/V or S; query embeds one supplied query and applies the classifier. Query timing excludes a new write. This is not full language-model decoding. Synthetic cost fixtures use eight keys/six values, generator `2026100421 + 1000*batch + length`; query is the first event's key, labels follow its last assignment. Length 128 is outside the quality-tested range. No accuracy/generalization claim from these fixtures.

## Preflight before freezing

[Preflight](../../results/retrieval_gpu_resources/preflight.json) executes actual CUDA helpers, initialized outputs/all-parameter gradients, wrong equations, corrupted outputs, state alias/count controls, checkpoint preservation, timings and a first Adam step. Both trained arms are profiled at batch 64, length 128 before freezing sizes. All six readers are checked on all original eight pools against float64 independent equations, both original and float32-rounded weights. Require elementwise error <= `1e-4 + 1e-4*abs(reference)` and zero argmax changes. Preserve any failures; do not relax tolerances or replace seeds.

## Fixed measurement budget

| Measurement | Fixed cases | Method |
|---|---|---|
| Inference state and timings | Six readers x batches 1/32 x lengths 8/32/128 x three fresh sequential processes = 108 cells | Three warmups, ten synchronized repetitions per prefill/query; publish wall and CUDA-event spans |
| Complete-step memory | Six readers x batch 64 x lengths 8/32/128 x three processes = 54 profiles | Disposable saved-weight copy; forward, final-query CE, backward and first Adam .003 step (`foreach=False`) |

Worker 1 reverses reader/shape order; workers 0/2 use forward order. Record process IDs, software, allocator, GPU/driver, before/after temperature/power readings. Three processes share one machine/session; repeated timings are not independent training or hardware experiments. No replacement trials, parameter grid or extension. Initialize CUDA event handles before timing. Exclude input transfer, model load and reference checks; retain state during query measurements. Event spans may include Python launch gaps.

Persistent float32 payload: softmax `4*B*n*(16+8)` bytes; delta `4*B*16*8` bytes. Independently derive shapes/storage counts. Measure retained-state allocated increment after warmed prefill paths relative to loaded model/input baseline. Record prefill/query peak allocated bytes separately. Model weights, input records, sequence counters and temporary query buffers are excluded from persistent payload. Allocator rounding can differ from payload; reserved bytes are not live tensor bytes.

Training profiles include gradients and first optimizer-state creation; batch 64, one update, 64 final-query targets and `64*n` event records. Record absolute baseline/peak allocated and peak reserved memory. Discard updated copies, never save changes. Fixed inference state does not imply constant backpropagation memory. These profiles do not establish convergence or training throughput.

## Audit and stopping

Freeze source/test/audit/protocol hashes in `lock.json` before worker execution. Targeted manifest preserves parent and recency artifacts and their inherited evidence; do not hash every historical result. Save numerical score tensors separately from JSON summaries. Independently recompute CPU matrix-transition/exponential references, payloads, sample medians, Cartesian coverage and optimizer tensor-state counts. Verify all original pool logits and original failed gate again. Exercise the actual audit entry helper during preflight, including a corrupted payload that must fail.

Publish every seed and process, summarize by shape using medians of nine seed/process medians and their range. Descriptive costs only, no universal speed criterion or architecture winner. Stop after these three processes and audit. Failures remain visible; no automatic optimization or quality training.

Sources: [installed PyTorch 2.11 CUDA semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html) for synchronization, allocated/reserved memory and TF32; [DeltaNet 2024](https://arxiv.org/html/2406.06484v2) for correction and optimized parallelization, which this eager reader does not reproduce. Original quality evidence: [parent report](../../results/delta_retrieval/report.json).
