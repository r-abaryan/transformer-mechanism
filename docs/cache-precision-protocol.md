# Cache precision: frozen numerical protocol

Question: can smaller stored K/V preserve predictions on a bounded reused recall subset? Freeze this document and both scripts before model evaluation. No training, parameter changes, latency measurement or tuning follows.

Use all ten final MHA/MQA checkpoints from `mqa_training.json`, checking hashes and its independent audit. CPU float64 weights and all arithmetic, one thread, deterministic inference. This isolates storage approximation; it is not FP16 model execution.

For each of 3/4/6 pairs select the first four stored test families: original rows0:144 (36 variants per family), expanded into all key queries by the archived function. Batch sizes288/432/720. These are reused diagnostics, with four families per length; not a fresh benchmark or independent evidence of general accuracy.

Six modes: float64 control, FP32 K/V, FP16 K/V, symmetric INT8 K/V, INT8 K only, INT8 V only. Unquantized sides in single-side controls remain float64. This yields180 cells (10 models x3 lengths x6 modes).

Execute every token individually at positions4+i. Each layer projects new Q/K/V, applies RoPE to Q/K, encodes rotated K and unchanged V immediately, and stores only encoded entries. Read/decode all past and current entries for attention. Earlier entries are never re-encoded. Later layers receive the perturbed residual stream. This tests accumulated effects during real cache reads, not quantization of a completed forward trace.

Floating storage casts to its stated dtype and reads into float64. INT8 uses one scale per token/head/vector across its eight channels: `s=FP32(max(abs(x))/127)`; zero vectors use s=1. Codes are nearest-even rounded `x/s`, clipped[-127,127] and stored as int8. Decode in float64 as `code*s`. Count codes AND FP32 scales in payload. Assert finite positive scales and reconstruction error <=s/2+1e-12 on every INT8 write. Do not confuse this symmetric per-token codec with KIVI's asymmetric per-channel/grouped-key design.

Compare full causal baseline against every mode. Retain all-token max error and argmax counts; final logits, labels, rowwise max error, top-two gap, final changed/correct/damaged/repaired counts, and actual final encoded tensor bytes. A row is prediction-certified when its reference top-two gap exceeds twice its final maximum logit error. Final changes are disclosed even if total task accuracy is unchanged.

Float64 control must agree in all30 cases within1e-10 and preserve all predictions. A smaller-storage mode earns **prediction preservation on this subset** only if all30 cases have zero final argmax differences and all payload checks pass. No tolerance substitution or seed selection. Differences are findings, not numerical-check crashes. No universal deployment/adoption rule is inferred.

Before checkpoint evaluation, test codec zeros, rounding ties, range and measured payload using explicit vectors. Include one attention/readout counterexample where INT8 rounding changes the prediction, demonstrating why no universal preservation guarantee follows. Save the witness and audit it independently.

The separate auditor checks source/protocol/checkpoint/input hashes, family membership and labels, saved metrics and decisions. It recomputes all180 cached cases using explicit normalization, rotation matrices, separate projections and SDPA, with a separate quantization implementation. Scalar Python rounding/struct checks independently corroborate the codecs. Agreement threshold1e-10. The auditor does not establish GPU behavior, new generalization or KIVI replication.

Write one concise educational results document with math, tables, sources and limits. Close after execution and audit, including failures. No bit-width sweep, retraining, additional families or architecture change.
