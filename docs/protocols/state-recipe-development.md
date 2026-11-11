# State-tracking recipe development

Frozen 2026-10-03 after complete optimizer-step profiling and before training. Question: can this architecture fit sustained state updates under one bounded optimization search? Eight fresh attempts only.

Reuse the original six training cells: three groups at 8/12 operations, 128/512 examples per group, 1,920 examples total. Preserve the original final-answer supervision, labels, complementary examples and uniform sampling from concatenated groups. Store a training-only snapshot. Historical validation/test pools are already opened; they will not be evaluated or used for selection here. Any confirmation must use a separate frozen stage.

| Recipe | Adam peak rate | Schedule |
|---|---:|---|
| constant_001 | .001 | Constant |
| constant_003 | .003 | Constant |
| cosine_001 | .001 | 200-update linear warmup, then cosine |
| cosine_003 | .003 | 200-update linear warmup, then cosine |

For scheduled recipes, rate at step s<=200 is peak*s/200. Afterwards it is peak*[.1+.9*(1+cos(pi*(s-200)/5800))/2], reaching .1*peak at update 6,000. No restarts, weight decay or gradient clipping. Constant arms also run 6,000 updates.

Development seeds 2903/2909. Identical initial full tensors and index streams within each seed; rotate recipe execution order. Same 4,080-parameter MQA decoder: width 16, two blocks, two Q/one KV head, RoPE, ReLU FFN32 and existing normalization. Architecture and supervision are unchanged. Batch 64, alternate lengths 8/12, Adam defaults otherwise. Each run processes 384,000 examples/final targets, 3,840,000 operations and 4,608,000 input tokens. This doubles the old per-run update/exposure budget, so historical scores are not paired controls.

CPU float64, one thread, deterministic operations. Saved profile uses a full length-12 forward/backward/Adam step; all optimizer states allocated. GPU allocated/reserved bytes are zero; CPU peak unmeasured. Profile seed 2897 is discarded. Report measured execution time without a speed claim.

Evaluate all six training cells every 1,000 updates and at the final checkpoint. These curves are descriptive; select only final weights. Eligibility requires >=99% accuracy in EVERY training group/length for BOTH development seeds. Among eligible recipes choose lowest maximum CE over all twelve seed/cells; fixed table order breaks exact ties. If none qualifies, select none. No replacement seeds, additional recipes, best-step selection or automatic budget extension.

Verify labels with last-SET/parity equations; constant baseline is 50% per cell. Deliberately ignoring FLIPs must cause label errors. At initialization check native/independent full-model outputs and parameter gradients <=1e-10 and detect shifted-label gradients >1e-6. Check schedule boundaries and missing-warmup control. Final reference logits, independent scalar CE/accuracy, exact argmax, paired initialization/stream hashes and optimizer step counts must pass. Replay data streams; verify targeted original sources/reports/checkpoints are unchanged. Validate prerequisites before creating training artifacts.

Record nonfinite attempts as failed without restarting; attempted work and saved checkpoints remain visible. Stop after eight attempts, audit and one concise educational results report. Successful fitting establishes recipe adequacy on these examples, not algorithm learning, convergence or transfer. This authorizes development only, not an architecture change or automatic confirmation run.
