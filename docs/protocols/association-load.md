# Fixed-length association load: frozen design

2026-10-04. User authorized the follow-up: how does a fixed delta matrix behave when a context contains more distinct associations? Earlier runs used only eight keys. This study changes information load while holding sequence length, key universe, reader widths and parameter counts fixed. It estimates behavior under a bounded recipe, not a universal memory-capacity threshold.

## Task and controls

Each sequence has 32 structured key/value events drawn from 32 known keys and six values. Loads M=2/8/16/32 are the number of distinct keys present. Each present key occurs exactly 32/M times and always carries the same value. The final query names a present key. Event order is shuffled; assignments vary independently between contexts. No conflicting updates, resets or unseen symbols. Target is that key's assigned value.

All six cyclic value renamings of a context stay together. Context family identity excludes query, so a context cannot cross pools by changing its query. Development and confirmation families are disjoint. Symbols and construction templates remain shared. Each load/pool covers all 32 query keys equally. Labels are exactly balanced by renaming. Baselines: constant class, final event's value and first matching event's value; the last is a supplied exact algorithm, not learned evidence.

At fixed length, load also changes repetition count and intervening keys. Equal multiplicities remove relative frequency as a query shortcut but do not isolate pure matrix rank. Do not call any failure an information-theoretic capacity bound.

Reuse [Reader](../../experiments/delta_retrieval.py) forward/read/prefill and independent matrix/exponential equations unchanged. Replace the key table with 32x16. Both readers learn 6x8 values and an 8-to-6 classifier; tied unit-normalized Q/K, epsilon 1e-12. Softmax learns similarity scale, initially 16; recency is fixed at zero through an exp(-infinity) buffer. Delta learns sigmoid(2) write strength. Each arm has 615 trainable parameters. Key/value/readout initial tensors match per seed. No FFN, residual, orthogonal slots, new update rule or full published-model reproduction.

## Development and confirmation

| Phase | Settings | Base contexts per load |
|---|---|---|
| Development | Seeds 3313/3319; both arms; constant Adam .003 or cosine .003 to .0003 | Train 128, validation 32; generator 2026100431 |
| Conditional confirmation | Seeds 3323/3329/3331; locked eligible recipe per arm | Fresh train 128, validation 32, test 64; generator 2026100437 |

Each base context has six renamings: 768 training, 192 validation and 384 test examples per load. Development has 3,840 examples; confirmation 5,376. No longer-sequence evaluation; all sequences stay length 32.

Each run completes exactly 3,000 Adam updates, batch 64, CPU float64, one thread, `foreach=False`. Loads cycle 2/8/16/32, giving 48,000 supervised final queries per load, 192,000 total and 6,144,000 input event records per run. Both arms receive identical seeded minibatch streams. No checkpoint selection, replacement seed, gradient clipping or extension. Evaluate complete training/validation pools at the final checkpoint.

**Recipe selection:** a candidate must reach >=99% training accuracy at load 2 in both development seeds. Among eligible candidates, choose minimum worst training CE across all four loads and seeds; tie favors constant. This is a low-load control, not a claim that higher loads fitted. If either arm has no eligible recipe, skip confirmation and report all eight attempts. Maximum budget: eight development plus six fresh confirmation runs.

Confirmation tests are opened only after all six training runs finish; preserve `pretest.json`. Report every seed and load: training/validation/test accuracy, CE and counts. Per-load fit = >=99% training in all three seeds; test reliability = >=95% in all three. A >=10-point test drop from M=2 to M=32 in every seed is the descriptive load-drop criterion. Check the load-2 control separately. Higher-load low training fit means the recipe did not master that load; it does not establish pure held-out failure or architectural impossibility. No adoption or speed criterion.

## Controlled write diagnostic

For each confirmation test example append one same-value write to the first event key different from the query. The query's correct answer remains unchanged. Freeze this selector before evaluation; do not choose damaging examples. Measure rescued/harmed classes, CE and mean logit-shift norm, including zero harms. This 33-event intervention is separate from the fixed-length task curve and outside the training length.

For delta verify the exact read change:

`delta_read = beta * (q dot k) * (v - S^T k)`.

Record mean absolute query/write overlap and mean read-shift norm. Apply the same added input to softmax. It changes its value multiplicities too, so this intervention does not isolate all differences to delta interference. Confirm before/after logits using independent equations. A class harm proves sensitivity to this write in a saved model, not the cause of the aggregate load curve.

## Preflight, preservation and stopping

[Preflight](../../results/association_load/preflight.json) executes initialized equations/all-parameter gradients at each load, deliberate equation/recency/label/split errors, actual evaluator and independent audit entry with saved initialized weights, absent groups, wrong scalar/selection controls, and write equations. Four disposable complete Adam steps, CPU/GPU float64, profile batch 64/length 32 including optimizer-state allocation before freezing the formal CPU choice. Earlier GPU resource evidence makes launch overhead relevant at this scale; the preflight is a feasibility check, not a speed benchmark.

The first preflight's recency error fixture happened to use identical values, making route changes invisible. Its sources and failure are preserved under `preflight_failure_001`. Corrected control uses explicitly distinct values and a distant query. No formal training preceded correction; equations, budgets and tolerances did not change.

Freeze source/test/audit/protocol hashes before formal training. Targeted manifests preserve earlier executed evidence. Post-run audit independently reduces CE with Python math, reloads every checkpoint, checks matrix equations at tolerance 1e-9 and zero argmax changes, verifies optimizer steps, all initialization/streams, six-member families and selection/decision rules. Preserve every failure. Stop after the declared runs and write diagnostic; no width/load grid extension or novel architecture proposal.

Sources: [DeltaNet 2024, section 2.2](https://arxiv.org/html/2406.06484v2) for correction and overlapping-key effects; [earlier local delta audit](../topics/attention/delta-memory-results.md) for the supplied-key witness. Trained interference must be measured separately from that mathematical example.
