# Delta retrieves reliably; the full tradeoff comparison does not qualify

Executed 2026-10-04. **Delta scores 100% in every evaluated group through 32 events. Both readers fit training perfectly, but the compact softmax reader fails long-input reliability. The frozen rule therefore skips GPU memory/timing measurements.** This is a conditional accuracy finding, not a measured efficiency win.

[Learning guide](delta-retrieval-guide.md) | [Frozen protocol](../../protocols/delta-retrieval.md) | [Source](../../../experiments/delta_retrieval.py) | [Raw report](../../../results/delta_retrieval/report.json) | [All metrics](../../../results/delta_retrieval/metrics.csv) | [Independent scalar audit](../../../results/delta_retrieval/independent_audit.json).

## Method and learning adequacy

Identical structured events, eight keys/six values, final-query latest-value targets. Unique queries occur once; repeats occur 2 through min(6,n) times with conflicting first/last values. Full pools are evaluated. Cyclic value-renaming families stay within one pool; confirmation excludes development families. Symbols and construction templates remain shared. No resets, absent keys or token-binding requirement.

Shared-start learned key/value tables and readout; tied unit-normalized key/query geometry. Softmax has 232 parameters and learns positive similarity scale/recency bias. Delta has 231 and learns a global correction strength. These are compact readers, not full Transformer/DeltaNet models. Different update/order mechanisms remain part of the treatment.

Eight development runs: two recipes, two seeds, two arms. All training cells reach 100%. Selection uses only worst full-training-group CE across seeds and lengths; both arms select constant Adam .003 over the declared cosine alternative.

| Reader | Constant worst training CE | Cosine worst training CE | Selected |
|---|---:|---:|---|
| Softmax | 0.000406 | 0.001351 | Constant |
| Delta | 0.000335 | 0.001074 | Constant |

Six fresh confirmation runs use seeds 3299/3301/3307 and new contexts. Each of all 14 formal runs completes 3,000 updates, batch 64, alternating lengths 4/8: 192,000 targets and 1,152,000 event records. Total: 42,000 updates, 2,688,000 targets. CPU float64, one thread. Summed per-run training/evaluation wall time is 141.5 seconds, excluding setup/final audit; this is execution provenance, not an architecture-speed benchmark. Two separate initialized optimizer-step profiles are not formal training runs.

## Fresh confirmation results

Accuracy percentages, three-seed means. Each test length has 384 sequences per seed (64 base-context families and six value renamings), balanced equally between unique/repeat queries. Delta's minimum over all seeds/groups is also 100%.

| Test events | Softmax unique | Softmax repeat | Softmax balanced | Delta, both groups |
|---|---:|---:|---:|---:|
| 4 | 100.00 | 100.00 | 100.00 | 100.00 |
| 8 | 100.00 | 100.00 | 100.00 | 100.00 |
| 16 | 86.46 | 95.83 | 91.15 | 100.00 |
| 32 | 58.33 | 90.63 | 74.48 | 100.00 |

| Seed | Softmax 16, unique/repeat | Softmax 32, unique/repeat | Delta 16/32, every group |
|---|---:|---:|---:|
| 3299 | 87.50 / 93.75 | 59.38 / 90.63 | 100.00 |
| 3301 | 87.50 / 96.88 | 56.25 / 93.75 | 100.00 |
| 3307 | 84.38 / 96.88 | 59.38 / 87.50 | 100.00 |

Both readers get every training, validation and short-test example correct. Thus softmax's longer errors are not incomplete short-task fitting. Delta preserves paired accuracy within the two-point allowance, but softmax fails 10 of 36 held-out reliability cells. No failed seed is replaced.

## What the result explains, and what remains open

Delta can learn these latest-value associations and retain useful answers over the tested longer contexts. The task only needs the current eight-key map, not every historical value. This favorable information requirement and fixed key universe limit broader conclusions.

The softmax reader's recency term has a known length sensitivity. For an older exact-match key at i and another key at j, with cosine similarity c and distance d=j-i:

\[
score_j-score_i=scale(c-1)+recency\,d.
\]

The newer key outranks the match when `recency*d > scale*(1-c)`. This follows from the declared reader equations. It is a candidate explanation for its transfer errors, not an isolated empirical cause: no recency intervention was performed. The result does not rank arbitrary softmax attention, contextual Transformer keys or published DeltaNet architectures.

The guide's cache/state byte counts are theoretical payloads. **No GPU state allocation, training peak or runtime benchmark was executed.** The combined accuracy-memory criterion is false because confirmation eligibility fails, not because a memory measurement failed. Whether this quality result comes with an actual memory/runtime advantage remains unmeasured.

## Verification and stopping

| Check | Evidence |
|---|---|
| Initialized math | 18 output/all-parameter-gradient cases and six directional checks pass |
| Error controls | Wrong correction, omitted recency, wrong labels/scalars, split leakage and wrong decision detected |
| Actual entry points | Missing-group metrics, initialized checkpoint reload, final audit and separate cache/read equivalence exercised on CPU |
| Final equations and scalar reductions | 80 pools, 36,096 predictions, 240 metric cells; zero argmax changes; max logit difference 4.40e-14 |
| Chronology/resources | Complete optimizer counters, common initial tensors, identical replayed batch streams; all 88 protected prior files unchanged |

The matrix-transition reference shares imported parameters/data; the post-run scalar reducer uses independent Python math. This is not a wholly independent software stack. Three fresh paired seeds support the task-specific result; prediction counts are not extra seeds.

The [2024 DeltaNet paper](https://arxiv.org/html/2406.06484v2) supplies the established correction and parallelization distinction. Our eager reader does not reproduce its full architecture, language benchmark or optimized throughput. [Newer context and scope](../../reference/sources.md).

Close this declared study. Preserve delta's reliable retrieval, the softmax transfer failure and the unmeasured resource tradeoff. No recency sweep, gate revision, new task or additional training is selected.
