# Learned memory control: both readers solve this bounded task

Executed2026-10-03 and closed. Six completed runs, no tuning or restarts. [Educational guide](learned-memory-control-guide.md) | [Frozen protocol](../../protocols/learned-memory-control.md).

**Both plain delta and gated delta achieve100% in every training, validation and test group across all three seeds.** The added forgetting gate yields zero accuracy gain; its frozen adoption criterion fails. This is a task ceiling, not evidence that forgetting is universally unnecessary.

## What was trained?

Small structured-event memory readers: three keys, four values, one reset marker and five output classes including unknown. Key/value projections, write strength, query embedding and classifier are learned. Gated delta additionally learns retention. Both use an8-by-8 state and L2-unit keys. There is no manual reset of model state when a marker appears.

Plain delta has222 parameters; gated delta231. The extra nine parameters and initial retention sigmoid(2) are part of the treatment. Shared initial tensors and data streams match exactly within seeds2711/2719/2729. These are memory modules, not complete Transformer or published DeltaNet models.

Each run:3,000 Adam updates, lr0.003, batch64 with16 examples per group, alternating6/8 events. That is192,000 final-query targets and1,344,000 event records. CPUfloat64, one thread. A complete optimizer step was profiled before freezing choices; GPU allocated/reserved bytes0, CPU peak memory unmeasured. Timings are not hardware performance evidence.

Fresh data contain4,992 sequences:3,072 training,768 validation and1,152 test. Four cyclic value renamings stay within each canonical context family. Families are disjoint across pools, but symbols and construction templates are shared. Tests cover6/8/12 events; all six attempts finish before test scoring.

## Accuracy and reliability

Each cell below is the minimum accuracy across all three seeds and all declared pools at that stage. Thus100% means no weak group or seed is hidden.

| Group | Delta train/validation6,8 | Gated train/validation6,8 | Delta test6,8,12 | Gated test6,8,12 |
|---|---:|---:|---:|---:|
| Retain another key | 100% | 100% | 100% | 100% |
| Overwrite queried key | 100% | 100% | 100% | 100% |
| Reset, queried key rewritten | 100% | 100% | 100% | 100% |
| Reset, queried key absent | 100% | 100% | 100% | 100% |

Both short-fit and reliability gates pass. Every seed's paired gain is0 points in every longer-test group. The required10-point improvement in both reset groups is absent.

| Seed | Delta test12 mean CE | Gated test12 mean CE | Accuracy gain |
|---|---:|---:|---:|
| 2711 | 0.00003907 | 0.00006964 | 0 points |
| 2719 | 0.00004387 | 0.00008014 | 0 points |
| 2729 | 0.00004349 | 0.00006739 | 0 points |

The [complete table](../../../results/learned_memory_control_metrics.csv) records accuracy and cross-entropy loss for all168 seed/arm/pool/group cells. Low losses support confident correct predictions here; the study did not select a winner by loss or establish convergence.

## Why the controls matter

| Test12 baseline | Retain | Overwrite | Reset present | Reset missing |
|---|---:|---:|---:|---:|
| Always unknown | 0% | 0% | 0% | 100% |
| First matching value | 100% | 0% | 0% | 0% |
| Ignore reset, use latest match | 100% | 100% | 100% | 0% |
| Correct oracle | 100% | 100% | 100% | 100% |

No listed shortcut solves every group. Nevertheless, these templates are limited: the query key occurs once or twice, and reset cases contain one marker. Success does not establish arbitrary event processing or rule learning beyond this distribution.

## Independent audit

Six initialization cases pass independent matrix-transition outputs and all parameter gradients, maximum gradient discrepancy8.33e-17. Omitting old-read subtraction is detected; shifted labels differ from independently audited targets. Family separation and baseline labels pass.

All42 final pools and29,952 predictions pass a separate matrix-transition forward calculation, with zero argmax changes. Maximum logit error9.77e-15; independent cross-entropy/group metric error1.60e-16. Checkpoint hashes,3,000 optimizer steps, shared initialization, replayed minibatch streams and pretest chronology pass. All270 prior JSON artifacts remain unchanged.

## Interpretation and stopping decision

The plain reader can produce the required reset-aware answers without an explicit scalar forgetting gate. That does not prove its full memory becomes zero, reveal its internal strategy or contradict the earlier arithmetic audit. Learned projections and the readout may encode the task differently from the hand-supplied examples.

[Gated Delta Networks v3, section3.1 equation10](https://arxiv.org/html/2412.06464v3) motivates the combined update. Our result neither reproduces its language benchmarks nor challenges its equation. It shows no additional accuracy benefit for the gate in this small, fully fitted task.

**Decision:** no evidence here to prefer the added gate. Close the declared study; do not increase difficulty or budget automatically to obtain a separating result.

Evidence: [execution and audit source](../../../experiments/learned_memory_control.py), [raw report](../../../results/learned_memory_control.json), [audit](../../../results/learned_memory_control_audit.json), [profile](../../../results/learned_memory_control_profile.json), [fresh data](../../../results/learned_memory_control_data.json).
