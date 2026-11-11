# Learned global relay: useful, but longer-input reliability fails

Executed and closed2026-10-02. Nine runs and the fixed ablation are independently audited. Global connections enable distant retrieval at the training length. One seed loses longer-input reliability despite a valid information path.

[Educational guide](global-retrieval-guide.md) | [Frozen protocol](../../protocols/global-retrieval.md)

## Main findings

All seeds have the following training, validation and length17 test accuracy:

| Distance to query | Full | Local | Local plus middle global |
|---|---:|---:|---:|
|1,4 |100% |100% |100% |
|5, after global |100% |16.67% |16.67% |
|12,16, before global |100% |16.67% |100% |

The local two-layer radius2 limit is four positions. The global marker creates a causal two-layer route for earlier values at12/16. It cannot collect the later value at distance5. Each six-answer family supplies identical accessible inputs when no route exists, so the exact unreachable score is one sixth.

At length21, full and local retain the same pattern. Global distant accuracy falls for one seed:

| Seed | Length17 distance12 /16 | Length21 distance12 /16 | Ablated distant accuracy, both lengths |
|---|---:|---:|---:|
| 2521 | 100.00 / 100.00 | 100.00 / 100.00 |16.67% |
| 2531 | 100.00 / 100.00 | 100.00 / 100.00 |16.67% |
| 2539 | 100.00 / 100.00 | 83.33 / 83.33 |16.67% |

Each test distance contains192 examples (32 families, six values). Seed2539 gets160/192 at both long distant groups; the other cells are192/192. Removing the extra global query/key connections gives32/192 in every distant test cell. Both lengths retain100% global accuracy at distances1/4 and16.67% at5. The mean94.44% at each long distant group hides the83.33% seed.

## Methods and budget

Three fresh matched sets, seeds2521/2531/2539, full/local/global. Each model has4544 parameters, two width16 pre-norm MQA blocks, two query heads/one K/V head, FFN32 ReLU and RoPE offset4. All arms receive marker11 at the middle, answer symbols12..17, distractors0..10 and final query18. Only the attention mask changes; initial states and training streams match within each seed.

CPU float64, one thread, Adam.001,6000 updates, effective batch60. Each run360,000 sampled examples/final targets and6,120,000 input tokens. All19 output classes remain available. No schedule, clipping, dropout or checkpoint selection. Full training/validation pools at2000/4000/6000; all nine final attempts saved before testing.

Train17 contains3840 examples; validation17/test17/test21 contain960 each. Whole distractor families and exact inputs are disjoint, while vocabulary/subsequences overlap. All five distances and six values are expanded per family. No new task training failures, replacements or extensions occurred.

The complete-step CPU profile includes optimizer-state allocation. GPU peaks are not applicable. Dense masking is used, so elapsed times are not sparse-kernel evidence. The longer test moves the global marker from position8 to10 and adds context while keeping source-to-query distances fixed. It does not test arbitrary-distance extrapolation.

## Per-seed losses

All-distance cross-entropy, including structurally unreachable examples:

| Seed | Arm | Train17 | Validation17 | Test17 | Test21 |
|---|---|---:|---:|---:|---:|
| 2521 | full | 0.000063 | 0.000063 | 0.000063 | 0.000065 |
| 2521 | local | 1.075167 | 1.075168 | 1.075165 | 1.075173 |
| 2521 | global | 0.358467 | 0.358475 | 0.358468 | 0.366129 |
| 2531 | local | 1.075118 | 1.075226 | 1.075130 | 1.075130 |
| 2531 | global | 0.358544 | 0.358544 | 0.358560 | 0.371720 |
| 2531 | full | 0.000051 | 0.000052 | 0.000052 | 0.000054 |
| 2539 | global | 0.358454 | 0.358459 | 0.358457 | 0.537943 |
| 2539 | full | 0.000058 | 0.000058 | 0.000058 | 0.000060 |
| 2539 | local | 1.075145 | 1.075166 | 1.075159 | 1.075166 |

Aggregate accuracy ceilings are50% local and83.33% global under perfect reachable performance. Corresponding loss lower bounds are `(3/5)ln(6)=1.075056` and `(1/5)ln(6)=0.358352`. The guide derives them. These bounds explain why aggregate loss/accuracy alone would misrepresent eligible fitting.

The oracle scores100% per group; constant12 and expected uniform-six prediction score1/6. Last-context prediction scores100% only at distance1 and0% elsewhere.

## Ablation and frozen decisions

The ablation removes both kinds of extra global edges in both layers, with weights fixed. Distant accuracy falls by83.33 percentage points in ten test cells, and66.67 points in seed2539's two long cells. This demonstrates a measured dependency on these connections under the intervention. It does not isolate query edges from key edges or identify a unique internal representation.

Nearby performance can also change. Native distance1/4 accuracy is100% throughout; ablated accuracy is:

| Seed | Test length | Distance1 | Distance4 |
|---|---:|---:|---:|
| 2521 | 17 | 93.75% | 81.25% |
| 2521 | 21 | 94.27% | 80.21% |
| 2531 | 17 | 100.00% | 95.83% |
| 2531 | 21 | 100.00% | 95.31% |
| 2539 | 17 | 100.00% | 99.48% |
| 2539 | 21 | 100.00% | 100.00% |

| Frozen criterion | Result |
|---|---|
| Every arm, eligible training>=99% |Pass |
| Every arm, eligible validation and test17>=95% |Pass |
| Full/local eligible test21>=95% |Pass |
| Global eligible test21>=95% in every seed/group |Fail: seed2539 distant groups83.33% |
| Global dependence rule across both lengths |Fail: native>=95% clause fails in those two cells, although every distant ablation reaches1/6 |
| Structurally unreachable family invariance |Pass in every native and ablated pool |

The combined dependence criterion must remain failed. The observed ablation effect remains valid evidence. Neither result diagnoses why the weak seed loses transfer.

## Independent verification and limits

Independent gathered attention, normalization, rotation and FFN equations check all native and ablated outputs. All48 evaluations (36 native,12 ablated),80,640 predictions and the independent decision check pass. Maximum logit error3.91e-14; metric error3.55e-15; zero argmax changes. Each count is an evaluation of a reused family, not an independent training seed.

Preflight verifies all parameter gradients, exact full-baseline equivalence, future-input masking, inaccessible-family invariance and a missing-window error control. Independent scans check6720 labels and224 families and detect intentionally shifted labels. Initial weights, sampled streams, checkpoint steps and the pretest snapshot are verified. All247 recorded historical JSON artifacts and178 frozen files retain their hashes.

Useful global relay is established for this finite task at length17. Reliable longer transfer fails for the global arm even though the structural route remains. Marker placement, learned representation, optimization and their interactions remain possible explanations; none is isolated here. General language quality, information capacity and sparse execution performance remain untested. Study closed without tuning or a new experiment.

[Raw report](../../../results/global_retrieval.json) | [Audit](../../../results/global_retrieval_audit.json) | [Dependence counts](../../../results/global_retrieval_dependence.json) | [Data](../../../results/global_retrieval_data.json) | [Training code](../../../experiments/global_retrieval.py) | [Audit code](../../../experiments/audit_global_retrieval.py)
