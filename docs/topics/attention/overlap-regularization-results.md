# Lower key overlap gives small gains, not complete retention

Executed 2026-10-04. **The selected overlap penalty improves mean 32-key all-query test accuracy from 62.19% to 65.79%. Every confirmation pair improves, but none meets the frozen 10-point gain rule. Neither arm produces a completely correct 32-key map.** State size remains 16x8.

[Learning guide](overlap-regularization-guide.md) | [Frozen protocol](../../protocols/overlap-regularization.md) | [Source](../../../experiments/overlap_regularization.py) | [Report](../../../results/overlap_regularization/report.json) | [Every seed/pool](../../../results/overlap_regularization/metrics.csv) | [Aggregates](../../../results/overlap_regularization/summary.json) | [Audit](../../../results/overlap_regularization/independent_audit.json).

## Method and selection

Unchanged delta reader: 615 parameters, 16-dimensional unit keys, 8-dimensional values, scalar write strength beta and six-class readout. Histories contain 32 events and 2/8/16/32 known keys with consistent values. Training supervises one final selected query per example. Objective is `CE + lambda*R`, where R is average squared dot product over 992 ordered pairs of different normalized keys. Baseline lambda is zero.

Development: seeds 3407/3413, lambda 0/1/10, six runs. Positive candidates require >=99% selected-query training fit at every load and validation 2/8 preservation within two points of matched baseline. Both qualify. Select by highest mean validation all-query accuracy at load 32, then CE, then lower weight. No test scores enter selection.

| Lambda | Development all-query validation % | CE | Eligible |
|---|---:|---:|---|
| 0, reference | 63.00 | 1.986244 | Baseline fit passes |
| 1 | 67.37 | 1.708058 | Yes |
| 10 | 69.43 | 1.573508 | Yes, selected |

Fresh confirmation: three baseline/lambda-10 pairs, seeds 3433/3449/3457, new context families disjoint from development and earlier association-load pools. Shared symbols/templates remain. Train/validation/test counts per load: 768/192/384 examples, or 128/32/64 context families with six cyclic value renamings. Every arm shares its seed's initialization and minibatch stream. All confirmations finish before test evaluation.

Every run uses Adam .003, 3,000 updates, batch 64, CPU float64 and one thread: 192,000 supervised queries and 6,144,000 input events, cycling loads evenly. All twelve runs complete: **36,000 updates, 2,304,000 supervised queries and 73,728,000 events**. Summed per-run training/evaluation time is 384.5 seconds, excluding preflight/audit; this is execution provenance, not a speed comparison.

## Confirmation: all 32 queries from each final state

Mean accuracy % / task CE across three seeds. Complete maps require all 32 queries correct. These additional query labels are evaluated, not supervised.

| Pool | Baseline accuracy % / CE | Penalty accuracy % / CE | Complete maps: baseline / penalty |
|---|---:|---:|---|
| Training | 63.59 / 2.036275 | 68.01 / 1.784874 | 0 / 0 of 2,304 each |
| Validation | 63.16 / 2.074401 | 66.56 / 1.841096 | 0 / 0 of 576 each |
| Test | 62.19 / 2.111699 | 65.79 / 1.896489 | 0 / 0 of 1,152 each |

| Seed | Baseline all-query test % | Penalty all-query test % | Gain, percentage points | R: baseline / penalty |
|---|---:|---:|---:|---|
| 3433 | 62.32 | 64.93 | 2.60 | 0.044122 / 0.032261 |
| 3449 | 62.54 | 66.46 | 3.91 | 0.041425 / 0.032261 |
| 3457 | 61.69 | 66.00 | 4.31 | 0.041763 / 0.032261 |

R's theoretical minimum is 1/31, approximately 0.032258. The treatment approaches that minimum without approaching complete retention. Mean beta also changes from 0.9337 to 0.9005. This is a matched effect of the added training penalty, not an isolated post-training geometry change.

## Selected-query results

Both arms score 100% on every supervised training pool in every confirmation seed. Mean selected-query training CE at load 32 is 0.001612 baseline versus 0.002503 penalty. High fit on these assigned queries does not establish that every association in each training history is retained.

| Keys | Baseline validation % | Penalty validation % | Baseline test % | Penalty test % |
|---|---:|---:|---:|---:|
| 2 | 100.00 | 100.00 | 100.00 | 100.00 |
| 8 | 100.00 | 98.96 | 99.48 | 100.00 |
| 16 | 78.99 | 85.76 | 90.45 | 94.88 |
| 32 | 61.11 | 61.11 | 62.85 | 63.11 |

All per-seed losses, selected-query scores and complete-map scores remain in the CSV/report. For example, treatment seed 3457 has validation-8 accuracy 96.875% while its test-8 accuracy is 100%. Do not substitute the mean for reliability. Constant-class chance is 16.67%; supplied dictionary lookup gets 100%. Pool-specific last-event controls are in the report.

## Frozen criteria and interpretation

| Criterion | Result |
|---|---|
| >=99% supervised training fit, every run/load | Pass |
| >=10-point all-query test gain, every pair | Fail |
| Test loads 2/8 preserved within two points | Pass |
| >=99% all-query training accuracy in both arms | Fail |
| >=95% penalty complete-map test accuracy, every seed | Fail |
| Bounded benefit / simultaneous-retention support | Both fail |

The penalty produces small consistent gains at the original memory size. Lower average overlap alone is insufficient for this recipe to retain complete maps. Because both arms also fail all-query training adequacy, the remaining simultaneous-retention failure is not solely a held-out generalization problem. This does not invalidate the verified 100% supervised selected-query fit.

Minimum average squared overlap need not separate every pair: duplicated basis keys can attain 1/31. The preflight verifies that witness. The observed low R does not establish that these trained keys form that particular configuration, identify the residual cause, prove a hard capacity limit or rule out other training objectives. No claim of convergence or general language improvement follows.

## Sources and comparison with published mathematics

[Yang et al., DeltaNet 2024, section 2.2](https://arxiv.org/html/2406.06484v2) was inspected for the correction rule. It motivates the query/write overlap term; it does not predict these local gains. Our scalar beta and compact reader differ from the paper's full models.

[Casazza, Fickus and Mixon, 2010, introduction and section 2](https://arxiv.org/pdf/1009.5562) define frame potential and its dimension-dependent bound. Our guide specializes that bound to R>=1/31. Joint CE plus overlap regularization with Adam is not their frame-only procedure and does not inherit its convergence results. This study tests an established geometric objective as a local training intervention, not a novel architecture or paper benchmark reproduction.

## Verification and stop

Five preflight methods pass: independent penalty value/gradient/directional checks; deliberate zero-gradient/diagonal errors; bound fixture; actual initialized checkpoint evaluation/audit with scalar/label errors; selection/decision inverse controls. Four disposable complete CPU/CUDA optimizer steps precede the formal lock. CUDA float64 allocated/reserved peaks are 71,904,256/73,400,320 bytes for both profiled weights, including optimizer state. Formal training remains CPU; those profiles are not formal training-memory or speed comparisons.

The audit passes all twelve saved runs, 150 metric cells and 497,664 evaluated predictions. Maximum logit equation error is 1.60e-14, with zero argmax disagreements. Scalar CE, complete-map counts and penalty use separate reductions; labels, disjoint families, optimizer counters, initializations and streams verify. Selection/decision replay shares producer functions with preflight inverse controls; data, parameters and PyTorch are also shared.

All 221 protected earlier files remain unchanged. The declared budget is closed, preserving the gains and failed criteria. No extra penalty strengths, epochs, replacement seeds or automatic architecture change.
