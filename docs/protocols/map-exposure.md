# Does changing the training maps improve unseen retrieval?

Frozen before formal training, 2026-10-05. Question: does replacing half the finite high-load training pool improve unseen-map retrieval while preserving original-map fitting? Saved trajectories show gains and harms on unseen maps; they do not establish that finite-pool reuse caused them. This is one data intervention, not an optimizer sweep or architecture comparison.

## Paired continuation

Restore seeds 3473/3479/3481 after 12,000 archived Adam updates, including weights, moments, counters and sampling streams. Use the unchanged compact delta reader and Adam: learning rate .003, betas (.9,.999), epsilon 1e-8, no decay, schedule or clipping. CPU float64, one thread, serial execution, batch 64. No new warm-up.

| Arm | Active load-32 pool | Other training loads |
|---|---|---|
| Reuse | Original 128 base histories and six cyclic value shifts each | Original pools |
| Half new | 64 original bases plus 64 newly drawn value maps, six shifts each | Identical original pools |

Data seed 2026100529 chooses the replacement blocks once, shared across seeds. Every corresponding row preserves key/event order and its assigned query. New assigned-query labels disagree with old labels; added semantic maps, identified modulo common cyclic value shifts, exclude all original high-load train/validation/test maps. Each active pool still has 768 histories per load. Half-new therefore changes cumulative map exposure without increasing active pool size. It also removes feedback on half the old maps; those effects are not separately isolated.

Cycle loads 2/8/16/32; supervise every queried key with equal per-update cross-entropy weighting. Match sampled history indices and query RNG draws between arms. Seed budgets are 6,096/6,144/6,096 updates per arm, inherited from archived Adam continuations. Stop at those budgets, irrespective of scores. Reuse must reproduce archived final weights, optimizer state and recorded trace exactly.

Formal totals: 36,672 updates and gradients, 2,347,008 sampled histories, 75,104,256 input events and 34,031,616 supervised answers. Per update, batch 64 and 32 events; each four-load cycle supplies 64(2+8+16+32)=3,712 answers. Restored warm-up, preflight and discarded profile steps are separate exposure.

## Labels, splits and measurements

Values stay constant within each key's history. Independently reconstruct labels from events. A fresh draw at seed 2026100530 supplies 32 validation and 64 test base histories per load, each with six shifts: 192/384 histories. Exclude original and added history families; fresh load-32 semantic maps also exclude original and added maps. Lower-load semantic overlap is allowed and counted, so those are fresh histories rather than guaranteed new associations.

Evaluate full active training at steps 0, 1,536, 3,072, 4,608 and final. At start/final also evaluate original training, original reference validation/test and fresh validation/test, all four loads. Report accuracy, cross-entropy and complete maps, with all/unique/repeat groups and explicit empty groups. Expected 360 cells and 2,672,640 repeated evaluated answers: six runs, each with 5(768)(58)+2(768+2(192+384))(58) answers. These counts are not independent experiments.

| Frozen criterion, separately for every seed | Threshold |
|---|---|
| Active training fit | Accuracy >=99% in every nonempty group at every load, both arms |
| Original-map retention | Same gate on the complete original training pools |
| Exposure benefit | Fresh load-32 test accuracy improves by >=0.5 percentage points, with both active fitting gates met |
| Joint repair | Exposure benefit plus half-new original-map retention |

Report every seed and the all-seed conjunction. CE and complete-map differences are descriptive; they cannot replace a failed primary criterion. Validation, reference scores and intermediate checkpoints do not select candidates. One candidate, no replacement seeds or extensions. Low training fit limits a learning explanation without invalidating a verified data intervention.

## Execution, audit and scope

Preflight exercised the actual initialized saved producer/evaluator/auditor, labels, disagreeing values, ties, empty/missing groups, gradients, scalar Adam, RNG streams and intentional faults: two test methods passed. Two discarded complete CPU steps include Adam state (9,860 bytes); CPU peak memory was not measured and CUDA peaks are inapplicable. Their times are startup-sensitive checks, not a speed comparison.

Bind protocol, sources, preflight/profile records, starts and historical controls before training. Preserve prior evidence. The audit independently reconstructs labels, data exclusions, every recorded gradient/Adam update, saved scores and primary fit/accuracy criteria. The decision reducer and inherited training helper are shared; exact control replay is an additional check. Stop after audit and interpretation.

This is an Adam continuation on three reused initialization seeds and one shared new dataset. It does not isolate why the earlier L-BFGS trajectories changed, prove memorization, establish convergence or supply fresh initialization confirmation. RL and other branches remain parked.

Sources inspected: [Yang et al. (2024), section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2), delta writes/reads; [Muennighoff et al., JMLR (2025), journal abstract](https://www.jmlr.org/papers/v26/24-1000.html), repeated versus unique language data at fixed compute. The latter reports settings where several repetitions cost little and later repetitions have diminishing returns, not a universal penalty for reuse. It motivates separating data diversity from exposure; it predicts none of our scores. Our finite-map intervention, thresholds and counts are local choices, not a replication of either paper.
