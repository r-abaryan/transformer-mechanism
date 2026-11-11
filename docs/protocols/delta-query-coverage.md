# Query coverage at the original delta state size

2026-10-04. Question: does sampling different queries for unchanged histories improve simultaneous retention? The previous overlap study fits assigned queries at 100% but all-query training at only 63.59%/68.01%. This is a training-exposure question, not a new architecture claim.

## Matched intervention

Reuse the consistent-assignment task: 32 events, loads 2/8/16/32, 32 known keys, six value classes. Each key keeps its value throughout a history. Original delta reader: 615 parameters, 16x8 state, learned unit keys, values, classifier and scalar beta. No overlap penalty or wider state.

Fixed arm always supervises the history's assigned query. Resampled arm draws uniformly from its present keys at each appearance. Exactly one CE target per example in both arms. Separate query RNG preserves identical history/minibatch streams. Both arms generate identical potential query slots; fixed ignores them. Record actual query/label stream and per-history/key exposure counts. Sampling with replacement does not guarantee every pair is supervised.

Both arms: Adam .003, 3,000 updates, batch 64, CPU float64, one thread, cycling loads evenly. Per run: 192,000 supervised targets, 6,144,000 events, 48,000 targets per load. Match initialization and histories within each seed. Disposable CPU/CUDA complete optimizer-step profiles precede the formal lock; report peak memory including optimizer state. These profiles are not timings or formal GPU training.

## Development, confirmation and stop

Development seeds 3461/3463; four runs. Data seed 2026100444. Per load: training 768 histories (128 families x six renamings), validation 192 (32 x six). Exclude association-load and overlap-study development/confirmation families. No recipe search.

Confirmation eligibility requires every development seed/load to reach >=99% training accuracy on its objective: assigned queries for fixed, all present queries for resampled. Assigned/all-query metrics remain visible for both arms. If either gate fails, stop after four runs without generating confirmation data or evaluating test pools. No extra updates, LR changes or replacement seeds.

If eligible, lock development and use fresh seeds 3467/3469/3491, data seed 2026100445, also excluding current development families. Same training/validation counts; test 384 histories per load (64 x six). Train all six runs before test evaluation. Maximum ten runs, 30,000 updates, 1,920,000 targets, 61,440,000 events.

Frozen confirmation criteria: objective training fit >=99% in every seed/load; resampled all-query test-32 gain >=10 points in every pair; all-query test loads 2/8 preserved within two points; resampled complete-map test-32 accuracy >=95% in every seed. `benefit` requires fit/gain/preservation; `retention_supported` also requires complete-map reliability. Report each criterion separately. Fixed all-query fit is a diagnostic, not its training objective. No checkpoint selection or automatic extension.

## Measurements and checks

At every load/split, report assigned-query and all-query accuracy/CE, plus complete-map accuracy. All queries read one unchanged final state; do not move their writes or rebuild state per query in the native evaluator. Evaluate every training history. Complete-map success requires every present key correct, not just a high mean.

Five reusable preflight methods check labels, sampled-query gradients against the finite-average objective, actual initialized saved evaluator/audit at all loads with absent groups, wrong labels/logits/scalars/map counts, and inverse gate/decision controls. Independent audit uses matrix updates, Python CE/map reductions and full stream/exposure reconstruction. Data/weights/PyTorch are shared; policy replay shares tested functions. Preserve earlier evidence. Expected audit: 64 cells/238,080 predictions if stopped; 208/738,048 if confirmed. Counts are predictions, not independent experiments.

## Mathematics and sources

[Yang et al., DeltaNet 2024, section 2.2](https://arxiv.org/html/2406.06484v2) defines the correction update. Its input-dependent beta differs from our shared learned scalar. Inspected the update/read equations; the source does not predict our query-coverage scores.

Local identity for a history H and present keys Q: `E_q CE(z(H,q),y(H,q)) = sum_{q in Q} CE(z(H,q),y(H,q))/|Q|` under uniform sampling. A one-query loss/gradient is unbiased for this finite average; a finite run need not fit it. All-query evaluation adds measurements, not training targets. Earlier [Transformer query exposure](../topics/training/query-exposure-results.md) uses a different model/task; its scores are not paired controls here.
