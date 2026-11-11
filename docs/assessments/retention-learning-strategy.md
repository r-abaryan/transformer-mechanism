# A successful reference for retention learning

Strategy decision, 2026-10-05. One central question: **can the unchanged reader reliably learn the failing finite training task, and what separates successful learning from unsuccessful learning?** [Development](../topics/attention/full-pool-fitting-results.md) and [fresh confirmation](../topics/attention/full-pool-fitting-confirmation-results.md) provide adequate-fitting references and perfect finite-pool witnesses. The [direction contrast](../topics/attention/fitting-direction-results.md) now identifies one procedure contribution at fixed full objective/search policy. Stop fitting search; the learned representation and generalization mechanisms remain unresolved.

## What the evidence distinguishes

| Explanation | Current evidence | Distinguishing prediction / missing evidence |
|---|---|---|
| The saved reads block perfect affine decoding | [Exact certificates](../topics/attention/shared-readout-recovery-results.md) establish this across all four loads in both endpoints | Already established for those fixed reads. Changing only their affine classifier cannot make the full pool perfect |
| The unchanged architecture cannot perfectly fit the finite pool | Development and fresh confirmation produce perfect normal-write checkpoints on their respective pools | Refuted for these two finite pools. Universal capacity and all-seed perfection are not established |
| Fitting depends on the training procedure | Direction-only L-BFGS versus negative gradient improves objective/errors in all three seeds under the same full objective/search policy | The direction rule contributes here, including scaling and subsequent search outcomes. It does not explain the entire multi-factor Adam comparison or identify the successful representation change |
| Direct routing/decoder curvature transport is needed to finish fitting | Removing it leaves 0/0/24 errors versus native 0/0/6; two perfect witnesses remain, with worse held-out 16/32 CE in every pair | The all-seed necessity prediction fails; transport helps the weakest seed under this cap. Global history/objective still couple the blocks, and from-scratch learning remains untested |
| Classifier expansion is the main fitting blocker in the tested window | [Norm cap](../topics/attention/readout-norm-cap-training-results.md) prevents head growth but worsens fitting; values and logits still grow | The tested cap fails as a repair. Its outcome cannot rule out all scale effects. Further scale variants need a distinct prediction and a check of total logits/read scale |
| Adequate fitting still leaves a generalization failure | Fresh test-32 accuracy is about 87–89%, with rare complete maps and L-BFGS CE above Adam's in every seed | Established on fresh same-length histories. Independent dataset replications and longer/distribution-shift tests remain separate; no learned algorithm is identified |

These explanations can coexist. Interference is a measured computation; optimization and representational restrictions can both contribute to it. An informative experiment must narrow the alternatives even when accuracy does not improve.

## The normal-write witness, explained

The actual reader computes

$$S_t=(I-\beta k_tk_t^T)S_{t-1}+\beta k_tv_t^T,\qquad S_0=0,$$
$$z_\theta(h,q)=W S_\theta(h)^Tq_\theta+b.$$

Keys, values, beta and the classifier share one parameter vector theta across histories. A perfect finite-pool witness means

$$\exists\theta\;\forall(h,q,y)\in D_{train}:\quad\operatorname{argmax} z_\theta(h,q)=y.$$

The state solver instead established, at fixed learned queries/classifier, a separate correct state for every tested load-32 history. It did not require that normal writes produce those states. The two new L-BFGS checkpoints meet the stronger finite-pool requirement through normal writes, with positive saved-logit margins. Oracle states and learned parameter settings are different evidence.

The finite pool and the full task distribution are different targets. A fitting witness applies to that pool. Universal representability would require its own mathematical argument or construction; finite testing cannot establish it.

## Bounded development requirements

The original requirements below led to the [frozen full-pool comparison](../protocols/full-pool-fitting.md), now completed and audited. They remain the boundary for development, not authorization to repeat it.

Retain the original 615-parameter model, normal zero-start writes, task labels, all four loads and exhaustive final-query objective. Inspect prior recipes first. Select the original procedure and **one justified alternative procedure**; freeze its settings, seeds, starts, compute allowance and selection rule after complete-step profiling.

Use training evidence for development selection. Preserve the original failing recipes. Report every attempt, full training accuracy/CE/maps, all relevant groups and weak seeds. Maintain the existing exact >=99% all-training-group fitting gate as a readiness criterion; 100% accuracy is the stronger finite-pool representability witness. Neither criterion proves reliable complete maps on unseen histories.

Declare how budgets match. Batch size, updates, exposure and optimizer cost can differ; report each. A multi-change recipe can establish a successful reference, but attributing its success to one change requires a subsequent matched intervention.

The original recipe is the failure reference. Already successful lower-load groups provide evaluator controls; they do not settle the high-load problem. Check the real evaluator/audit and intentional faults before training. Verify any proposed intervention changes its intended mechanism.

| Outcome at the frozen limit | Permitted decision |
|---|---|
| All declared development seeds meet the fitting gate | Lock the recipe; reserve fresh confirmation, then compare successful/failing trajectories using predeclared predictions |
| A single parameter setting fits the whole training pool perfectly | Record a finite-pool normal-write witness; measure reliability separately |
| One seed passes, others fail | Record seed dependence; the reliability gate remains unmet |
| No candidate passes | Retain unresolved feasibility/optimization; reassess the model/task scope. Stop the attempt without extra settings, replacement seeds or extended budgets |

Development and fresh confirmation have stopped at their declared budgets. The 99% gate repeats in three fresh seeds on one new dataset; all-seed perfection fails. Preserve the held-out losses, incomplete maps and six-error exception. More fitting updates or recipe candidates are not the next step.

The subsequent [frozen direction test](../protocols/fitting-direction.md) has completed: 766 gradients, 306 updates, 168 cells, all audited. L-BFGS leaves 0/0/6 errors versus 122/176/204 for negative gradient and passes the predeclared benefit rule. Reused starts/pools limit confirmation scope. L-BFGS makes more updates within the evaluation cap, yet held-out CE worsens. Close direction variants and extra fitting budgets.

The [saved routing/decoder diagnosis](../topics/attention/fitting-representation-results.md) now tests component sufficiency. Both single-donor rules fail in all three seeds, with one strong routing-only partial success. Conditional decoder accuracy effects reverse with the routing donor. At fixed routing, the L-BFGS decoder carries a held-out CE penalty. All 144 cells audit with zero training; these are endpoint effects, not a historical learning explanation.

The [fixed scale/shape diagnostic](../topics/attention/decoder-scale-shape-results.md) now passes all 12 primary scale-dominance contrasts. Scale removal sharply lowers held-out CE while retaining every answer and map, including 0/0/6 training errors. Non-scalar decoder differences still matter for the class-decision improvement. All 288 cells audit with zero training; the weight projection is metric dependent and does not measure calibration.

Close scale/donor variants. The [saved transport review](fitting-transport-review.md) verifies all 306 calls and finds cross-block action at every later L-BFGS call. That action usually opposes local descent, while the within-block part supplies stronger descent. Presence and cancellation do not establish fitting benefit or isolate non-scalar decoding.

The [controlled removal](../topics/attention/fitting-transport-results.md) is now complete. Native traces/endpoints match exactly; all 765 gradients, 369 calls and 168 cells audit. Removing direct cross-block action preserves zero training errors in two seeds, but raises the weakest from 6 to 24. Its training CE improves in all pairs while held-out 16/32 CE worsens throughout. The all-seed fitting-benefit rule fails; the weakest-seed contribution remains evidence.

Close transport/direction variants and fitting extensions. The [causal account](retention-learning-causal-account.md) now consolidates measured effects and unknowns without new experiments. Prioritize how fitting gains produce representations that remain unreliable on unseen maps, with competing predictions before selecting another test. Global secant history and the shared objective still connect the blocks. Within-block preconditioning and full historical learning remain unresolved.

The [fixed-example trajectory diagnostic](../topics/attention/fitting-trajectory-results.md) now measures all saved stages. Both coefficient-margin signatures fail. Initially wrong test cohorts improve, while initially correct cohorts lose margin and many answers. This supports examining training exposure as a candidate explanation, not a finding of memorization or universal decoder failure. Close further trajectory partitions; any intervention needs a distinct frozen protocol and adequate fitting. Last-write exceptions remain evidence to preserve. Copy/state remain deferred and RL remains parked.

## Source and review scope

The separately [frozen map-exposure contrast](../topics/attention/map-exposure-results.md) now completes six matched Adam continuations. New maps lower held-out CE but produce mixed accuracy, active fit below 99% and lost original-context performance. All 36,672 updates and 360 cells audit with exact controls. Close this replacement policy. It couples new exposure to removal of old feedback and does not establish finite-pool memorization. A stronger future comparison needs bounded training-adequacy development before confirmation; no new run is queued.

[Yang et al. (2024), section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2) was reopened for delta writes/reads. The equations above transpose its memory orientation to match [our reader](../../experiments/delta_retrieval.py); [association-load construction](../../experiments/association_load.py) supplies the expanded key vocabulary. Our tied key/query, scalar-beta reader differs from full DeltaNet. The witness statement, hypothesis table and development rules are local reasoning, not paper predictions.

Initial review inspected the checkpoint/agenda, failure register, continuation, shared-state feasibility, fixed-read recovery and norm-cap reports, plus the actual write/read implementation. That review executed no optimizer calls, gradients or model evaluations and was not an exhaustive code audit. Subsequent training and verification are separately documented in the [development](../topics/attention/full-pool-fitting-results.md) and [confirmation](../topics/attention/full-pool-fitting-confirmation-results.md) reports.
