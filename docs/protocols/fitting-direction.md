# Full-pool fitting: direction intervention

Frozen before formal execution on 2026-10-05. Question: does the L-BFGS direction improve fitting relative to the negative gradient when full-objective feedback and line-search policy remain fixed?

## Prediction and boundary

The direction explanation predicts lower final equal-load training CE by more than 1e-8 and no more training errors for L-BFGS in **each** seed. A failed conjunction limits this explanation. Report every paired effect even if the conjunction fails. No model or recipe selection; no further settings or budget extensions. This is diagnosis on previously opened pools, not fresh confirmation.

## Locked comparison

| Factor | Both branches |
|---|---|
| Starts | Saved Adam-12000 weights, seeds 3473, 3479, 3481 from fitting confirmation |
| Model | Unchanged 615-parameter delta reader, normal zero-state writes |
| Data | Identical confirmation pools: loads 2/8/16/32, train 768, validation 192, test 384 histories per load |
| Objective | Mean of four load-specific CEs; all 44,544 training answers per gradient |
| Budget | At most 128 closure/gradient evaluations per branch, including rejected trials; stop with fewer than two remaining |
| Solver | PyTorch 2.11 L-BFGS, lr 1, one iteration/call, history 20, max_eval min(26, remaining minus 1), gradient tolerance 1e-10, change tolerance 1e-12 |
| Search | Same native strong-Wolfe code, c1=1e-4, c2=0.9; same counters and initial trial-length policy |
| Runtime | CPU float64, deterministic, one thread/worker, three seed workers |
| Measurements | Training at start and crossed 32-gradient milestones/final; final validation/test, CE, accuracy, complete maps, all nonempty groups |

Intervention: immediately before computing the directional derivative, replace `d` with `-g` in one branch. Retain native curvature bookkeeping in both branches. First-step direction and initial line search must match exactly. Later accepted lengths, trial counts and update counts may differ as consequences of the intervention; exposure is capped, not assumed identical.

The installed step has no public direction hook. A guarded source adapter inserts only this assignment; hash the original and adapted source. Verify the control against unmodified native L-BFGS and previously saved control endpoints. Stop on implementation/audit failure, preserve artifacts and diagnose separately.

## Verification before training

Run the actual evaluator and audit on an initialized saved model through both branches. Independently check labels, full matrix gradients, directions, trial parameters, accepted displacement and all score cells. Inject gradient, direction, label, logit, counter and missing-group defects; require rejection. Profile two complete optimizer calls per branch, including optimizer state; exclude profiling from formal counts. Check the first-step null control and a non-isotropic quadratic where directions diverge.

Final audit checks every archived gradient and step, all groups and source hashes. Measure accepted-step Wolfe inequalities rather than assuming a bounded search always satisfies them. Archive trial losses, gradients, parameters and optimizer states. Publish all-seed primary decision before held-out evaluation; held-out results are descriptive and do not select settings.

## Interpretation and sources

The dataset assigns one constant value per key per history. It does not test changing-value overwrites. A direction benefit under this line-search policy does not establish that curvature is the only reason Adam differed, explain learned representations, prove convergence, or demonstrate generalization.

Inspected: [PyTorch 2.11 documentation](https://docs.pytorch.org/docs/2.11/generated/torch.optim.LBFGS.html) and [implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/lbfgs.py), including `step` and `_strong_wolfe`; [Liu and Nocedal (1989), author abstract](https://users.iems.northwestern.edu/~nocedal/Abstracts/limited-memory.html) for limited-memory curvature methods. Its uniformly convex convergence statement is not a theorem for our neural objective. The prediction and intervention are local, not source-backed score forecasts.
