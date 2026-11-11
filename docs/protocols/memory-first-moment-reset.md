# One-time memory first-moment reset: frozen causal test

Question: does removing stored memory first moments before step 11004 improve the errors present at that starting model, and does the effect survive one four-load cycle?

The [saved-step study](../topics/attention/adam-memory-error-transport-results.md) identified a seed-3463 stored/current numerator reversal at 11004. This is a targeted intervention on two reused seeds, selected after inspecting that behavior, not fresh confirmation or a proposed general optimizer.

| Item | Frozen choice |
| --- | --- |
| Starts | Existing audited traces' before-model/optimizer/RNG for seed 3461/3463 step 11004, after 11003 updates |
| Arms | Control unchanged; reset once by zeroing exp_avg for key.weight, value.weight and write_logit |
| Preserved | Model initialization, exp_avg_sq, Adam step counters, classifier moments, LR .003/betas .9/.999/eps 1e-8, data streams and objective |
| Budget | Exactly four updates per arm: 11004–11007, loads 32/2/8/16; four arms, 16 optimizer calls; no continuation beyond this horizon |
| Training | Batch 64 histories, all-query CE mean; same history indices and otherwise unused query draws per pair |
| Exposure | Per arm 256 sampled histories, 8192 writes, 3712 answer targets; total 1024/32768/14848 |
| Primary outcomes | Reset-minus-control CE on fixed starting train-32 errors, immediately and after four updates; full train-32 accuracy and equal-load training CE after four |
| Decision | Immediate benefit: error CE lower by >1e-10 in both seeds. Four-update benefit: error CE and equal-load train CE lower by >1e-10 and train-32 accuracy no lower, in both seeds. Report each rule separately; no recipe selection follows |
| Evaluation | Baseline and final: full train/validation/test pools at all four loads; immediate: full train-32 only; all/unique/repeat accuracy/CE/maps and fixed-error transitions |
| Counts | 76 evaluation cells, 566,016 answer evaluations; group counts overlap and are not independent seeds |
| Controls | Native control first step must equal archived trace exactly; all control beta logs must match original continuation; only declared first moments change at reset |
| Verification | Complete-step CPU preflight profile, actual train/evaluator/audit from initialized saved checkpoint, scalar Adam on every formal update, independent matrix labels/gradients/scores, corrupt moments/steps/labels/RNG/counts and missing groups |
| Precision | CPU float64 one thread; no GPU use; report full-step preflight time and tensor-state bytes, effective batch/exposure; GPU allocated/reserved zero, no CPU peak-memory claim |
| Stop | Four updates regardless of scores. No second reset, alternate reset, extra update, strength/LR search, replacement seed or held-out tuning |

Error membership is fixed at the common start, so both arms score the same cohort. Full training criteria still require all groups and the exact prior 99% fit gate; the intervention decision does not substitute for adequate task fitting. Validation/test are descriptive on already opened pools, never selection metrics. Old checkpoints and failed budgets remain unchanged.

Resetting exp_avg removes one historical numerator contribution at the first shared-gradient update. It does not reset all Adam history: second moments and bias-correction counters remain. After that step, the arms' gradients and second moments can diverge. Finite outcomes need not equal first-order projections.

Sources inspected: [Kingma and Ba, Algorithm 1](https://arxiv.org/pdf/1412.6980) and [PyTorch 2.11 Adam](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html), moment and bias-corrected update equations. The intervention, outcomes and local expected first-step decomposition are our construction; sources predict no local score or reset benefit.
