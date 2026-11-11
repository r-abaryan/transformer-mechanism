# Intermediate supervision helps learning, but transfer remains weak

**Executed and closed.** Six matched attempts finish and the independent audit passes. Extra operation-state supervision improves training fit and mean short-length held-out accuracy. The fixed success rule fails; difficult16/24-operation tests remain near50% on average.

[Educational guide](state-supervision-guide.md) · [Frozen protocol](../../protocols/state-supervision.md).

## Method and math

Same4,080-parameter MQA architecture, input tokens, training pools and optimizer. Fresh paired seeds2039/2053/2063 have identical initial parameters and sampling streams within each pair. Three3,000-update attempts per arm, Adam.001, batch64, alternating8/12 operations. CPU float64, one thread, PyTorch2.11.0+cu128. Arm order alternates across seeds.

Final-only loss is final QUERY cross-entropy. Intermediate loss adds mean cross-entropy of the state AFTER every operation, with auxiliary coefficient1. The final-answer coefficient remains1. For each supervised logit, `dCE/dz = softmax(z) - one_hot(target)`, weighted by its averaging factor. No intermediate answers enter the inputs or are fed back during inference.

Each arm sees192,000 final targets and2,304,000 input tokens per run. Intermediate arms additionally use1,920,000 operation-state targets. Objective scale, feedback quantity and supervised positions change together; this comparison does not isolate their separate effects or establish equal runtime cost. All final training-pool, validation and test metrics were planned before training. Validation/test examples are reused from the prior study; this is not fresh confirmation.

## Final-answer results

Accuracy (%), equal means across three seeds. F=final-only; I=intermediate. Each8-operation training/validation/test cell contains128/64/64 examples; each12-operation cell512/128/256. Longer test cells contain512 examples. Constant prediction scores50%.

| Operations | Group | Train F | Train I | Validation F | Validation I | Test F | Test I |
|---|---|---:|---:|---:|---:|---:|---:|
|8|No resets|64.58|90.10|47.92|90.10|40.62|84.90|
|8|Long tail|63.54|96.35|40.62|93.23|42.19|98.96|
|8|Mixed resets|99.48|99.74|95.83|95.83|96.88|98.96|
|12|No resets|52.21|74.22|48.70|69.53|48.70|71.09|
|12|Long tail|52.02|77.67|50.26|70.05|49.87|77.47|
|12|Mixed resets|95.83|97.92|91.15|96.61|94.01|96.48|

| Operations | Group | Test F | Test I |
|---|---|---:|---:|
|16|No resets|50.07|50.65|
|16|Long tail|51.37|55.27|
|16|Mixed resets|93.62|94.99|
|24|No resets|50.13|49.41|
|24|Long tail|51.04|50.59|
|24|Mixed resets|91.80|93.68|

## Seeds and fixed decision

Primary metric averages the12-operation no-reset and long-tail groups equally. Decision is validation-only, frozen and hashed before test scoring.

| Seed | Train F | Train I | Validation F | Validation I | Validation gain, pp | Failed condition |
|---|---:|---:|---:|---:|---:|---|
|2039|52.83|96.48|49.22|93.75|44.53|None|
|2053|52.25|69.82|47.27|62.50|15.23|Training fit below90%|
|2063|51.27|61.52|51.95|53.12|1.17|Gain below10pp; fit below90%; mixed retention|

Seed2063's eight-operation mixed-reset validation falls100% to87.50%, exceeding the allowed5pp loss. Mean mixed-reset improvements elsewhere do not erase this regression. Separately,26 of36 intermediate test seed/group/length cells miss the original99% short/90% long thresholds, versus29 of36 final-only cells. These thresholds are practical criteria, not significance tests.

## What we can conclude

The same architecture learns substantially more with this objective package in some runs. Therefore the earlier failure cannot justify claiming an architectural inability. Extra supervision does not reliably solve this setup: two seeds still fit the difficult training groups incompletely.

Even seed2039, with96.48% difficult-group training fit at12, scores51.17%/60.35% on16-operation no-reset/long-tail tests and48.63%/50.39% at24. Better training fit does not guarantee length transfer. Which representation, optimization trajectory or positional behavior explains this remains unknown.

Intermediate-readout accuracy can also hide final-answer failures. Seed2053 scores90.09%/92.66% on the twelve-operation training states, yet its final-answer accuracy is66.60%/73.05% for the same groups. An average over positions is not an end-of-sequence reliability measure. Raw records retain all seeds, groups, labels and curves.

## Verification and stopping

All99,840 prefix labels agree with the independent last-reset/parity solver. Six objective/gradient preflights pass, maximum gradient error6.11e-16; deliberately shifted targets change loss and gradients. Original numerical verification is reused with source hashes checked.

Independent audit passes144 final-checkpoint cells:39,168 final predictions and599,040 operation-state predictions, zero argmax differences. Maximum full-logit error1.16e-13; final-loss error3.11e-15. Paired initial states, sample streams, checkpoint hashes,3,000 optimizer steps and the validation-only decision are verified. These predictions share data and weights; they are not independent training replications.

Recorded loops total279.39seconds, including periodic training/validation scoring, excluding setup, final evaluation and audit. This is not a speed benchmark. All attempts stayed finite. Stop here without extra seeds, training extension or an architecture sweep. Previous runs remain valid for final-only supervision; no general mechanism, convergence or architecture superiority is established.

[Raw results](../../../results/state_supervision.json), [independent audit](../../../results/state_supervision_audit.json), [frozen decision](../../../results/state_supervision_decision.json), [full scores](../../../results/state_supervision_scores.pt), [runner](../../../experiments/state_supervision.py), [auditor](../../../experiments/audit_state_supervision.py). Existing state-tracking papers supply context, as recorded in the original protocol; this is a local intervention, not their replication.
