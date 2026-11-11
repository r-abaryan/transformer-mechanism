# Depth replication and length transfer

New training runs test whether the depth advantage survives regenerated data and new random seeds. Longer sequences test a different question: whether the learned solution transfers beyond three pairs. The [predeclared protocol](../../depth-confirmation-protocol.md) fixes both criteria separately.

## Method and scope

Train five two-block models (4,896 parameters) and five wider one-block controls (5,152 parameters), using seeds 137/163/191/223/257. Keep Adam 0.001, 3,000 updates, batch 64 and CPU float64 unchanged. Reuse verified implementations; match minibatch indices within each seed. Final checkpoints only, no tuning. See [earlier equations and architecture comparison](depth-comparison-results.md#method-and-essential-math).

Each input is `[k1,v1,...,kn,vn,marker,query]`; the label is the queried key's value. Minimize \(L=-B^{-1}\sum_b\log p_\theta(v_b\mid x_b)\). Train only at n=3 (8 tokens); evaluate n=4 (10 tokens) and n=6 (14 tokens) without updating weights. Increasing n changes distractor count and absolute positions together, so a transfer failure does not isolate either cause.

## Results

Accuracy percentages; each cell is deep / wide. Every deep seed also scored 100% on training and validation; wide means were 56.75% / 55.20%.

| Seed | Three-pair test | Four pairs | Six pairs |
|---|---:|---:|---:|
| 137 | 100 / 53.12 | 81.45 / 52.34 | 68.75 / 46.78 |
| 163 | 100 / 54.10 | 79.98 / 51.27 | 62.89 / 41.99 |
| 191 | 100 / 54.20 | 80.47 / 51.07 | 64.36 / 47.85 |
| 223 | 100 / 54.30 | 79.39 / 47.85 | 54.88 / 47.07 |
| 257 | 100 / 55.08 | 80.86 / 51.46 | 70.31 / 43.16 |
| Mean | 100 / 54.16 | 80.43 / 50.80 | 64.24 / 45.37 |
| Frequency-value baseline | 54.30 | 52.64 | 49.12 |

The baseline predicts the most frequent supplied value (smallest value breaks ties), ignoring the query. Uniform guessing has expected accuracy 25%; exact lookup is 100% by construction.

| Predeclared criterion | Observation | Outcome |
|---|---|---|
| Every deep seed >=90% train and three-pair test | All 100% | Passed |
| Mean deep-minus-wide test advantage >=10 points | 45.84 points; sample SD across paired seeds 0.70 point | Passed |
| Every deep seed >=90% at four pairs | Range 79.39%-81.45% | Failed |
| Every deep seed >=90% at six pairs | Range 54.88%-70.31% | Failed |

| Mean cross-entropy, nats | Three-pair test | Four pairs | Six pairs |
|---|---:|---:|---:|
| Deep | 0.000510 | 1.1740 | 2.2840 |
| Wide | 0.8075 | 1.0637 | 1.4428 |

Depth's same-length advantage replicated under the fixed recipe. Reliable length transfer did not. Deep models have higher longer-sequence accuracy but worse log loss than the wide controls: the metrics give different information about prediction quality. No calibration analysis was performed. Compute remains unmatched, and five seeds on one shared regenerated split do not measure variability across independently regenerated splits.

## What is held out?

Current train/validation/test sizes are 8,192/1,024/1,024. Canonical (mapping, query) identities are disjoint, including reordered copies. Different queries about a shared mapping remain allowed: 906 of the test set's 943 distinct mappings also occur in training. This tests held-out queries/mapping identities, not exclusively unseen mappings.

| New split | Overlap with original train | Original validation | Original test |
|---|---:|---:|---:|
| Train | 6,257 | 757 | 792 |
| Validation | 796 | 97 | 83 |
| Test | 761 | 106 | 103 |

The finite three-pair universe has \(\binom83 4^3\times3=10,752\) identities; earlier complement testing also examined the remainder. These are new runs with internally held-out data, not globally untouched cases or an independent reproduction. Both new longer-sequence sets have zero overlap with the original corresponding evaluation identities.

## Evidence and next investigation

[Raw report](../../../results/depth_confirmation.json), [exact split data](../../../results/depth_confirmation_splits.json), [training script](../../../experiments/confirm_depth.py) and [checkpoint audit](../../../results/depth_confirmation_audit.json) preserve results and provenance. Split isolation, labels, parameter counts, hashes, paired minibatches and criterion calculations passed. Reloading all ten checkpoints reproduced test/transfer metrics exactly (maximum difference 0). Forward/gradient/causal fixture errors were 0; every training loss and gradient was finite. This is a separate verification path using the same model/evaluation library, not an independent implementation.

The [original paper comparison](depth-comparison-results.md#paper-comparison-and-remaining-question) still applies: stacked attention is established architecture; these experiments neither reproduce translation benchmarks nor prove novelty. Accuracy alone does not establish a binding circuit.

Follow-up completed: [controlled position changes and distractor removal](../positions/position-distractor-results.md), with unchanged-input and sham checks for every seed. Layer/path explanations remain untested; do not alter the architecture to chase the transfer score.
