# Training on varied positions

Varied training largely removes errors at trained offsets, but does not establish robustness beyond them. All ten models were trained and audited. The [protocol](../../position-training-protocol.md) fixed the comparison and thresholds before training.

## What this changes

A position offset moves every position number together. Tokens, order, pair distances and answers stay the same. Fixed training always starts at position 0. Varied training chooses a start from 0 through 4 for each minibatch. Evaluation uses ordinary attention, without forced routes.

This tests whether training exposure can reduce the position sensitivity seen in the [previous cycle](../training/reserved-cycle-results.md). The model architecture stays unchanged.

## Math in brief

For token t at index i, the model receives x_i = E(t_i) + PE(i+s). Here E is the learned token embedding and s is the offset. With width d=16:

PE(p,2j)=sin(p/10000^(2j/d)); PE(p,2j+1)=cos(p/10000^(2j/d)).

Fixed training minimizes average negative log probability of the correct answer with s=0. Varied training samples s uniformly from {0,1,2,3,4} and minimizes the same loss. One sampled offset is shared by the 64 examples in that update. This approximates an average over offsets; it does not impose exact invariance.

Distances remain (i+s)-(j+s)=i-j. Yet adding PE to token embeddings changes the inputs to normalization, Q/K/V projections and residual branches. Preserving distance alone therefore does not guarantee identical predictions.

## Matched method

Five matched seeds, two groups, 5,088 parameters each. Same initial weights and exact minibatch index stream within each pair. Each run uses Adam .001 for 3,000 updates of 64 examples, CPU float64. Only position offsets differ between groups. Validation uses offset 0; it never selects checkpoints or settings.

Reserve 1,600/400/400 whole mappings for training/validation/test. All are absent from every split of the previous cycle. Each mapping has 18 order/query variants, giving 28,800/7,200/7,200 sequences. Values are distinct, so duplicate correct answers cannot hide wrong-pair retrieval. Test predictions begin after all ten checkpoints are saved.

## How to read the results

Accuracy counts correct answers. Cross-entropy measures how much probability the model assigns to the answer; lower is better. Mapping correctness requires all 18 variants of a mapping to be correct. It is stricter than sequence accuracy.

Offsets 0..4 are familiar offset settings on unseen mappings. Offsets 6/8/12 test transfer outside that range. Some absolute token positions overlap training positions even at an unseen offset. Every input still has eight tokens. This is not a longer-sequence test.

## Results

Each entry below is a five-seed mean. Loss is in nats. The last column requires every order/query variant of a mapping to succeed.

| Offset | Fixed accuracy | Varied accuracy | Fixed loss | Varied loss | Varied mapping correctness |
|---|---:|---:|---:|---:|---:|
| 0 | 99.997% | 99.981% | .0010 | .0018 | 99.65% |
| +1 | 91.150% | 99.983% | .4931 | .0013 | 99.70% |
| +2 | 72.214% | 99.997% | 1.5684 | .0010 | 99.95% |
| +3 | 64.144% | 99.997% | 1.9606 | .0011 | 99.95% |
| +4 | 56.917% | 99.975% | 2.2184 | .0018 | 99.60% |
| +6 | 86.397% | 93.650% | .6046 | .3229 | 36.80% |
| +8 | 59.503% | 77.650% | 2.0779 | 1.1138 | 7.30% |
| +12 | 60.422% | 60.381% | 1.8738 | 1.9103 | 1.05% |

Per-seed accuracy (%). Both groups use the same test examples.

| Seed/group | 0 | +1 | +2 | +3 | +4 | +6 | +8 | +12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 461 fixed | 100 | 90.44 | 75.75 | 66.65 | 67.56 | 85.97 | 59.88 | 67.25 |
| 461 varied | 100 | 99.97 | 100 | 99.99 | 99.92 | 97.00 | 89.50 | 72.69 |
| 487 fixed | 100 | 93.42 | 65.89 | 58.78 | 36.03 | 85.63 | 63.97 | 69.85 |
| 487 varied | 99.93 | 99.94 | 99.99 | 100 | 99.97 | 97.33 | 84.96 | 76.79 |
| 509 fixed | 100 | 88.94 | 70.01 | 67.58 | 63.82 | 90.33 | 49.24 | 52.88 |
| 509 varied | 100 | 100 | 100 | 100 | 100 | 94.08 | 81.79 | 53.26 |
| 541 fixed | 100 | 97.51 | 82.14 | 76.63 | 75.82 | 81.19 | 57.83 | 44.86 |
| 541 varied | 100 | 100 | 100 | 100 | 100 | 87.06 | 61.81 | 49.31 |
| 571 fixed | 99.99 | 85.43 | 67.28 | 51.08 | 41.36 | 88.86 | 66.60 | 67.28 |
| 571 varied | 99.97 | 100 | 100 | 100 | 99.99 | 92.78 | 70.19 | 49.85 |

| Predeclared criterion | Outcome |
|---|---|
| Every varied seed >=90% at all trained offsets | Passed |
| Offset-0 loss <=1 point against matched fixed, every seed | Passed; worst loss .0694 point |
| Every varied seed >=90% at +6/+8/+12, separately | Failed at each; +6 fails seed 541, +8/+12 fail all five |
| Mean gain >=5 points at +6/+8/+12 | Passed +6 (+7.253), +8 (+18.147); failed +12 (-.042) |
| Outside-range conjunction | Failed |

## What we learned

Observed: changing training positions alone substantially improves accuracy on unseen mappings at familiar offsets. The matched groups isolate this training change within our implementation and budget. Attention forcing is unnecessary for these gains.

Observed: transfer is limited. Seed 571 loses 17.43 points relative to its fixed counterpart at +12. Positive averages at +6/+8 do not mean every seed is reliable. At +6, 93.65% mean sequence accuracy coexists with only 36.80% fully correct mappings.

Supported interpretation: position sensitivity is partly addressable through training exposure without changing the architecture. Unknown: whether the model learned a transferable relative-position strategy, which internal computation changed, and whether gains survive longer sequences. This experiment cannot separate those mechanisms.

Follow-up completed: [routing inspection and QK/V replacements](position-routing-results.md). Matched QK rescues varied +8 accuracy to 99.44%, but slot-aligned unrelated QK also gives 98.02%. The next control must separate target-slot alignment from content matching. These findings are exploratory on opened data; reserve new mappings before confirmation. Do not simply widen the training range until scores pass.

## Paper comparison and limits

[Ruoss et al., ACL 2023](https://aclanthology.org/2023.acl-short.161/) study randomized positional encodings using ordered subsets of a larger position range. They report length-generalization gains across algorithmic tasks. Our global-offset experiment preserves every gap and keeps sequence length fixed. It tests a narrower question and does not reproduce their method or benchmark. The paper's abstract and method description were checked; a full implementation audit remains absent.

Five seeds and one task family cannot establish general superiority. Mapping variants are dependent; there are 400 test mapping units, not 7,200 independent mappings. Training compute is matched by updates and examples, not measured wall-clock time. Better predictions alone would not identify which learned circuit changed.

## Evidence

[Training/evaluation code](../../../experiments/position_training.py), [reserved splits](../../../results/position_training_splits.json), [raw results](../../../results/position_training.json), and [audit code](../../../experiments/audit_position_training.py). The report records source/protocol hashes, matched initialization/minibatch hashes, offset counts, curves and final checkpoint hashes.

The [executed audit](../../../results/position_training_audit.json) reloaded all ten checkpoints and reproduced all 80 offset conditions using explicit branch execution, rather than the evaluation embedding hook. Maximum accuracy/loss/mapping-metric discrepancy was 1.33e-15. Split isolation, complete mapping variants, labels, balance, paired hashes and criterion arithmetic passed. Before training, forward and gradient parity errors were 3.33e-16 and 1.11e-16. All training losses/gradients were finite. These checks share PyTorch and existing helpers; they are not an independent framework replication.
