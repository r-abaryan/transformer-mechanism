# New vocabulary and reserved whole-mapping evaluation

This cycle retrains five models on a new task distribution. The prior attention intervention remains fixed. The [protocol](../../reserved-cycle-protocol.md) was saved before generation and training; no test predictions were computed until all five final checkpoints existed.

## What changed, in plain language?

There are now twelve possible keys and six possible values. Every example has three distinct keys and three distinct values. A wrong pair's value is therefore a wrong answer. The query still asks for one of the three supplied mappings.

Whole mappings are assigned to training, validation or test. Reordering a test mapping or asking another question about it never puts it in training. This is stronger separation than holding out only a mapping/query combination.

| Split | Whole mappings | Ordered/query sequences |
|---|---:|---:|
| Training | 1,600 | 28,800 |
| Validation | 400 | 7,200 |
| Reserved test | 400 | 7,200 |

Each split balances answer labels and query ranks exactly. Orders/queries of one mapping remain dependent; 7,200 test sequences are 400 mapping units, not 7,200 independent mappings. The generator and task design share ideas with earlier research. “Reserved” describes evaluation discipline for this new cycle, not independent replication or a completely unrelated semantic task.

## Fixed method

Two pre-norm blocks, width 16, two heads, FFN width 32. Embedding/output vocabulary expands to 19 tokens, giving 5,088 parameters. Five seeds: 307/331/367/401/433. Adam 0.001, 3,000 updates, batch 64, CPU float64. Each run processes 192,000 examples. No tuning, early stopping, checkpoint selection or intervention during training. Only final checkpoints are evaluated.

The intervention forces block-1 attention at value rows 3/5 to read their immediately preceding keys. Both heads are affected. Row 1 remains unchanged. All downstream computation runs normally; no compact donor is supplied. The test compares ordinary execution and this same fixed rule at shifts 0/1/2/4/6/8. Moving all positions by s preserves distances: (i+s)-(j+s)=i-j. Inputs remain eight tokens long.

Vocabulary, value distribution, training coverage and split design changed together. Results cannot isolate one of those changes. There is no new architecture claim; the [attention equations and source comparison](../attention/qkv-results.md#learn-the-operation) remain applicable.

## Results

All runs reached 100% training accuracy. Final validation ranged from 99.972% to 100%. Original unshifted reserved-test accuracy ranged from 99.944% to 100%; the all-seed 90% learning criterion passed.

| Shift | Original accuracy | Fixed intervention | Intervention loss, nats | Intervention: every order/query correct |
|---|---:|---:|---:|---:|
| 0 | 99.986% | 99.49% | 0.0139 | 96.25% |
| +1 | 92.23% | 98.81% | 0.0346 | 89.60% |
| +2 | 72.70% | 95.70% | 0.1591 | 68.75% |
| +4 | 49.05% | 89.00% | 0.4760 | 42.70% |
| +6 | 77.37% | 86.35% | 0.5996 | 35.30% |
| +8 | 55.33% | 77.71% | 0.9198 | 17.15% |

All entries are five-seed means. The last column is the fraction of test mappings answered correctly for all 18 order/query variants, averaged across seeds.

| Seed, intervention | Shift 0 | +1 | +2 | +4 | +6 | +8 |
|---|---:|---:|---:|---:|---:|---:|
| 307 | 100% | 99.00% | 94.89% | 91.68% | 89.74% | 82.64% |
| 331 | 99.99% | 99.39% | 95.10% | 83.69% | 82.35% | 79.83% |
| 367 | 100% | 99.67% | 97.58% | 93.68% | 94.17% | 79.07% |
| 401 | 99.25% | 98.53% | 95.06% | 82.21% | 78.56% | 77.10% |
| 433 | 98.24% | 97.46% | 95.89% | 93.75% | 86.94% | 69.90% |

| Predeclared criterion | Outcome |
|---|---|
| Every seed original unshifted test >=90% | Passed |
| Every seed patched >=90%, separately by shift | Passed +1/+2; failed +4/+6/+8 |
| Mean patched gain >=5 points at each positive shift | Passed at all five shifts |
| Across-positive-shift conjunction | Failed |
| Compact loss <=1 point in every seed | Failed: seed 433 loses 1.71 points |

The last failure matters: forcing a useful-looking attention pattern can damage a model that already answers the ordinary task correctly. Seed 433 falls from 99.944% to 98.236% at shift 0. Mean compact loss of 0.49 point would hide that failure.

## Bounded conclusion

The new reserved cycle confirms a benefit from this fixed task-specific intervention, but also confirms its limits. Whole-mapping holdout and distinct values do not remove position sensitivity. Reliability fails at larger shifts, and compact preservation is no longer satisfied for every seed. The earlier partial rescue is not a generally reliable repair.

Do not merge results across task distributions as if they were one controlled ablation. This cycle changed several data conditions together. The new test set was reserved before training and opened after all checkpoints were fixed, but the architecture and intervention were motivated by earlier research. Independent implementation/reproduction and full long-sequence evaluation remain absent.

This is a useful stopping point for the forcing rule. Keep it as a diagnostic control, not a proposed production change. The next research decision should name a new bounded question and reserve its evaluation before testing; avoid further patches selected against these now-inspected test results.

## Evidence

[Training/evaluation script](../../../experiments/reserved_cycle.py), [exact reserved splits](../../../results/reserved_cycle_splits.json), [raw report](../../../results/reserved_cycle.json), and [checkpoint audit](../../../results/reserved_cycle_audit.json) preserve source/protocol hashes, learning curves, losses, per-seed test conditions, attention measures and checkpoints. The implementation's test phase follows the completed five-run training loop; test metrics did not select weights or recipes.

Parameter counts, mapping isolation, answer/rank balance and labels passed. Every training loss/gradient was finite. Explicit/library baseline and attention checks passed with maximum error 1.76e-13. Checkpoints remained unchanged. The separate audit verified source/split/protocol hashes, mapping separation and paired arithmetic, then reloaded all five checkpoints; unshifted test accuracy/loss reproduced exactly. These checks share PyTorch and do not constitute an independent framework implementation.
