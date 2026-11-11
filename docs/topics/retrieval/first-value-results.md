# Does the omitted first value position explain the failure?

**Including it does not fix the failure.** All-three paired-key forcing averages 93.31% at shift +4. Seed 223 reaches 81.09%, below the predeclared 90% criterion.

## Short explanation

The previous intervention changed value positions 3 and 5. Position 1 was untouched because it has no earlier nonpaired key for that control. Here we force position 1 to read its paired key at position 0, or itself. We test both choices with later positions unchanged and with later positions already paired-forced.

Every downstream operation runs normally. No compact activations are copied and no weights are trained. Forcing sets an attention row to one-hot: its head output becomes the chosen source's V before output projection. The intervention uses the task's known alternating layout.

## Method and results

[Protocol](../../first-value-protocol.md): five frozen models, the same 5,120 inspected crops each, both heads, shifts 0/+4. Every crop has three pairs and eight tokens. Mean accuracy / cross-entropy in nats:

| Intervention | Compact | Shift +4 |
|---|---:|---:|
| Original | 99.996% / 0.0005 | 60.60% / 2.2016 |
| Later two values read paired keys | 99.988% / 0.0008 | 93.48% / 0.3065 |
| First value reads paired key only | 99.91% / 0.0024 | 60.57% / 2.2068 |
| First value reads self only | 78.91% / 1.5280 | 44.55% / 3.8867 |
| All three values read paired keys | 99.91% / 0.0024 | 93.31% / 0.3246 |
| First reads self, later two read paired keys | 79.42% / 1.4566 | 72.33% / 1.8157 |

| Seed | Shifted: later two paired | Shifted: all three paired | Increment, percentage points |
|---|---:|---:|---:|
| 137 | 98.01% | 98.26% | +0.25 |
| 163 | 99.34% | 99.47% | +0.14 |
| 191 | 93.79% | 93.73% | -0.06 |
| 223 | 82.79% | 81.09% | -1.70 |
| 257 | 93.46% | 93.98% | +0.53 |

The every-seed shifted 90% criterion **failed**. Compact preservation within one percentage point **passed**; the largest compact loss was 0.176 point. Means alone would hide the failed seed.

First-position effects depend on the background. Adding paired-key forcing changes mean accuracy by -0.04 point with later rows unchanged, or -0.17 point with later rows paired-forced. Adding self forcing changes those means by -16.05 and -21.15 points respectively. Forcing self is harmful; this does not imply that forcing the paired source improves an already useful learned mixture.

For seed 223, queried-pair-rank accuracies under later-two forcing were 85.53%, 82.55%, 80.11%. With all-three forcing they were 79.22%, 83.57%, 80.65%. Failure was not confined to queries about the first pair. These subgroups were predeclared descriptive checks, not independent trials.

## Conclusion and limits

The untouched first row is not a sufficient explanation for the remaining shifted failure. Expanding forcing is not a reliable improvement here. This does not show the first row is irrelevant: self forcing clearly harms performance. The mechanism can involve later routing or interactions that remain untested.

This is a bounded negative result on reused crops, one nonzero shift and known task layout. It does not establish behavior on full long sequences or language tasks. The [attention definitions and source comparison](../attention/qkv-results.md#learn-the-operation) remain applicable. No new architecture is proposed.

## Verification and next step

[Script](../../../experiments/first_value.py) and [raw report](../../../results/first_value.json) preserve all 12 conditions per seed, per-query-rank accuracy and paired rescue/harm counts. Original/later-two baselines reproduced within 1e-10. Maximum sham logit error was 6.27e-14. Labels, finite outputs, causal zeros, normalized rows and unchanged checkpoints passed. Source hashes and paired-count arithmetic were checked separately.

Follow-up completed: [fixed intervention on fresh parents across shifts](pair-confirmation-results.md). Gains persist, but the all-seed criterion fails at +4/+6/+8. Crop overlap with earlier data is disclosed. Do not repeatedly expand the intervention until the weakest seed passes.
