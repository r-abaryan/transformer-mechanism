# Repetition helps moderate loads, but leaves the 32-key gap

Executed 2026-10-04. **Four passes raise delta's mean 16-key accuracy from 90.45% to 97.92%. At 32 keys, accuracy stays near 58% and loss does not improve.** No-recency softmax remains 100% correct with numerically unchanged logits.

[Learning guide](history-repetition-guide.md) | [Frozen protocol](../../protocols/history-repetition.md) | [Source](../../../experiments/history_repetition.py) | [Report](../../../results/history_repetition/report.json) | [Every seed's accuracy and CE](../../../results/history_repetition/metrics.csv) | [Aggregates](../../../results/history_repetition/summary.json) | [Independent audit](../../../results/history_repetition/independent_audit.json).

## Method

Use the six saved [association-load checkpoints](association-load-results.md): softmax/delta, seeds 3323/3329/3331, 615 parameters each. Repeat each original 32-event test history as an ordered block 1/2/4 times, giving 32/64/128 events. Query, labels, key/value assignments and distinct-key load stay fixed. Each load has 384 examples per seed, organized as 64 contexts with six cyclic value renamings.

Zero training updates. CPU float64, one thread. The 72 seed/reader/load/factor cells contain 27,648 evaluated predictions; these are not independent training runs. Test pools were already opened. Lengths 64/128 are outside training support.

## Accuracy and loss

Each entry gives **three-seed mean accuracy % / cross-entropy**. Seeds and pools have equal counts.

| Distinct keys | Delta: 1 pass | Delta: 2 passes | Delta: 4 passes |
|---|---:|---:|---:|
| 2 | 100.00 / 0.000016 | 100.00 / 0.000016 | 100.00 / 0.000016 |
| 8 | 99.48 / 0.013528 | 100.00 / 0.000310 | 100.00 / 0.000019 |
| 16 | 90.45 / 0.356397 | 96.70 / 0.167173 | 97.92 / 0.067885 |
| 32 | 58.42 / 2.321675 | 57.90 / 2.387876 | 58.51 / 2.377796 |

Softmax: 100% for every seed/load/factor, with mean CE approximately 0.000279. Maximum repeated-versus-original logit difference is 7.11e-15, below the frozen 1e-9 tolerance.

At the highest load, the small mean changes hide different seed responses:

| Seed | Delta: 1 pass % | Delta: 2 passes % | Delta: 4 passes % |
|---|---:|---:|---:|
| 3323 | 61.20 | 60.16 | 60.68 |
| 3329 | 51.04 | 51.30 | 52.34 |
| 3331 | 63.02 | 62.24 | 62.50 |

The CSV retains all per-seed losses and lower-load scores.

## Which answers change?

Counts below compare each factor with the original pass, summed across three seeds: 1,152 paired examples per load. A rescue corrects an error; a harm loses a correct answer.

| Keys | 2 passes: rescues / harms | 4 passes: rescues / harms |
|---|---:|---:|
| 2 | 0 / 0 | 0 / 0 |
| 8 | 6 / 0 | 6 / 0 |
| 16 | 72 / 0 | 86 / 0 |
| 32 | 44 / 50 | 47 / 46 |

Softmax has zero rescues or harms throughout. Both delta factors pass the declared two-point preservation rule at loads 2/8. Both fail the all-seed 10-point gain and 95% accuracy rules at load 32. These are separate diagnostic criteria, not an adoption decision.

## What this establishes

More processing length can improve answers without adding associations. The improvement at 16 keys shows that the original errors there were sensitive to repeated writes. It does not establish that inadequate training exposure caused them.

At 32 keys, repeating the same block two or four times does not repair this saved reader. Rescues and harms coexist. This rules out a simple uniform benefit within the tested intervention; it does not prove a hard state-capacity bound or identify the cause of each error. The original 100% training fit and held-out gap remain valid.

The softmax identity and delta block equations are explained in the guide. Neither primary paper predicts these local scores. This compares compact readers with consistent assignments, not complete language models or conflicting latest-value retrieval.

## Verification and stopping

Four preflight methods pass, including actual initialized checkpoint reload/evaluation/audit and deliberate wrong-order, partial-duplication, label/scalar and decision controls. The audit separately constructs repeated lists, evaluates matrix-transition/explicit-exponential equations and reduces CE and paired categories. All 72 cells and 48 paired comparisons pass: maximum logit equation error 2.13e-14, zero argmax disagreements. Shared data, weights and PyTorch remain common dependencies.

Original-pass metrics match the archived report; all 186 protected files remain unchanged. The declared diagnostic is complete. No additional factors, training, wider-state search or architecture adoption follows automatically.
