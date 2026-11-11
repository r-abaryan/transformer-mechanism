# Copy stopping: content also fails

Executed and closed. Blocking early EOS does not recover reliable long copying. No training or weights changed.

## What we tested

EOS means end of sequence. Ordinary generation chooses the highest-scoring token and stops at EOS. We reused all three saved models and all eight original pools, including training and validation. The intervention forbids only EOS for the first n output positions, then restores it. The original n+3 limit remains. It supplies the source length, so this is a diagnostic, not a replacement benchmark.

**Content exact** means the first n generated symbols match the source. **Full exact** also requires EOS immediately afterward. A correct prefix can therefore fail full exact.

| Test length | Ordinary full exact | Blocked-EOS content exact | Blocked-EOS full exact |
|---|---:|---:|---:|
| 3 | 96.88% | 99.48% | 98.96% |
| 4 | 90.63% | 95.83% | 94.27% |
| 6 | 0.00% | 1.56% | 0.52% |
| 8 | 0.00% | 0.00% | 0.00% |

All training sequences remain exactly correct. At six symbols, content is correct in only6/384 evaluations; full output in2/384. At eight, both are0/384. These counts reuse128 examples across three independently trained seeds; they are not384 independent training runs.

## Position and score explanation

Teacher forcing supplies the correct preceding symbols. It tests next-token prediction without accumulated generated-prefix errors. For position j, the EOS margin is:

`m_j = z_j(EOS) - z_j(correct symbol)`

Softmax gives `p(EOS)/p(correct) = exp(m_j)`. A positive margin favors EOS over that symbol, but another token could still rank first. A mean margin is not a percentage of examples. Terminal EOS is excluded below because its margin against itself is zero.

| Length | Position | Teacher-forced correct | Mean EOS margin |
|---|---:|---:|---:|
| 6 | 1 | 100.00% | -17.60 |
| 6 | 2 | 98.70% | -14.91 |
| 6 | 3 | 92.45% | -15.84 |
| 6 | 4 | 84.64% | -11.77 |
| 6 | 5 | 1.04% | 14.61 |
| 6 | 6 | 1.56% | 13.31 |
| 8 | 1 | 66.93% | -16.00 |
| 8 | 2 | 54.17% | -11.30 |
| 8 | 3 | 84.64% | -14.70 |
| 8 | 4 | 87.76% | -11.70 |
| 8 | 5 | 0.78% | 14.12 |
| 8 | 6 | 0.78% | 13.52 |
| 8 | 7 | 3.91% | 6.59 |
| 8 | 8 | 9.38% | 0.80 |

At six symbols, prediction drops sharply at position5, beyond the longest trained content length4. This alignment is an observation, not proof of a learned length rule. Eight-symbol inputs also fail at earlier positions. Correct prefixes therefore do not eliminate all errors, and preventing stopping does not restore the missing content.

## Every seed and group

Entries are blocked-EOS **content exact / full exact**, percentages. D means all symbols distinct; R means at least one repeats. Full position counts, EOS timing, ordinary results and margins for every group are in the raw report.

| Seed | Pool | D | R |
|---|---|---:|---:|
| 2371 | train_3 | 100.00 / 100.00 | 100.00 / 100.00 |
| 2371 | train_4 | 100.00 / 100.00 | 100.00 / 100.00 |
| 2371 | validation_3 | 100.00 / 100.00 | 96.88 / 93.75 |
| 2371 | validation_4 | 100.00 / 96.88 | 96.88 / 93.75 |
| 2371 | test_3 | 100.00 / 100.00 | 96.88 / 93.75 |
| 2371 | test_4 | 93.75 / 93.75 | 100.00 / 93.75 |
| 2371 | test_6 | 3.13 / 0.00 | 0.00 / 0.00 |
| 2371 | test_8 | 0.00 / 0.00 | 0.00 / 0.00 |
| 2377 | train_3 | 100.00 / 100.00 | 100.00 / 100.00 |
| 2377 | train_4 | 100.00 / 100.00 | 100.00 / 100.00 |
| 2377 | validation_3 | 100.00 / 100.00 | 96.88 / 96.88 |
| 2377 | validation_4 | 96.88 / 96.88 | 100.00 / 100.00 |
| 2377 | test_3 | 100.00 / 100.00 | 100.00 / 100.00 |
| 2377 | test_4 | 100.00 / 100.00 | 90.63 / 90.63 |
| 2377 | test_6 | 3.13 / 0.00 | 3.13 / 3.13 |
| 2377 | test_8 | 0.00 / 0.00 | 0.00 / 0.00 |
| 2381 | train_3 | 100.00 / 100.00 | 100.00 / 100.00 |
| 2381 | train_4 | 100.00 / 100.00 | 100.00 / 100.00 |
| 2381 | validation_3 | 100.00 / 100.00 | 100.00 / 100.00 |
| 2381 | validation_4 | 100.00 / 100.00 | 100.00 / 96.88 |
| 2381 | test_3 | 100.00 / 100.00 | 100.00 / 100.00 |
| 2381 | test_4 | 100.00 / 100.00 | 90.63 / 87.50 |
| 2381 | test_6 | 0.00 / 0.00 | 0.00 / 0.00 |
| 2381 | test_8 | 0.00 / 0.00 | 0.00 / 0.00 |

Only length3 passes the frozen95% content threshold in every seed/group. No test length passes the full-output threshold in every group. The length4 content mean exceeds95%, but weak groups fail. Original study decisions remain unchanged.

## Evidence and limits

Independent layer equations and a separate per-example decoder reproduce all15,798 intervention tokens across24 pools, with zero differences. Maximum teacher-logit error is1.15e-13, below1e-9. Separate tensor checks verify teacher correctness, margins and content/full counts. Scripted immediate-EOS and off-by-one suppression controls pass. Original checkpoints, saved scores, report, data, audit and source/protocol hashes remain unchanged.

The intervention is insufficient to repair long copying. It does not identify whether positions, learned representations, training exposure or another factor caused the errors. EOS suppression also changes later prefixes. These opened pools are descriptive evidence, not fresh confirmation. No architecture change, training extension or further diagnostic follows automatically.

[Protocol](../../protocols/copy-stopping.md) ? [Original study](encoder-decoder-copy-results.md) ? [Raw counts, outputs and audit](../../../results/copy_stopping.json) ? [Code](../../../experiments/copy_stopping.py)
