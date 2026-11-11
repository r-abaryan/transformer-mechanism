# Graph composition: both recipes fail short fitting

Executed and closed, 2026-10-02. All six fixed-budget runs completed. Neither recipe mastered the one-step control or the two-step task. The GRU has a modest relative advantage; the predeclared ten-point paired advantage rule fails.

[Educational explanation and diagram](recurrent-composition-guide.md). [Frozen protocol](../../protocols/recurrent-composition.md).

## Methods

Each input supplies a shuffled directed cycle, a query node and a one/two-step marker. The label is `f(q)` or `f(f(q))`. A separate dictionary walk checks labels; the producer indexes a cycle.

| Item | Executed setup |
|---|---|
| Data |416 mapping families;104,448 expanded examples. At4/6 edges:128 training,32 validation and32 test families each. At8:32 test families. Full-pool evaluation, not a training subset |
| Splits/controls | Families include all12 cyclic symbol renamings, two reverse-related edge orders, every node query and both hop counts. No family crosses pools. All answers are balanced across12 symbols; both required-edge orders are balanced |
| Support | Enumerated251 four-edge families modulo cyclic renaming;192 used. All graphs share cycle topology. Individual edges and node symbols can recur across pools |
| Models | Transformer: two width16 MQA blocks, RoPE, LayerNorm, ReLU32;4,416 parameters. GRU: embedding16, one hidden28 layer, zero state each sequence;4,524 parameters,2.45% more |
| Training | Seeds2237/2243/2251; shared initial embedding values and sampled indices within each pair. Other initialization differs. CPU float64, torch2.11.0+cu128, deterministic, one thread |
| Budget/objective |6,000 Adam updates, lr0.001, batch64, alternating4/6 edges,32 examples per hop each batch. Final-position15-class cross-entropy only;384,000 targets/run. Equal examples/updates do not imply equal compute |
| Evaluation | Full train/validation at5,000 and6,000 updates. All six final checkpoints and short-pool results saved before test evaluation. No checkpoint selection, tuning or extension |

## Mean findings

Accuracy in percent, averaged across three training seeds and rounded to two decimals (half-up). Decisions use unrounded scores. Loss is mean cross-entropy over both hop groups, in nats. Expanded examples and renamings are correlated variants, not independent training replications.

| Pool | Transformer one | Transformer two | GRU one | GRU two | Transformer loss | GRU loss |
|---|---:|---:|---:|---:|---:|---:|
| train 4 | 27.56 | 27.63 | 33.50 | 34.23 | 1.4104 | 1.2111 |
| train 6 | 17.26 | 17.42 | 20.58 | 20.10 | 1.8363 | 1.7294 |
| validation 4 | 27.21 | 27.59 | 33.96 | 33.12 | 1.4129 | 1.2161 |
| validation 6 | 17.43 | 17.27 | 20.25 | 20.79 | 1.8371 | 1.7348 |
| test 4 | 27.16 | 28.20 | 33.18 | 32.42 | 1.4125 | 1.2176 |
| test 6 | 17.17 | 17.52 | 20.37 | 19.59 | 1.8393 | 1.7358 |
| test 8 | 12.48 | 12.84 | 14.51 | 14.26 | 2.1795 | 2.1425 |

The one-step control also fails on the training pools. Therefore these results do not isolate a two-step composition difficulty or a pure length-transfer failure. Test8 adds distractors and changes positional exposure while keeping the hop count fixed.

## Per-seed accuracy

One-step control:

| Seed | Model | Train4 | Train6 | Val4 | Val6 | Test4 | Test6 | Test8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2237 | Transformer | 27.82 | 17.74 | 28.22 | 18.06 | 27.47 | 17.64 | 13.02 |
| 2237 | GRU | 33.66 | 20.86 | 34.21 | 20.99 | 34.02 | 20.51 | 14.32 |
| 2243 | Transformer | 28.67 | 17.44 | 27.83 | 17.34 | 28.26 | 17.45 | 12.19 |
| 2243 | GRU | 33.40 | 20.49 | 34.44 | 20.72 | 32.10 | 20.07 | 14.49 |
| 2251 | Transformer | 26.18 | 16.60 | 25.59 | 16.88 | 25.75 | 16.43 | 12.22 |
| 2251 | GRU | 33.44 | 20.39 | 33.24 | 19.05 | 33.43 | 20.53 | 14.73 |

Two-step task:

| Seed | Model | Train4 | Train6 | Val4 | Val6 | Test4 | Test6 | Test8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2237 | Transformer | 28.79 | 18.00 | 28.13 | 18.36 | 29.59 | 17.88 | 13.13 |
| 2237 | GRU | 34.77 | 19.93 | 32.75 | 20.01 | 32.62 | 19.44 | 14.11 |
| 2243 | Transformer | 28.01 | 17.52 | 28.29 | 17.49 | 28.35 | 17.84 | 13.10 |
| 2243 | GRU | 33.94 | 20.27 | 33.27 | 21.16 | 32.26 | 19.47 | 14.42 |
| 2251 | Transformer | 26.10 | 16.74 | 26.37 | 15.97 | 26.66 | 16.84 | 12.29 |
| 2251 | GRU | 33.98 | 20.11 | 33.33 | 21.20 | 32.39 | 19.86 | 14.26 |

## Required-edge order

“Before” means the edge from the query appears before the edge from its intermediate successor. These are descriptive group scores, not evidence of a learned procedure.

| Pool | Transformer before | Transformer after | GRU before | GRU after |
|---|---:|---:|---:|---:|
| train 4 | 25.68 | 29.59 | 32.85 | 35.61 |
| train 6 | 16.51 | 18.32 | 21.15 | 19.06 |
| validation 4 | 24.98 | 30.21 | 32.68 | 33.55 |
| validation 6 | 15.83 | 18.72 | 22.28 | 19.30 |
| test 4 | 25.61 | 30.79 | 31.42 | 33.42 |
| test 6 | 16.59 | 18.45 | 20.93 | 18.24 |
| test 8 | 13.63 | 12.06 | 15.90 | 12.63 |

Every per-seed group accuracy, count and loss is retained in the [raw report](../../../results/recurrent_composition.json).

## Baselines and frozen decision

| Baseline | One-step accuracy | Two-step accuracy | Meaning |
|---|---:|---:|---|
| Exact graph walk |100% |100% | The task has an unambiguous exact answer |
| Always one-step successor |100% |0% | Ordinary lookup cannot solve the two-step labels |
| Return final listed destination |25% /16.67% /12.50% |25% /16.67% /12.50% | For4/6/8 edges; ignores the query |
| Always0 |8.33% |8.33% | Verified balanced labels |
| Uniform12-symbol choice |8.33% expected |8.33% expected | Analytical expectation |
| Return query node |0% |0% | Neither one nor two steps returns to the start in these cycles |

Similarity to a baseline score does not establish that a model uses its rule.

Primary comparison: two-step validation accuracy, with4/6-edge scores equally weighted. A consistent advantage requires at least10 percentage points in every pair.

| Seed | Transformer | GRU | GRU minus Transformer, points |
|---|---:|---:|---:|
| 2237 | 23.24 | 26.38 | +3.14 |
| 2243 | 22.89 | 27.21 | +4.32 |
| 2251 | 21.17 | 27.27 | +6.10 |

Both short-fit99% gates fail. Both reliability rules fail. All paired GRU gains are positive, but each is below10 points; the consistent-advantage rule fails. This does not prove equal quality or universal architectural inability.

## Training limit

Final validation loss minus the5,000-update value:

| Seed | Model | Four edges | Six edges |
|---|---|---:|---:|
| 2237 | Transformer | -0.026025 | -0.021689 |
| 2237 | GRU | -0.048199 | -0.043879 |
| 2243 | Transformer | -0.013874 | -0.017908 |
| 2243 | GRU | -0.030730 | -0.015010 |
| 2251 | Transformer | -0.029479 | -0.011429 |
| 2251 | GRU | -0.027679 | -0.009890 |

Every final interval improves. These measurements do not establish convergence or the ranking under longer training. The study ends at the frozen budget.

## Verification and interpretation

All104,448 labels and family/balance controls pass. Intentional one-step-for-two-step labels fail on all52,224 two-step cases. All six initial output/gradient comparisons pass; maximum gradient difference1.11e-16. Prefix, changed-future, batch-isolation and zero-state checks pass; resetting GRU state every token is detected in every GRU seed.

The [independent final audit](../../../results/recurrent_composition_audit.json) passes all42 pools and626,688 predictions: maximum logit difference2.132e-14, maximum metric difference4.441e-16, zero argmax changes. Checkpoint steps, paired streams, pretest snapshot,165 source/protocol hashes and11 earlier report hashes pass. The decision is independently rederived and an intentional wrong decision is detected. These checks support the measured implementation within tolerance; they do not prove every possible behavior correct.

**Observation:** both recipes have low training and held-out accuracy. **Interpretation:** this task was not mastered under the fixed setup. **Unknown:** whether optimization, capacity, exposure, input encoding or objective choices account for the weakness, and whether either model learned a reusable lookup mechanism.

The [guide's paper comparison](recurrent-composition-guide.md#relation-to-papers) separates our in-context graph task from facts learned in weights and from theoretical compositional-question results. This is neither a paper replication nor a refutation.

The fixed study is closed without tuning. [Cross-task consolidation](../../assessments/capability-summary.md) summarizes the completed comparisons. No further experiment is selected.

To audit saved artifacts: `.venv-gpu/Scripts/python.exe experiments/audit_recurrent_composition.py`. The training script refuses to overwrite existing attempts; a fresh execution needs an isolated workspace.
