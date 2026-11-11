# What the state-tracking models respond to

Executed 2026-10-03. Saved checkpoints only. This diagnostic follows the [direction review](../../assessments/research-direction-review.md) and asks why recent resets work better than sustained parity. [Protocol](../../protocols/state-sensitivity.md) | [Source](../../../experiments/state_sensitivity.py) | [Raw evidence](../../../results/state_sensitivity.json) | [Every metric](../../../results/state_sensitivity_metrics.csv).

**The models respond to inputs but do not reliably respect state-update logic.** Literal initial-input neglect is insufficient as an explanation. Recent-position sensitivity appears in some comparisons but does not explain every seed or length. The optimization or representation reason remains open.

## Learn the distinction

For KEEP/FLIP sequences, the answer is `initial bit XOR (number of FLIPs mod 2)`. Changing the initial bit or one relevant KEEP/FLIP must invert it. Changing two operations after the last SET must preserve it. A later SET erases earlier changes, so those should have no effect. We check these label relations independently of the model.

We toggle KEEP to FLIP or FLIP to KEEP; SETs stay fixed. We compare first/middle/last eligible operations in the same input. In mixed sequences, responses before and after the last SET are reported separately. A middle eligible operation need not be the physical midpoint. A two-toggle control is skipped if fewer than two eligible suffix operations exist.

A changed class is not necessarily a correct answer. **Both-correct** requires correctness before and after the intervention. Probability change measures sensitivity even when the highest-scoring class stays fixed. These are input interventions, not identified internal circuits.

## Required changes often fail

Twelve-operation training inputs, 512 examples per row. Values are predicted class-change percentages. First/last toggles and the reset-free initial-bit change must change the true label. The final column changes two suffix operations, which must preserve the true label.

| Seed | Group | Initial-bit change | First toggle | Last toggle | Two-toggle class changes |
|---|---|---:|---:|---:|---:|
| 2003 | reset_free | 0.00% | 0.00% | 0.00% | 0.00% |
| 2003 | long_tail | Label unchanged | 0.39% | 0.20% | 0.59% |
| 2011 | reset_free | 10.16% | 7.03% | 10.55% | 9.96% |
| 2011 | long_tail | Label unchanged | 5.86% | 17.38% | 16.41% |
| 2027 | reset_free | 10.16% | 31.64% | 30.66% | 42.97% |
| 2027 | long_tail | Label unchanged | 16.99% | 20.31% | 20.70% |

For seed 2003 without resets, initial-bit changes alter probabilities by an average .01058, but change no class predictions. Both answers are therefore never correct together. A constant class can pass the two-toggle invariance check while failing the task; invariance alone is insufficient. In long-tail sequences, changing the irrelevant initial bit still changes classes by .39%/31.64%/11.52% across seeds. The reader is neither universally insensitive nor consistently reset-invariant.

## Strong responses depend on the context

Mean absolute change in probability of class 1 for a relevant final toggle on twelve-operation training inputs:

| Seed | No resets, n=512 | Initial reset and long tail, n=512 | Mixed resets, n=246 |
|---|---:|---:|---:|
| 2003 | 0.02468 | 0.02241 | 0.89573 |
| 2011 | 0.07925 | 0.09873 | 0.67446 |
| 2027 | 0.04589 | 0.03120 | 0.85288 |

Mixed-reset relevant final toggles change classes in 92.68%/73.98%/88.21% of cases; both-correct rates are 91.87%/69.11%/87.40%. That is substantially more responsive than the sustained-update groups, but remains imperfect. The mixed group has a shorter relevant suffix and different token content; this comparison does not isolate suffix length as the cause. The other mixed examples have a later SET that erases the selected toggle, and are analyzed separately.

Last-toggle probability sensitivity exceeds first-toggle sensitivity for all three seeds in both length-12 training sustained-update groups. This is not universal: seed 2027 at test length 24 without resets responds more to the first toggle (.05989) than the last (.04060). It also changes classes on 55.66% of two-toggle unchanged-parity controls; both-correct is only 22.27%. A simple reliable recent-token rule is not established.

## Count-stratified evidence

The raw report includes all FLIP-count bins with denominators. Seed 2003 has exactly 50% twelve-operation no-reset training accuracy in every observed count bin. Other seeds vary without a demonstrated consistent parity rule. Sparse tails can contain only two examples; a 100% tail score would be weak evidence. These complement-balanced bins do not identify a counting mechanism or a cause of failure.

## Scope and verification

All original train 8/12 and test 8/12/24 pools were reused, across three groups and three checkpoints. There are 13,248 original evaluations and 62,814 intervened evaluations, totaling 76,062. These are three trained models and dependent derived inputs, not 76,062 independent experiments. Reset-free initial-bit flips are already-present complement partners in their pools. Other derivatives can overlap existing inputs. No newly reserved confirmation set is claimed.

All labels match sequential and last-SET/parity solvers. The intentionally wrong always-flips-label rule fails on 6,530 generated examples. Native outputs match the previously verified independent forward within 1.02e-13, with zero argmax differences. Originals match archived scores exactly. Scalar recomputation of all 261 metric rows differs by at most 1.26e-13. The metadata records actual positions and skipped two-toggle cases. Targeted original data/source/report/checkpoint hashes are preserved. CPU float64, one thread, Python 3.14.7, PyTorch 2.11.0+cu128. No training or gradients were updated.

## What this changes in the investigation

| Hypothesis | What the evidence permits |
|---|---|
| The model completely ignores the initial bit | Rejected as a literal function claim: probabilities change, while logical class changes remain unreliable |
| Failure is just a lack of initial-state sensitivity | Insufficient: even relevant FLIP changes in the long-tail group rarely receive the required response |
| A universal recent-token shortcut explains failure | Unsupported: position sensitivity is seed/length/context dependent and parity-preserving controls fail |
| The model learned a complete parity/reset algorithm | Rejected for these checkpoints and inputs |
| Optimization, exposure or capacity caused the failure | Still unisolated; input responses do not distinguish these causes |

[Liu et al., author explanation of Transformers Learn Shortcuts to Automata](https://clarabing.github.io/shortcut_automata/) distinguishes parallel automaton computation from recurrent updates and uses state-sequence learning. It motivates candidate mechanisms, but this final-answer diagnostic neither reproduces that setup nor identifies a learned shortcut.

The next useful decision is a small training-only recipe-development phase for this same question. Select candidates by full training-group loss and fit, with a fixed search budget; record every attempt. Lock a recipe before fresh confirmation. This addresses learning adequacy without rewriting the original fixed-budget failure. A suitable development question is whether improved optimization/exposure can fit sustained updates in the existing architecture. No claim about eventual superiority follows, and no such sweep has run here.
