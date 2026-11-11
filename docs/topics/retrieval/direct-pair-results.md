# Direct paired-key forcing: substantial but incomplete rescue

**Mean shifted accuracy rises to 93.48%, without compact donors. The every-seed 90% criterion fails.** Seed 223 reaches only 82.79%.

## Short explanation

Previously, the test supplied compact final Q and constructed selected downstream keys. This test removes those helpers. We force two block-1 value positions to read their paired keys. Then every later operation runs normally on the shifted input.

No compact activations are copied. No weights are trained. A compact run is evaluated separately to check for harm. The intervention still uses the known alternating key/value layout, so it is not a general-purpose architectural solution.

## Method

The [fixed protocol](../../direct-pair-protocol.md) uses five frozen models and the same 5,120 inspected six-derived crops per model. Each crop has three pairs and eight tokens. Test global shifts 0 and +4. At value rows 3/5 in both block-1 heads, set attention to one source: paired key at row-1, self at row, or earlier nonpaired key at row-3. Leave row 1 unchanged for the same causal control as before.

If A is attention probability and V its current value matrix, forcing source j makes the selected head output equal V_j before output projection. Recompute block-1 FFN and all block-2 computations normally. There are no donor Q/K/V values, saved FFN outputs or selective downstream K replacements.

## Results

Mean accuracy / cross-entropy in nats across five seeds.

| Block-1 intervention | Compact input | Shift +4 |
|---|---:|---:|
| None | 99.996% / 0.0005 | 60.60% / 2.2016 |
| Force paired key | 99.988% / 0.0008 | 93.48% / 0.3065 |
| Force self | 62.37% / 2.7805 | 63.18% / 2.5298 |
| Force earlier nonpaired key | 44.27% / 4.0303 | 47.33% / 3.6058 |

| Seed | Shifted baseline | Shifted paired-key forcing | Compact paired-key forcing |
|---|---:|---:|---:|
| 137 | 65.74% | 98.01% | 100% |
| 163 | 58.61% | 99.34% | 100% |
| 191 | 61.07% | 93.79% | 100% |
| 223 | 54.98% | 82.79% | 99.94% |
| 257 | 62.60% | 93.46% | 100% |

| Predeclared criterion | Outcome |
|---|---|
| Every seed >=90% after shifted paired-key forcing | Failed: seed 223 |
| Mean advantage over each shifted source control >=10 points | Passed: +30.30 points versus self, +46.15 versus other key |
| Compact accuracy loss <=1 point in every seed | Passed; largest loss 0.059 point |

Across 25,600 model/crop evaluations, shifted forcing rescued 8,921 original errors but harmed 505 correct predictions. Compact forcing rescued one and harmed three. These evaluations share crops and parent episodes; they are not independent trials. Passing the compact tolerance does not mean zero harm.

## What this changes

Paired-key forcing helps a complete shifted forward pass, not only an artificial downstream key construction. This supports its behavioral relevance. It does not fully repair shift sensitivity, and the earlier controlled rescue cannot be generalized to reliable whole-model performance.

Remaining errors could involve the untouched first value row, later query/routing representations or interactions. None is established by these results. Only two rows and one shift were tested. The data are reused; full long sequences and new distributions remain untested here. This intervention uses task structure and does not establish a novel architecture or language-model improvement. See [attention definitions](../attention/qkv-results.md#learn-the-operation) for the underlying equations and source comparison.

## Verification and next question

[Script](../../../experiments/direct_pair.py) and [raw results](../../../results/direct_pair.json) preserve all eight conditions per seed. Original attention/full-forward reconstruction matched within 6.27e-14. Shift baselines reproduced within 1e-10. Labels, causal zeros, softmax row sums, finite outputs and unchanged checkpoints passed. Source hashes and paired rescue/harm arithmetic were separately checked.

Follow-up completed: [including the first value row](first-value-results.md) failed to resolve the remaining failure. Freeze the original intervention and predeclare fresh-example/additional-shift confirmation before broader claims.
