# Retrieval: Transformer wins this bounded comparison

Executed2026-10-02. Across three seed pairs, six-pair test accuracy averages **97.82% Transformer versus23.36% GRU**, weighting repeated and unique queries equally. The Transformer passes the frozen reliability rule; the GRU fails it. GRU training fit is already weak, so the gap is not purely length transfer.

[Short educational guide](recurrent-retrieval-guide.md) · [Frozen protocol](../../protocols/recurrent-retrieval.md).

## Method

Six fresh runs, seeds2131/2143/2161. Two-block MQA Transformer: width16, RoPE, LayerNorm, ReLU32,4,544 parameters. Single-layer GRU: embedding16, hidden27, bias-free19-class readout,4,462 parameters (1.80% fewer). Each pair shares initial embedding values and sampled indices; other initialization and architecture properties differ.

Both receive6,000 Adam0.001 updates, batch64, alternating3/4 pairs, CPU float64, one thread, PyTorch2.11.0+cu128. Each run processes384,000 final targets and3,456,000 input tokens. The objective is final-query cross-entropy only, unlike the intermediate supervision used in the state study. Equal updates are not equal compute. No tuning or checkpoint selection followed evaluation.

Reused MQA dataset, expanded to every distinct query key. Family counts at3 pairs:160/20/40 train/validation/test; at4:800/100/200; at6:200 test. Each family has36 reordered/value-renamed contexts and each context has n-1 queries. All170,640 expanded examples were label-checked. Splits remain family-separated; previously opened data are not fresh confirmation. Full training/validation results and checkpoint hashes were saved before test evaluation.

## Accuracy and loss

Means across three seeds. R/U means repeated-key/unique-key accuracy (%). Loss is ordinary mean cross-entropy across examples, so U has greater weight when there are more unique keys. The headline instead weights R/U equally.

| Pool | Transformer R/U | GRU R/U | Transformer loss | GRU loss |
|---|---:|---:|---:|---:|
| Train3 |100.00 /100.00|65.17 /34.59|0.0004|1.0136|
| Validation3 |100.00 /100.00|65.05 /33.89|0.0005|1.0191|
| Test3 |100.00 /99.98|65.65 /32.34|0.0006|1.0380|
| Train4 |99.97 /99.99|46.31 /25.20|0.0010|1.3671|
| Validation4 |99.96 /99.99|45.31 /25.44|0.0012|1.3699|
| Test4 |99.91 /99.98|44.83 /25.41|0.0018|1.3722|
| Test6 |98.69 /96.94|30.19 /16.53|0.1454|1.8693|

Per-seed detail prevents the mean from hiding a failed run:

| Seed | Model | Train4 R/U | Validation4 R/U | Test6 R/U |
|---|---|---:|---:|---:|
|2131|Transformer|100.00 /99.99|100.00 /100.00|99.56 /95.75|
|2131|GRU|47.07 /25.14|45.69 /25.39|31.56 /16.57|
|2143|Transformer|99.97 /99.99|99.97 /99.99|97.07 /96.39|
|2143|GRU|47.61 /25.05|46.33 /25.54|33.79 /16.33|
|2161|Transformer|99.95 /100.00|99.92 /99.97|99.46 /98.68|
|2161|GRU|44.25 /25.40|43.92 /25.40|25.21 /16.69|

Transformer six-pair gains are73.59,71.67 and78.12 percentage points, exceeding the frozen10pp margin in every pair. Every Transformer short-length cell exceeds99% and every six-pair query-type cell exceeds90%. Both-model short fitting fails because of the GRU. Full per-seed/group accuracy and loss are in the [raw report](../../../results/recurrent_retrieval.json).

## Shortcut controls

Six-pair test accuracy (%), R/U: constant12 and uniform-value expectation16.67/16.67; first matching value0/100; final context value34.83/16.29; exact last-match oracle100/100. Distinct values make the first-match repeated-key failure intentional. These baselines show why both query groups matter. Similar GRU scores do not prove it learned any particular shortcut.

## Verification

All42 full-pool evaluations pass:1,023,840 final predictions, zero independent argmax disagreements, maximum logit difference1.11e-13 and metric difference4.44e-16 (tolerance1e-10). The auditor independently unrolls GRU gates, recomputes attention and recomputes cross-entropy from log-sum-exp. Preflight gradients, causal prefixes and batch isolation pass for all six initial models; resetting hidden state at every token is detected for every GRU seed. Source/input hashes, paired streams, optimizer steps and saved pretest metrics also match. These numerical checks do not establish a learned algorithm or generalization beyond the tested data.

[Independent audit](../../../results/recurrent_retrieval_audit.json), [pretest snapshot](../../../results/recurrent_retrieval_pretest.json), [training source](../../../experiments/recurrent_retrieval.py), [audit source](../../../experiments/audit_recurrent_retrieval.py). There are three independent training seeds per model, not a million independent experiments.

## Interpretation and stopping decision

**Observed:** the tested Transformer package is substantially better at retrieval. The tested GRU package was better at the earlier state task. Those are separate fresh-training comparisons, not zero-shot transfer or a combined benchmark.

**Unknown:** why this GRU recipe fits poorly; whether other budgets, widths, supervision or initialization would change the ranking. Several model properties change together, so the result does not isolate attention versus recurrence alone. Language and broader reasoning remain untested for this matched architecture comparison.

The study is closed after the fixed six attempts and independent audit. No automatic tuning, extra seeds or language experiment. Original studies remain valid. The [sources and mechanism equations](recurrent-retrieval-guide.md) explain the models; this does not reproduce those papers' task results.
