# Does changing the query change the answer?

**Yes: these models respond to the query, but transfer is uneven.** Mean switched-query accuracy is **81.59%**, versus **98.64%** on the original repeated-key queries. Three of five seeds miss the fixed90% transfer criterion. Diagnostic closed; no training occurred.

## What we changed

Keep the context fixed. Change only its final query token.

```text
Context: A:red | B:blue | A:green | C:yellow
Original query A -> correct answer green
Changed query B  -> correct answer blue
Changed query C  -> correct answer yellow
```

This is an explanatory example. Actual inputs use numeric token IDs. Training always queried the sole repeated key. The changed queries ask about keys appearing once, so they are outside that query distribution.

## Method

Use all five saved FFN32/RoPE/ReLU models from the width study. Evaluate every four-pair test context:7,200 contexts from200 dependent families, with one original and two changed queries each. This gives14,400 changed-query answers per seed. Four pairs is a training length, avoiding an additional length change. Contexts were already evaluated, not newly reserved.

For context c and queries q and q', compare `f(c,q)` with `f(c,q')`. We hold weights and context tokens fixed. A changed prediction demonstrates an effect of the query token on the output for that input. Correctly changing to the requested value additionally tests retrieval behavior. It does not locate the internal causal path.

## Findings

All percentages. Unchanged means the predicted token stays the same after the query changes. Duplicate-answer rate means the changed-query output equals the latest value of the repeated key.

| Seed | Original accuracy | Changed-query accuracy | Unchanged prediction | Duplicate-answer rate | All3 queries correct per context |
|---|---:|---:|---:|---:|---:|
| 1201 | 99.96% | 90.93% | 4.06% | 4.06% | 81.93% |
| 1217 | 99.94% | 94.26% | 2.92% | 2.89% | 88.47% |
| 1237 | 99.85% | 85.71% | 6.10% | 6.08% | 71.68% |
| 1259 | 99.39% | 70.62% | 11.59% | 11.36% | 45.47% |
| 1277 | 94.08% | 66.44% | 8.68% | 6.54% | 33.97% |

Mean unchanged-prediction rate is 6.67%. An always-return-the-duplicate-answer rule would score0% on changed queries because all values differ. The observed scores rule out that rule as a complete explanation of these models' behavior.

Only seeds1201 and1217 reach90% changed-query accuracy. Partial transfer is real, but the all-seed criterion fails. Errors do not by themselves show that a model ignored the query: it may use it incorrectly. Likewise, unchanged argmax does not imply identical logits.

## Verification

Only the last token changes. Labels were checked against an independent reverse scan of the context. Original predictions match saved logits exactly for all five models. Independent matrix-rotation/SDPA computation matches all original and changed logits within8.7e-14. Checkpoint/data/source hashes pass. Logits change in every model, with per-model maximum differences from14.91 to20.79.

[Fixed protocol](../../query-switch-protocol.md), [results and hashes](../../../results/query_switch.json), [executable](../../../experiments/query_switch.py). Exact inputs, labels and logits are saved in `results/query_switch_scores.pt`.

```powershell
.venv/Scripts/python.exe experiments/query_switch.py
```

## What this adds to the walkthrough

The [attention trace](../retrieval/trained-recall-walkthrough.md) showed where two examples attended. This intervention now shows that changing the input query changes behavior across a fixed dataset. It refutes exact query invariance here; it does not prove that query processing exclusively follows any particular head or Q/K/V route.

Five seeds share contexts; variants within families are dependent. Unique-key queries were not trained. Failure is not evidence that transformers cannot learn them. No architecture changes, retraining, new seeds or parameter search follow this diagnostic.
