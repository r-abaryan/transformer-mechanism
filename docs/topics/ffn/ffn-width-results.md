# FFN width results:32 versus64

**Average gains, but the fixed adoption rule fails.** Six-pair mean accuracy rises from **85.86% to 91.44%** (+5.59 points). Four seeds improve; one loses 7.50 points and falls below 90%. Retain width 32 as the recipe baseline under the stopping rule. This does not mean it is more accurate or reliably adequate: both widths have failing seeds. Ten runs and all 30 independent evaluations completed; comparison closed.

## What changed

Only FFN hidden width changes in both blocks, from 32 to 64. Both arms use RoPE, ReLU, two pre-norm blocks, model width 16 and two width 8 heads. The [learning guide](../../ffn-width-guide.md) explains the diagram, feature-sum equation, rank limit and distinction from adding heads or layers.

Each FFN computes `W2*ReLU(W1*z+b1)+b2`, with normalized token vector z. More hidden units add nonlinear feature terms while input/output remain 16-dimensional. The wide model can represent the narrow function; our zero-padding witness checks that construction. Standard training initialization is different and need not discover a better function.

| Cost | Width 32 | Width 64 |
|---|---:|---:|
| Whole-model parameters | 5,088 | 7,200 |
| FFN projection MACs per token per block | 1,024 | 2,048 |
| Hidden features per token per FFN | 32 | 64 |

The larger model adds 2,112 parameters (+41.51%) and doubles FFN projection arithmetic, not necessarily whole-model runtime or memory.

## Method and data honesty

Five paired seeds, 6,000 Adam .001 updates, batch 64, alternating three/four-pair training. Non-FFN tensors start identically; different FFN shapes use standard shape-specific initializations and are not function-matched. All parameters train. Same minibatches and final-checkpoint rule; test predictions opened after all ten attempts. No failed or replaced run.

Three-pair train/validation/test data is **reused unchanged** from the GELU study: 160/20/40 families. Its test is an exposed diagnostic, not fresh confirmation. All 1,320 families under that definition had already been used; none were silently relabelled as fresh.

Four-pair families are fresh: 800 train/100 validation/200 test. Six-pair testing has 200 fresh families only. Audit checks their exclusion from the 3,920 earlier families and separation across new splits. Each family has 36 variants; six-pair testing therefore has 7,200 answers per model. Smaller fragments and other relabelings can remain familiar. Queries always target the sole duplicated key, so query-token dependence remains unproven.

See the [frozen protocol](../../ffn-width-protocol.md). Historical scores use different seeds/data and are not matched baselines for this comparison.

## Paired findings

Accuracy is individual-answer correctness. Three-pair columns are reused diagnostics; four/six-pair columns use fresh families.

| Seed | Width 32:3 /4 pairs, % | Width 64:3 /4 pairs, % | Width 32:6 pairs | Width 64:6 pairs | Gain, pp |
|---|---:|---:|---:|---:|---:|
| 1201 | 100.00 / 99.96 | 100.00 / 100.00 | 95.58% | 96.14% | +0.56 |
| 1217 | 100.00 / 99.94 | 100.00 / 99.88 | 91.96% | 92.64% | +0.68 |
| 1237 | 100.00 / 99.85 | 100.00 / 99.54 | 93.40% | 85.90% | -7.50 |
| 1259 | 100.00 / 99.39 | 99.72 / 99.67 | 83.04% | 90.43% | +7.39 |
| 1277 | 99.58 / 94.08 | 99.79 / 99.76 | 65.31% | 92.11% | +26.81 |

Mean paired gain is 5.59 points; median gain is 0.68. The largest gain (+26.81) comes from seed 1277, whose baseline also struggles at four pairs (94.08% versus 99.76%). Thus this comparison does not isolate length extrapolation from learning the shorter task. Seed 1237 worsens from 93.40% to 85.90% despite the larger function class.

| Six-pair mean metric | Width 32 | Width 64 | Meaning |
|---|---:|---:|---|
| Family accuracy | 30.50% | 41.30% | All 36 variants correct. |
| Reversal accuracy | 73.33% | 83.25% | Both an order and its reversal correct. |
| Obsolete-value rate | 2.31% | 1.70% | Choosing the older matching value; lower is better. |
| Cross-entropy | 0.78257 | 0.49498 | Penalty for true-answer probability; lower is better. |

Detailed diagnostics:

| Seed / width | CE:3 /4 /6 pairs | Six-pair family / reversal / obsolete, % | Accuracy by latest pair position2..6, % |
|---|---|---|---|
| 1201 / 32 | 0.00015 / 0.00137 / 0.25441 | 64.00 / 91.17 / 0.10 | 75.86 / 89.28 / 94.97 / 98.81 / 100.00 |
| 1201 / 64 | 0.00010 / 0.00018 / 0.19361 | 68.00 / 92.28 / 0.03 | 75.67 / 88.54 / 97.15 / 99.75 / 100.00 |
| 1217 / 32 | 0.00036 / 0.00301 / 0.51997 | 42.00 / 83.92 / 1.19 | 46.55 / 70.28 / 98.87 / 99.75 / 100.00 |
| 1217 / 64 | 0.00062 / 0.00466 / 0.39818 | 40.50 / 85.36 / 1.62 | 74.52 / 77.39 / 90.39 / 97.37 / 99.87 |
| 1237 / 32 | 0.00137 / 0.00482 / 0.32233 | 34.50 / 86.92 / 1.92 | 76.05 / 84.93 / 90.32 / 97.62 / 98.70 |
| 1237 / 64 | 0.00064 / 0.01653 / 0.87321 | 15.00 / 72.97 / 2.99 | 79.31 / 50.96 / 78.45 / 96.88 / 96.02 |
| 1259 / 32 | 0.00096 / 0.02288 / 0.92023 | 9.00 / 68.42 / 3.43 | 66.86 / 67.94 / 76.50 / 88.94 / 91.21 |
| 1259 / 64 | 0.00770 / 0.01076 / 0.53584 | 34.00 / 81.22 / 0.71 | 54.60 / 71.97 / 93.17 / 96.08 / 99.25 |
| 1277 / 32 | 0.01860 / 0.16082 / 1.89594 | 3.00 / 36.25 / 4.89 | 0.57 / 1.49 / 60.51 / 79.37 / 95.44 |
| 1277 / 64 | 0.00637 / 0.00647 / 0.47407 | 49.00 / 84.42 / 3.14 | 76.82 / 71.02 / 91.44 / 97.17 / 99.87 |

## Verification and stopping decision

| Check | Evidence |
|---|---|
| Parameters / initialization | Counts5,088/7,200 verified; shared non-FFN tensors match in all paired seeds. Full model initialization hashes reproduced separately. |
| Forward / gradients | Narrow matches prior implementation exactly on the fixture; wide errors <=4.45e-16 /3.56e-15 against independent attention. |
| Capacity / causality | Zero-padding witness matches within4.44e-16; future-token changes leave earlier outputs unchanged on both fixtures. |
| Data / optimizer | Reuse, exclusions, labels, reversal controls, regenerated streams and 6,000 optimizer steps verified. |
| Saved predictions | All 30 independent matrix-rotation/SDPA evaluations agree within8.97e-14; metrics and frozen hashes pass. |
| Mean six-pair gain >=2pp | Pass: +5.59pp. |
| Every wide seed >=90%; no seed worse | Both fail at 1237:85.90%, down 7.50pp. |
| Short-length preservation | All wide seeds >=99% at 3/4 pairs and mean losses <=.5pp: pass. Three-pair evidence remains diagnostic. |

CPU float64, one thread, PyTorch 2.14.0+cpu. Each run processes 3,456,000 tokens and 31,488,000 batch-weighted squared sequence lengths before head/layer factors. Loop totals: width 32 319.03s; width 64 334.48s. They include periodic validation, exclude checkpoint writes/final tests and are not controlled speed benchmarks.

## What we can conclude

Extra FFN capacity helps several runs substantially, but does not guarantee reliable improvement under this budget. The larger function class and the function found by training are different questions. This study does not identify whether initialization, optimization or learned feature use explains individual gains/losses. It does not prove that 32 units are universally sufficient or that 64 cannot solve the task.

Five seeds share one split; variants within each family are dependent. We did not compare equal-compute budgets, tuned optimizers, other tasks or other widths. Preserve the measured gains and failed rule. No further width, seed or budget search follows this result.

[Vaswani et al., v7, section3.3](https://arxiv.org/html/1706.03762v7) supplies the FFN structure. Our parameter counts and containment construction follow from it. This is a local recall comparison, not a reproduction of the paper's translation results or evidence of a universal FFN bottleneck.

## Evidence and reproduction

[Protocol](../../ffn-width-protocol.md), [training and curves](../../../results/ffn_width.json), [verification](../../../results/ffn_width_verification.json), [audit](../../../results/ffn_width_audit.json), [data](../../../results/ffn_width_data.json). Final weights and logits are retained in `results/ffn_width_checkpoints/` and `results/ffn_width_scores.pt`.

```powershell
.venv/Scripts/python.exe experiments/verify_ffn_width.py
.venv/Scripts/python.exe experiments/ffn_width.py
.venv/Scripts/python.exe experiments/audit_ffn_width.py
```

The runner resumes compatible state and leaves a completed report unchanged. Keep the recorded source/data hashes and prior artifacts needed to audit exclusions and reuse.
