# Trained competition: sharper versus correctly directed attention

**Blind sharpening fails the fixed improvement criterion. Target-directed attention largely rescues the same models without changing V.** This establishes a local intervention effect, not a deployable repair or proof that distractor-count dilution caused the original failures.

## What changed, and why?

Five frozen two-block models, seeds 137/163/191/223/257; the same 1,024 archived six-pair examples per model. This is reused diagnostic data, not fresh confirmation. No training, position changes or token removal. Only the last query's weights in block 2 change, in both heads. Their value vectors and incoming residual are identical across conditions; the FFN and readout then run normally. [Frozen protocol](../../trained-competition-protocol.md).

| Condition | Math / explanation |
|---|---|
| Ordinary | `a=softmax(s)`. |
| Sharpen x2 | `a'=softmax(2s)`. Equivalent to halving temperature. Preserves score ordering, so it cannot correct an incorrectly ranked target. |
| Target floor .9 | For the known target-value token, `p'=max(p,.9)`; multiply every other weight by `(1-p')/(1-p)`. Preserves ratios among the other tokens. Uses task knowledge: an oracle diagnostic. |
| Wrong-slot floor .9 | Apply the same rule to the cyclic next pair's value token. A positional control, not always a wrong answer: 260/1,024 examples share the target value at that slot. |

The floor intervention can be implemented independently by setting the chosen score to `max(s_t, logsumexp(s_others)+log(9))`. Unlike uniform sharpening, this can change the winner. Neither operation reproduces the representational changes caused by deleting real distractors.

## Results

Accuracy percentages. Twenty conditions total.

| Seed | Ordinary | Sharpen x2 | Target floor | Wrong-slot floor |
|---|---:|---:|---:|---:|
| 137 | 68.75 | 68.85 | 100.00 | 25.49 |
| 163 | 62.89 | 62.99 | 100.00 | 25.39 |
| 191 | 64.36 | 63.38 | 99.90 | 25.59 |
| 223 | 54.88 | 51.37 | 100.00 | 25.39 |
| 257 | 70.31 | 68.46 | 98.83 | 25.68 |
| Mean | **64.24** | **63.01** | **99.75** | **25.51** |

| Intervention | Original errors rescued | Originally correct predictions harmed | Mean cross-entropy |
|---|---:|---:|---:|
| Ordinary | 0 | 0 | 2.28405 |
| Sharpen x2 | 159 | 222 | 2.61494 |
| Target floor | 1,820 | 2 | .00853 |
| Wrong-slot floor | 315 | 2,298 | 6.71594 |

Counts pool 5,120 model/example evaluations, not independent trials. Target-floor evaluation still has 13 errors: 11 original errors remain and two new errors appear. No failed case was removed.

## Is this dilution or incorrect ranking?

Of 1,831 original errors, only 87 have the target ranked first in both heads; the other 1,744 do not. These descriptive strata do not prove that both heads must select the target.

| Original-error stratum | Count | Rescued by x2 | Rescued by target floor |
|---|---:|---:|---:|
| Target first in both heads | 87 | 61 | 85 |
| Other rankings | 1,744 | 98 | 1,735 |

For the first stratum, increasing concentration can rescue some errors without changing the target's rank. Across the complete diagnostic, simply making existing preferences stronger is insufficient. Target-floor rescue shows that a weight replacement is often sufficient with the existing values and downstream computation. It does not show that values are perfect, isolate one head, or establish the cause of the poor scores.

## Relation to current research

[Velickovic et al., ICML 2025](https://proceedings.mlr.press/v267/velickovic25a.html) studies dispersion and proposes adaptive temperature. Our fixed factor of two is not their adaptive method; this failure does not refute that paper. The [recent-source comparison](attention-competition-results.md#updated-research-comparison) also records ICML 2025 sparse-dependency theory and ICML 2026 Softplus/re-weighting work, with reading depth explicit. None is reproduced by this diagnostic.

## Verification and decision

Archived baseline metrics match within 1e-10. Explicit, library-hook and sham controls have maximum logit difference 2.94e-14 (rounded upward). Checkpoints and V remain unchanged. All weights normalize and all logits are finite. A separate audit uses squared/renormalized probabilities for sharpening, the logit-bias construction for floors, separate head projections, and final-block replay. Its maximum logit error is 7.11e-15. It independently reproduces all twenty metrics and paired counts. Both paths still use PyTorch, not independent frameworks. The audit initially stopped on a float32/float64 comparison mismatch; fixing its reference dtype allowed the unchanged experiment evidence to pass.

**Reject this fixed x2 setting:** mean gain is -1.23 percentage points, and three seeds worsen. It fails the predeclared requirement of at least +5 points on average with positive gain in every seed. Do not try more temperatures on this opened evaluation. Target-floor results are explanatory only. Close this diagnostic; do not turn the oracle into an architectural improvement claim.

[Runner](../../../experiments/trained_competition.py), [raw report](../../../results/trained_competition.json), [saved scores, values, residuals and logits](../../../results/trained_competition_tensors.pt), [auditor](../../../experiments/audit_trained_competition.py), [audit summary](../../../results/trained_competition_audit.json). Run each script with `.venv/Scripts/python.exe`; the runner protects completed artifacts.
