# Positional comparison: sinusoidal versus RoPE

**Keep RoPE for this task and budget.** Mean six-pair accuracy rises from **81.91% to 99.86%** (+17.95 percentage points). All five paired seeds improve. All frozen criteria pass. This bounded comparison is closed.

## What changed, and why it matters

Sinusoidal positions are added to token embeddings. RoPE instead rotates Q/K inside each attention block. It changes how location enters the model, without adding learned parameters. The [short guide](positional-comparison-guide.md) explains both paths and includes a diagram.

For projected queries and keys, `(R(i)q)^T(R(j)k) = q^T R(j-i)k`. The explicit rotation depends on relative distance. This identity does **not** guarantee successful length transfer; training results test that separate claim. Removing additive positions and introducing rotations change multiple paths together, so this study does not isolate the cause of the gain.

## Method

The [frozen protocol](positional-comparison-protocol.md) fixed five paired seeds, 6,000 Adam updates, batch64, and alternating three/four-pair training. Both models have two pre-norm blocks, width16, two width8 heads, FFN32 and 5,088 parameters. CPU float64, one thread; PyTorch 2.14.0+cpu. Test predictions were computed only after all ten training attempts finished. No run failed or was replaced.

Both arms share initial learned tensors and minibatches. Each sees 3,456,000 tokens; the sum of batch_size*sequence_length^2 is 31,488,000 per run (before head/layer factors). RoPE adds rotation arithmetic, so equal tokens are not equal FLOPs. Recorded training-loop times include periodic validation and exclude checkpoint writes, initial validation and final tests: baseline 155.61s total, RoPE 189.18s. These sequential measurements are not a controlled speed benchmark.

Fresh full mappings exclude the recorded earlier datasets. Three-pair splits contain 1,600/200/400 mappings; four-pair splits 1,605/210/420. Six-pair testing uses 420 mappings and 15,120 predictions per model. Each mapping has six orders and every query. Smaller submappings can remain familiar. Predictions within a mapping are dependent.

## Findings

Accuracy measures individual answers. Mapping accuracy requires every tested order/query for that mapping to be correct: 36 answers at six pairs. Cross-entropy measures probability assigned to the correct answer; lower is better.

**Every run scores 100% test accuracy and mapping accuracy at three and four pairs.** Six-pair results:

| Seed | Sinusoidal accuracy | RoPE accuracy | Gain, pp | Sinusoidal mapping | RoPE mapping |
|---|---:|---:|---:|---:|---:|
| 907 | 74.05% | 99.77% | +25.71 | 0.00% | 94.52% |
| 929 | 87.51% | 99.92% | +12.41 | 1.43% | 97.62% |
| 953 | 76.29% | 99.85% | +23.56 | 0.00% | 95.00% |
| 977 | 88.96% | 99.87% | +10.91 | 0.95% | 95.71% |
| 997 | 82.74% | 99.89% | +17.14 | 0.48% | 96.67% |

Cross-entropy and six-pair accuracy by queried pair position (first to sixth):

| Seed / method | CE: 3 / 4 / 6 pairs | Accuracy by position, % |
|---|---|---|
| 907 / baseline | 0.000140 / 0.000166 / 1.670445 | 90.00 / 87.06 / 89.40 / 88.65 / 61.07 / 28.13 |
| 907 / rope | 0.000071 / 0.000077 / 0.006469 | 99.05 / 99.88 / 100.00 / 99.92 / 99.84 / 99.92 |
| 929 / baseline | 0.000131 / 0.000145 / 0.726371 | 97.66 / 97.46 / 95.52 / 89.88 / 76.67 / 67.90 |
| 929 / rope | 0.000068 / 0.000081 / 0.002081 | 99.72 / 99.88 / 100.00 / 99.96 / 99.96 / 100.00 |
| 953 / baseline | 0.000127 / 0.000151 / 1.609977 | 98.73 / 99.17 / 98.17 / 91.11 / 44.13 / 26.43 |
| 953 / rope | 0.000078 / 0.000086 / 0.004410 | 100.00 / 100.00 / 100.00 / 99.68 / 99.64 / 99.76 |
| 977 / baseline | 0.000109 / 0.000143 / 0.802980 | 100.00 / 99.80 / 99.96 / 99.40 / 85.75 / 48.85 |
| 977 / rope | 0.000082 / 0.000087 / 0.003272 | 99.60 / 100.00 / 99.96 / 99.96 / 99.84 / 99.88 |
| 997 / baseline | 0.000115 / 0.000192 / 0.938178 | 95.24 / 92.70 / 92.54 / 88.41 / 74.05 / 53.53 |
| 997 / rope | 0.000084 / 0.000094 / 0.004062 | 99.40 / 99.96 / 100.00 / 99.96 / 100.00 / 100.00 |

The baseline is weakest at later pair positions in these tests. RoPE improves the overall result, but still makes errors: its mean all-variant mapping accuracy is 95.90%. Near-perfect answer accuracy is not perfect reliability.

## Verification and decision

| Check | Evidence |
|---|---|
| Baseline equivalence | Output error 2.23e-16; gradient error 1.34e-15 against the existing implementation. |
| Rotation math | Matrix, norm, relative-position and common-offset checks pass; maximum error below 1.8e-15. |
| Gradient / causality | One parameter finite-difference error 2.55e-10; future-token change leaves earlier outputs unchanged on the fixture. This is not an exhaustive gradient proof. |
| Independent audit | All 30 checkpoint evaluations match saved logits within 1.5e-13. Baseline uses library attention; RoPE uses explicit rotation matrices and PyTorch SDPA. |
| Provenance and data | Protocol/source/checkpoint hashes, mapping isolation, label/rank balance, paired initial tensors and regenerated minibatch streams pass. |
| Frozen keep rule | All RoPE seeds >=90% at six pairs; mean gain >=5pp; no seed worsens; all >=99% at three pairs; no mean short-length loss. All pass. |

Checkpoint files retain the latest resumable state; validation curves retain steps 0/1000/.../6000. Raw metrics and hashes are in [training results](../results/positional_comparison.json), [verification](../results/positional_comparison_verification.json) and [audit](../results/positional_comparison_audit.json). The runner's opening docstring was inherited from the exposure study; its executed loop trains **both** arms on three/four pairs, as verified by stream hashes and token counts.

## Paper comparison and limits

[RoFormer, Su et al., section 3](https://arxiv.org/html/2104.09864v5) supplies the rotation construction and relative-position identity. Our numerical checks agree with those identities. Our recall experiment supplies separate evidence for transfer from three/four to six pairs; it does not reproduce the paper's benchmarks or establish universal superiority.

Five initialization seeds share one generated split. This is not five independent datasets or a significance test. Other lengths, tasks, model sizes, precisions and training budgets remain untested. The full representation replacement helps here; a routing-only causal explanation remains unresolved. No novel architecture is claimed.

Stop here: no extra seeds, frequency tuning or alternative selection on this test set. Preserve RoPE as the successful candidate for a separately scoped future study.

## Reproduce

```powershell
.venv/Scripts/python.exe experiments/verify_positional_comparison.py
.venv/Scripts/python.exe experiments/positional_comparison.py
.venv/Scripts/python.exe experiments/audit_positional_comparison.py
```

The runner resumes compatible checkpoints and leaves a completed report unchanged. Preserve the input exclusion artifacts and recorded source hashes.
