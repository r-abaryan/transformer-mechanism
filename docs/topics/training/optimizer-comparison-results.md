# SGD versus Adam on associative recall

**The tested Adam recipe did not resolve the learning plateau.** All three predefined criteria failed. Ten runs completed; original SGD train/validation metrics reproduced exactly.

## Method: change the update rule, keep the model fixed

An optimizer converts gradients into parameter updates. Comparing matched runs tests whether one specific update recipe overcomes the observed training plateau. It does not isolate all possible optimization choices.

| Setting | SGD | Adam |
|---|---|---|
| Learning rate | 0.1 | 0.001 |
| Momentum / moments | No momentum | beta1=0.9, beta2=0.999 |
| Epsilon | Not applicable | 1e-8 |
| Weight decay, scheduler, clipping | None | None |

Both use five matched seeds, the same one-block 2,672-parameter model, identical initial weights and minibatch indices, 3,000 updates, batch 64, CPU float64. Final weights only. The original train/validation splits are reused; no test-set evaluation is performed. See the [predefined protocol](../../optimizer-comparison-protocol.md).

## Essential mathematics

For gradient g_t, SGD uses theta_t=theta_(t-1)-eta*g_t. Adam maintains per-parameter averages:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]
\[
\hat m_t=m_t/(1-\beta_1^t),\quad \hat v_t=v_t/(1-\beta_2^t),\quad
\theta_t=\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon).
\]

Operations are elementwise; moments start at zero. Bias correction compensates for that initialization. The implementation follows [PyTorch 2.14 Adam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html); the primary method is [Kingma and Ba](https://arxiv.org/abs/1412.6980v9). This experiment does not reproduce their benchmarks or audit convergence proofs.

## Evidence boundary

This compares optimizer-plus-learning-rate recipes, not optimizers with equal tuning budgets. Equal updates do not imply equal runtime or memory use. The reused validation set supports diagnostic comparison, not untouched confirmation. Matching hashes establish that paired models start and receive data identically; they do not prove anything about untested recipes.

## Results: every matched pair

Training accuracy tests whether the full training set was learned; validation assesses transfer to the reused diagnostic split. All values below are percentages, and gains are percentage points.

| Seed | SGD train | Adam train | SGD validation | Adam validation | Paired gain |
|---|---:|---:|---:|---:|---:|
| 11 | 56.03 | 56.85 | 54.10 | 55.37 | +1.27 |
| 29 | 55.92 | 56.97 | 54.59 | 56.15 | +1.56 |
| 47 | 54.11 | 56.87 | 53.52 | 56.64 | +3.12 |
| 73 | 56.03 | 56.38 | 54.30 | 56.64 | +2.34 |
| 101 | 55.96 | 56.75 | 55.27 | 54.69 | -0.59 |
| Mean | 55.61 | 56.77 | 54.36 | 55.90 | +1.54 |

Paired gain sample SD: 1.39 points across seeds. Mean validation cross-entropy decreased from 0.796727 to 0.786184 nats. The previously measured frequency baseline achieves 56.35% on this validation split; the model mean remains below it. No significance or population-uncertainty claim is made from these five pairs.

| Predefined criterion | Outcome |
|---|---|
| Adam training accuracy at least 90% for every seed | Failed |
| Adam validation accuracy at least 90% for every seed | Failed |
| Mean paired validation gain at least 5 points | Failed: 1.54 points |

## Verification and interpretation

Initialization and minibatch hashes matched within every pair. All losses and gradients were finite. Every SGD final training/validation accuracy and loss matched the earlier report with difference zero. Each run consumed 192,000 examples and 1,536,000 input tokens. Original experiment checkpoints were preserved; ten new final checkpoints were saved separately.

The evidence rules out the narrow prediction that this fixed Adam recipe would solve the task within 3,000 updates. It does not rule out longer training, different learning rates, schedules, initialization or other optimizers. It also does not establish a one-layer impossibility result or an architectural defect. Adam's published algorithm and our synthetic-task outcome answer different questions; no paper benchmark was reproduced.

## Reproduce and next decision

Run `.venv/Scripts/python.exe experiments/compare_optimizers.py`. [Code](../../../experiments/compare_optimizers.py), [raw report](../../../results/optimizer_comparison.json), and [protocol](../../optimizer-comparison-protocol.md) retain all curves, hashes and final weights. Rerunning replaces comparison artifacts. CPU timings are descriptive and include evaluation; optimizer-state memory was not measured.

Stop this comparison at its declared budget. The next useful research question is whether an additional sequential attention step makes the required binding easier to learn. A depth experiment would need a wider one-block parameter control, fixed training recipes and fresh evaluation design. This would investigate a standard transformer design choice, not propose a novel architecture or assume depth is the cause.
