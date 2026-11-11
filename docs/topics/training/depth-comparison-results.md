# Associative recall: depth versus width

This experiment asks whether sequential processing helps where two one-block optimizer recipes failed. It compares standard transformer configurations, not a novel architecture. The [protocol](../../depth-protocol.md) fixes training and criteria; [raw results](../../../results/depth_comparison.json) retain curves and checkpoints.

## Method and essential math

Each input contains three key/value pairs, a marker and a queried key. Predict its associated value using last-position cross-entropy. The existing 8,192 training and 1,024 validation episodes exclude reordered duplicates by canonical mapping/query identity. Validation has been inspected before: this is diagnostic evidence, not untouched confirmation.

A pre-norm block computes \(u=x+\mathrm{Attention}(\mathrm{LN}_1(x))\), then \(B(x)=u+\mathrm{FFN}(\mathrm{LN}_2(u))\). Depth composes \(B_2(B_1(x))\), allowing the second attention operation to read contextualized representations. Width enlarges vectors within one block. Whether the trained model actually uses a key/value binding mechanism remains untested.

For width d, FFN width f and L blocks, this implementation has \(P=28d+L(4d^2+2df+9d+f)\) parameters, including vocabulary-13 embedding/output matrices, biases and normalization. Both configurations use two heads and sinusoidal positions.

| Configuration | Blocks / width / FFN | Parameters |
|---|---|---:|
| Reused Adam baseline | 1 / 16 / 32 | 2,672 |
| Deep | 2 / 16 / 32 | 4,896 |
| Wide control | 1 / 24 / 40 | 5,152 |

All use Adam 0.001, betas 0.9/0.999, epsilon 1e-8, no decay, 3,000 updates, batch 64, CPU float64. Each run processes 192,000 examples (1,536,000 tokens). Five seeds share exactly the baseline minibatch sequences, verified by hashes. No tuning or checkpoint selection. Same seeds do not imply identical weights across different shapes. The wide control has 5.23% more parameters; compute, head width and elapsed time are not matched.

## Findings

Accuracy percentages; each cell is training / validation at update 3,000.

| Seed | Original one block | Two blocks | Wide one block |
|---|---:|---:|---:|
| 11 | 56.85 / 55.37 | 100 / 100 | 57.09 / 55.27 |
| 29 | 56.97 / 56.15 | 100 / 100 | 56.53 / 54.98 |
| 47 | 56.87 / 56.64 | 100 / 100 | 56.96 / 56.05 |
| 73 | 56.38 / 56.64 | 100 / 100 | 56.60 / 57.42 |
| 101 | 56.75 / 54.69 | 100 / 100 | 56.92 / 55.18 |
| Mean | 56.77 / 55.90 | 100 / 100 | 56.82 / 55.78 |

| Predeclared criterion | Observed | Outcome |
|---|---|---|
| Every deep seed >=90% train and validation | All 100% | Passed |
| Mean deep-minus-wide validation >=10 percentage points | 44.22 points; sample SD across five paired seeds 1.00 point | Passed |

Deep validation cross-entropy ranges from 0.000348 to 0.000570 nats. All final configurations have zero invalid-token predictions. The wider control remains below 90% on every seed; increasing this one block's parameter count did not resolve the plateau. Two blocks solved the inspected same-length split under this fixed recipe. This does not establish optimality, equal-compute superiority, longer-sequence transfer or a universal one-block limitation.

## Verification and execution correction

Copied-weight one-block output and gradient checks, and two-block future-token masking checks, all produced maximum error 0 in the tested fixtures. Checkpoint, source, protocol and minibatch hashes were verified after completion, including both execution stages and all ten new checkpoints. Finite losses and gradients passed at every update.

The initial protocol mistakenly counted the specified wide configuration as 5,024 parameters. Its assertion stopped execution after the five deep runs, before wide training. Direct counting gives 5,152. Dimensions and decision criteria were unchanged; completed deep checkpoints were hash-verified and reused. The [original code](../../../results/depth_run_provenance/compare_depth_initial.py) and [original protocol](../../../results/depth_run_provenance/depth-protocol-initial.md) are archived; the report identifies both execution stages. This was a counting error, not a failed training run.

## Paper comparison and remaining question

[Vaswani et al., section 3.1](https://arxiv.org/html/1706.03762v7#S3.SS1) already define stacked attention/FFN layers. We use a small causal pre-norm model without cross-attention, unlike their post-norm translation encoder-decoder. This is a local architecture comparison, not a reproduction of their benchmark or proof that one block cannot solve recall.

Follow-up: [fixed-recipe replication and separate length-transfer testing](depth-confirmation-results.md). This records overlap with earlier data. Controlled layer/path interventions are still needed to test the binding-mechanism hypothesis; do not infer it from accuracy or attention pictures alone.
