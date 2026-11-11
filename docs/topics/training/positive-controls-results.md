# Positive controls: learning works, general binding remains unproven

**Both predefined fitting criteria passed for every seed.** The existing architecture and SGD recipe can learn single-pair recall and fit 32 three-pair episodes. Small-set fit did not transfer to held-out episodes.

## What the controls test

A positive control asks whether the pipeline can succeed under simpler conditions. It does not identify every cause of failure on a harder task.

| Control | Data | Question | Limit |
|---|---|---|---|
| Single pair | All 32 key/value combinations | Can the model learn to output the sole stored value? | No competing bindings; copying the only value suffices |
| Small three-pair set | 32 balanced original training episodes, eight per target | Can the model fit a small sample of the actual task? | Fit may rely on memorization or sample-specific shortcuts |

The [protocol](../../positive-controls-protocol.md) fixed five seeds, 3,000 full-batch updates, learning rate 0.1 and a 99% final fitting criterion. Each batch contains all 32 episodes, giving 96,000 example presentations per run. Model: TinyDecoder(13,16,2,32), 2,672 parameters, CPU float64. Earlier checkpoints remain untouched; these controls use fresh models.

## Math and measurements

At the final query, training minimizes L=-(1/32) sum_i log p_theta(target_i | episode_i), followed by theta <- theta - 0.1 gradient(L). Accuracy measures the top prediction; loss measures its assigned probability. Both fitting loss and held-out results are necessary to distinguish fit from transfer.

| Seed | Single-pair fit | Three-pair subset fit | Subset fit loss, nats | Held-out three-pair accuracy | Held-out loss, nats |
|---|---:|---:|---:|---:|---:|
| 11 | 100% | 100% | 0.000231 | 44.04% | 4.7043 |
| 29 | 100% | 100% | 0.000250 | 41.31% | 4.9844 |
| 47 | 100% | 100% | 0.000274 | 40.92% | 5.2658 |
| 73 | 100% | 100% | 0.000207 | 41.89% | 5.1054 |
| 101 | 100% | 100% | 0.000231 | 42.58% | 4.9070 |

Single-pair losses range from 0.000235 to 0.000267 nats. Held-out subset-model accuracy averages 42.15%; the existing frequency baseline scores 55.57% on that same held-out set. Descriptive baseline checks after training give 100% for frequency prediction on single-pair data and 56.25% on the 32-example three-pair subset. These baseline checks were not separate predeclared success criteria.

## What changed in our assessment

| Evidence | Supported conclusion | Still unknown |
|---|---|---|
| Single-pair fitting passed | Basic input-to-loss-to-update learning works on this task encoding | Whether all larger-task behavior is implemented or optimized adequately |
| Tiny three-pair fitting passed | There is no blanket inability to fit these 32 examples | Why training on 8,192 episodes plateaued near 56% |
| Held-out performance remained weak | Small-set fit does not establish general associative recall | Roles of optimization, capacity, representation and learned shortcuts |

Full-batch training on 32 examples differs from sampled minibatches on the larger dataset. This is not a controlled optimizer comparison. The high held-out losses show that good training fit did not produce reliable held-out probabilities either.

The [original transformer paper](https://arxiv.org/html/1706.03762v7) describes a deeper encoder-decoder and other tasks; these local controls neither reproduce its results nor refute its architecture. We retain the bounded conclusions from [the associative experiment](../retrieval/associative-recall-results.md), without an impossibility claim.

## Reproduce and next boundary

Run `.venv/Scripts/python.exe experiments/positive_controls.py`. [Code](../../../experiments/positive_controls.py), [raw results](../../../results/positive_controls.json) and the [protocol](../../positive-controls-protocol.md) preserve the exact two training datasets, curves, ten checkpoints and hashes. Rerunning replaces these generated control artifacts. Execution completed with exit code 0; no GPU performance measurement or architectural change was made.

Next useful experiment: a fixed-budget optimizer comparison on the original full associative dataset, using training/validation outcomes to assess learnability before evaluating fresh held-out cases. That would test an optimization hypothesis directly. Do not infer a depth requirement or propose a new architecture from these controls alone.
