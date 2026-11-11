# From toy tasks to WikiText-2

**Completed:** WikiText-2 word-level prediction, comparing MQA with matched MHA under [Decision0003](../../decisions/0003-wikitext2-matched-baseline.md). The [audited results](wikitext2-results.md) accept half-size compact K/V payload with 1.96% worse geometric mean perplexity. Study closed.

## What changes, and why

Recall tests retrieve an associated value. Majority tests aggregate binary tokens. Here the model predicts the next word in published natural text. The official split replaces our generated examples. This broadens the evidence without proving general language-model superiority.

Example of correct target shifting:

```text
Inputs:   the   cat   sat
Targets:  cat   sat   down
```

Each position predicts the following word. Predicting the same input word would allow a copying shortcut. The causal mask and the shifted targets protect different parts of the task.

Perplexity is `exp(mean negative log probability of the true next word)`. Lower is better. A model assigning each true word probability 1/100 has perplexity 100. Real probabilities vary; token losses are averaged before exponentiation. Perplexity is not accuracy.

## What we verified

| Check | Live result | Limit |
|---|---|---|
| Publisher files | All three hashes match; 7,371,282 bytes total | One pinned word-level revision |
| Corpus encoding | Independent parsing agrees: 2,088,628 / 217,646 / 245,569 tokens; train-only vocabulary 33,278 | Publisher `<unk>` tokens remain |
| Matched initialization | Common tensors and retained first K/V head match in all three paired seeds | Sharing changes tensor shapes and parameter count |
| GPU functions and gradients | All six preflights pass against CPU float64 | Tested FP32 native Windows path |
| Causality and targets | Prefix checks and deliberate shifting errors detected; exactly-once evaluation coverage passes | Context resets every 64 positions |
| Final test loss | All 1,473,408 model/token losses pass independent float64 audit | One argmax difference; loss agreement is within tolerance |

[Acquisition](../../../results/wikitext2_acquisition.json), [publisher attribution](../../../data/wikitext-2-v1/publisher-README.md), [data audit](../../../results/wikitext2_data_audit.json), [GPU preflight](../../../results/wikitext2_gpu_verification.json), [final audit](../../../results/wikitext2_training_audit.json).

For vocabulary V, parameters are `32V+4480` (MHA) and `32V+3936` (MQA). With the actual vocabulary these are 1,069,376 / 1,068,832. Saving 544 parameters is about 0.051%, because word embeddings/readout dominate. Potential compact 64-token FP32 K/V payload halves, 16 KiB to 8 KiB per sequence. Training uses no cache. Parameter count, cache size and speed are separate measurements.

## Completed bounded comparison

Three paired seeds completed 6,000 updates each with matched token starts. All six attempts finished before test scoring. Both arms beat the training-only unigram baseline; geometric mean MQA/MHA perplexity ratio is 1.01964 (limit 1.05), with every seed below 1.10. MQA is worse in every pair, despite passing this declared tolerance. No sweep or budget extension is selected.

Evaluation resets context every 64 positions. These scores are specific to that protocol. They do not reproduce published long-context perplexities or establish speed gains. [Merity et al. 2016](https://arxiv.org/html/1609.07843v1) and the [publisher card](https://huggingface.co/datasets/Salesforce/wikitext) establish the dataset's origin and variants; the relevant original source takes priority over age.
