# Encoder-decoder contract results

Executed2026-10-02. All18 cases pass. The study is closed without training or selecting a next experiment.

[Learning guide and diagram](encoder-decoder-guide.md) · [Frozen protocol](../../protocols/encoder-decoder-contract.md) · [Raw evidence](../../../results/encoder_decoder_contract.json) · [Native and independent equations](../../../experiments/encoder_decoder_contract.py)

## Method

Seeds2341/2347/2351 × target/source lengths3/5,5/3,4/4 × target padding absent/present. Each batch has two examples; the second source always has one padded position. Cases are numerical fixtures, not trained replications.

One post-norm encoder and decoder layer, width8, two heads, ReLU FFN16, separate16-symbol embeddings, sinusoidal positions, bias-free output projection:1,888 parameters. CPU float64, PyTorch2.11.0+cu128, dropout0, zero optimizer steps. LayerNorm epsilon1e-5. Target fixtures end with EOS; BOS starts the shifted input. These random source/target fixtures define no learned sequence task.

Independent matrix equations reconstruct the three attentions, residuals, norms, FFNs and output. Shared embedding/position input construction is also exercised by a joint parameter finite difference. Masked loss is independently reduced by iterating valid target positions. Every parameter gradient and source/target feature gradient is compared for that scalar loss.

## Numerical agreement

| Check | Cases or positions | Worst absolute error | Limit |
|---|---:|---:|---:|
| Full logits |18 |5.551e-16 |1e-10 |
| Masked mean token loss |18 |8.882e-16 |1e-10 |
| Parameter and input-feature gradients |18 |2.220e-16 |1e-10 |
| Joint parameter directional difference, epsilon1e-6 |18 |6.296e-10 |1e-7 |
| Same-prefix decoding versus teacher forcing |135 positions |5.551e-16 |1e-10 |
| Perturb source padding before the encoder: valid logits/loss |18 |0 |1e-10 |
| Padded source feature gradients |18 |0 |1e-10 |
| Future target perturbation: earlier logits |18 |0 |1e-10 |
| Padded target feature and logit gradients |9 |0 |1e-10 |
| Perturb padded logits: valid-target loss |9 |0 |1e-10 |

The directional check covers one joint direction per case; it does not independently finite-difference every scalar. Source-padding perturbations act on embedded input features before encoding, not a tokenizer. Prefix decoding recomputes from identical supplied prefixes, without caches or sampled predictions.

| Seed | Cases | Logit error | Gradient error | Prefix error | Directional error |
|---|---:|---:|---:|---:|---:|
|2341 |6 |5.551e-16 |1.943e-16 |5.551e-16 |6.296e-10 |
|2347 |6 |4.441e-16 |1.388e-16 |4.441e-16 |3.573e-10 |
|2351 |6 |5.551e-16 |2.220e-16 |4.441e-16 |3.321e-10 |

## Intentional errors and input guards

Both signs of padding/future-input perturbations are checked. All numerical error controls exceed1e-6. Mask omissions are applied individually, so one cannot conceal the other.

| Violation | Detected | Smallest effect |
|---|---:|---:|
| Omit encoder source mask |18/18 |0.023641 logit difference |
| Omit decoder memory mask |18/18 |0.042925 logit difference |
| Omit decoder causal mask |18/18 |0.218803 earlier-logit difference |
| Include padded target rows in loss |9/9 |0.093650 padded-logit gradient |
| Supply unshifted target labels |18/18 |Index contract rejects |
| Empty source example |18/18 |Wrapper rejects |
| Empty valid-target loss |18/18 |Wrapper rejects |

No failed cases or threshold changes occurred in this study. It uses the two-sided perturbation lesson from the preserved [cross-attention fixture correction](../attention/cross-attention-results.md#recorded-fixture-correction).

## Conclusion and limits

**Observed:** this finite complete model agrees with independent layer/loss equations. Source masking works through both encoder and decoder; target loss ignores padding; fixed-prefix decoding preserves teacher-forced logits.

**Not established:** useful learning, generalization, free-running sequence accuracy, EOS stopping, GPU/reduced precision, caching, left padding or packed sequences. No accuracy baseline or train/test split is applicable because no learning task was run.

Post-norm layer order follows the [original Transformer](https://arxiv.org/html/1706.03762v7) and [PyTorch decoder contract](https://docs.pytorch.org/docs/2.11/generated/torch.nn.TransformerDecoderLayer.html). Loss masking follows [CrossEntropyLoss](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html). This verifies the selected implementation conventions, not the paper's benchmark claims.
