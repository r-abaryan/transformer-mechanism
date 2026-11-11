# Which attention operation needed positional exposure?

The [previous study](copy-position-exposure-results.md) fixed longer copying by training on spaced positions. But it changed three attention operations together. This comparison separates two groups of operations.

| Operation | What it does | What spacing changes |
|---|---|---|
| Encoder self-attention | Builds source representations | Relative positions among source tokens |
| Decoder self-attention | Processes the generated prefix | Relative positions among prefix tokens |
| Cross-attention | Reads the source from the decoder | Relative positions between reading queries and source keys |

Self-attention here groups the first two rows. Four training conditions enable spacing nowhere, in self-attention only, in cross-attention only, or in both. All use the same model, examples and matched update budget. Evaluation returns to ordinary positions.

RoPE gives the score `qᵀ R_(r−p) k / √d`. Changing the positions changes the rotation between the query and key. A token at position 2 can therefore appear at sampled position 7 without adding five tokens. This exposes the model to different distances while keeping content length short.

Corresponding source and decoder rows share their ID. The correct copy position therefore keeps relative gap zero. Spacing broadens the gaps to other positions. It changes the offsets of competing keys without adding longer training strings; it does not force the model to select the correct key.

The distinction matters: success from cross-only exposure would show that broader positional exposure inside self-attention is not needed for this tested repair. It would **not** mean the encoder never changed. Cross-attention's training loss sends gradients back through the encoder too.

At the first decoding step, the prefix contains only BOS. Decoder self-attention has one allowed key, so its softmax weight is 1 regardless of that score. Later steps have more prefix tokens. This is why first-token rankings cannot replace full-sequence copying as the outcome.

We require each arm to learn the short task before interpreting long-copy differences. The combined arm must reproduce the earlier benefit on fresh examples. Single-site success must preserve copying and improve the predicted ranking measure. A good mean cannot hide a failed seed or group.

[Ruoss et al. (2023), section 3](https://arxiv.org/html/2305.16843v1#S3) supplies the sorted positional-sampling method; that section was inspected. The equation is the existing RoPE rotation identity. Which site helps this small trained model is our empirical question, not a prediction from the paper. This is an ablation of an existing method, not a new architecture.

**Measured result:** combined exposure reaches 100% random long-copy accuracy in all three seeds. Cross-only succeeds in two seeds but fails at length 10 in the third; self-only remains weak. Every arm fits short training. The combination is reliable under the tested rule, but cross-only success disproves universal necessity of self-site exposure. [Results and all-group tables](copy-position-sites-results.md).

Test positions remain inside the sampled training range. No arbitrary-length claim follows. [Protocol](../../protocols/copy-position-sites.md). Execution and audit are complete.
