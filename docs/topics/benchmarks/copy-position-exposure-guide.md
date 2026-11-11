# Teaching a short sequence about larger positional gaps

Our saved copy models often rank a later source position above the first one when the source becomes longer. That suggests a learning question: did short training provide too little experience with larger relative positions?

The experiment changes position IDs during training. Tokens, labels and sequence lengths stay the same.

| Example with three content tokens | Source position IDs | Decoder position IDs |
|---|---|---|
| Ordinary positions | 0, 1, 2, 3 | 0, 1, 2, 3 |
| One sampled layout | 1, 4, 7, 10 | 1, 4, 7, 10 |

The source is content+EOS. The decoder input is BOS+content, and its targets are content+EOS. Matching source/output rows share position IDs. The final ID belongs to the source marker and the decoder row predicting EOS. Each batch draws a new ordered subset without replacement from 0..10. No content tokens are added.

For rotary attention,

\[
(R_pq)^\top(R_rk)=q^\top R_{r-p}k.
\]

The relative gap changes the dot product. Short examples with spaced IDs expose the model to larger gaps while retaining the same number of competing tokens. This changes the training inputs to attention; it does not increase parameter count or add supervision targets.

**The prediction has two parts:** fewer first-source ranking failures on fresh longer inputs, and better full-sequence copying. Better rankings without better copying would be insufficient. Better copying without the ranking change would leave this particular explanation unsupported.

Ordinary evaluation uses contiguous IDs and generated prefixes. No correct route is supplied. Additional fixed spaced layouts check whether the treatment learned its training setting; four layouts per length are a representative probe, not exhaustive positional coverage.

A development pair can use 6,000 or 12,000 updates, selected only by declared training fit. Three fresh confirmation pairs follow only if fitting succeeds. Earlier short pools are reused; the random long test sequences are newly reserved. All attempts and failed criteria remain visible.

[Ruoss et al. (2023), section 3](https://arxiv.org/html/2305.16843v1#S3) was inspected for ordered positional sampling once per batch. This is an existing idea. Our small range, paired encoder/decoder IDs and contiguous test positions make this a local adaptation, not a paper reproduction or novel architecture. The formula is the standard rotation identity already used by the repository's RoPE implementation; the predicted local effects remain empirical questions.

[Results](copy-position-exposure-results.md): all three confirmation seeds reach 100% full copying on reserved length-8 and length-10 examples, with zero first-source outranking. The architecture and short token streams are unchanged. This supports the tested exposure hypothesis, but does not establish ranking as the sole cause. Test positions remain within the sampled training range. [Frozen protocol](../../protocols/copy-position-exposure.md).
