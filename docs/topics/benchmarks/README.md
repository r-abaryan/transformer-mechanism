# Benchmark comparisons

- [Tracing the exposure effect through residual, normalization and FFN stages](copy-exposure-stage-trace-results.md)

- [Shared-checkpoint positional exposure: improved copying, incomplete repair](copy-exposure-continuation-results.md)

- [Whole cross-attention output transfer: failed portability and short-task damage](copy-cross-output-transfer-results.md)

- [Classifier-only terminal gradients: stopping and training-fit failure](copy-terminal-gradient-route-results.md)

- [Terminal feedback during development: conditional improvement, lost stopping](copy-eos-development-results.md)

- [Head-0 pattern transfer: complete local recovery with a location control](copy-head-pattern-transfer-results.md)

- [Intact developmental trace: attention pattern and downstream differences](copy-intact-development-results.md)

- [Historical Q/K updates: conditional effects and hybrid damage](copy-routing-development-results.md)

- [Copy learning trajectory: verified replay and early divergence](copy-learning-trajectory-results.md)

- [Added decoder-read masking: partial recovery](copy-added-read-mask-results.md)

- [Matched-prefix extension triggers early EOS](copy-prefix-extension-results.md)

- [Direct end-marker route: failed early-EOS hypothesis and terminal harms](copy-end-route-results.md)

- [Cross-only failures: early stopping hides mostly correct content](copy-site-failure-results.md)

- [Where positional exposure helps copying: results](copy-position-sites-results.md), [guide](copy-position-sites-guide.md) and [four-condition protocol](../../protocols/copy-position-sites.md)

- [Positional-exposure learning: confirmed long-copy repair](copy-position-exposure-results.md), [educational guide](copy-position-exposure-guide.md) and [bounded protocol](../../protocols/copy-position-exposure.md)

- [Copy source visibility: first-token recovery with the native encoder](copy-source-visibility-results.md) and [short guide](copy-source-visibility-guide.md)

- [Marked RoPE source alignment: conditional full recovery](copy-marker-alignment-results.md)

- [Training with a source marker: learned short copying, weak extrapolation](copy-marker-training-results.md)

- [Source-end marker: breaking a collision versus learning to stop](copy-source-end-marker.md)

- [Sinusoidal/RoPE copying: results](copy-rope-results.md) and [learning guide](copy-rope-guide.md)

- [Removing terminal EOS supervision: results](copy-eos-training-results.md) and [learning guide](copy-eos-training-guide.md)

- [Broader copy lengths: results](copy-length-exposure-results.md) and [learning guide](copy-length-exposure-guide.md)

- [Long-copy content: aligned cross-attention intervention](copy-aligned-routing-results.md)

- [Does a one-time momentum reset help continued state learning?](state-momentum-training-results.md)

- [Saved Adam momentum: direction changes and remaining reset tradeoffs](state-adam-momentum-results.md)

- [Remaining state-training signal: gradients and group tradeoffs](state-training-gradients-results.md)

- [Block-1 initial-cue transport: K, V and destination groups](state-block1-transport-results.md)

- [Initial-bit transport through attention and residual paths](state-initial-paths-results.md)

- [Reset response versus suffix parity: weak-seed diagnostic](state-reset-response-results.md)

- [Independent reset exposure: paired training and counterfactual results](state-reset-exposure-results.md)

- [Remaining state errors: parity and reset cue interventions](state-recipe-errors-results.md)

- [Training-only state recipe development: eight attempts](state-recipe-development-results.md)

- [State-tracking input sensitivity and parity controls](state-sensitivity-results.md)

- [Learned causal global relay](../attention/global-retrieval-results.md)

- [Full/local marked-value retrieval](../attention/window-retrieval-results.md)

- [Copy stopping diagnostic](copy-stopping-results.md)

Word prediction, retrieval, majority aggregation, sequential state tracking, graph composition and sequence copying.

## Learn

- [Sequence copying: teacher forcing, generation and EOS](encoder-decoder-copy-guide.md)
- [Learning adequacy: fit lookup before interpreting composition](composition-learning-adequacy-guide.md)
- [Combining two relationships: graph lookup](recurrent-composition-guide.md)
- [GRU versus Transformer: predicting the next word](recurrent-language-guide.md)
- [GRU versus Transformer: retrieving associations](recurrent-retrieval-guide.md)
- [GRU versus Transformer: how state is carried](recurrent-state-guide.md)
- [Intermediate-state supervision: targets and gradients](state-supervision-guide.md)
- [State tracking: ordered updates and the reset shortcut](state-tracking-guide.md)
- [From toy tasks to WikiText-2](wikitext2-comparison-guide.md)

## Findings

- [Encoder-decoder copy: perfect training fit, limited transfer](encoder-decoder-copy-results.md)
- [GPU graph lookup: gains, but the fit gate still fails](composition-learning-adequacy-results.md)
- [One-step/two-step graph lookup: both recipes fail short fitting](recurrent-composition-results.md)
- [GRU versus Transformer: language prediction within a fixed budget](recurrent-language-results.md)
- [GRU versus Transformer: retrieval comparison](recurrent-retrieval-results.md)
- [GRU versus Transformer: bounded state-tracking advantage](recurrent-state-results.md)
- [Intermediate supervision: improved fitting, weak length transfer](state-supervision-results.md)
- [State tracking: reset-free updates fail at seen lengths](state-tracking-results.md)
- [Majority transfer: aggregation works, reliability has a boundary](majority-transfer-results.md)
- [WikiText-2: shared K/V with measured quality loss](wikitext2-results.md)
- [WikiText-2: normalization transfer](../normalization/wikitext2-norm-results.md)
- [WikiText-2: input/output weight tying](../embeddings/weight-tying-results.md)

[All subjects](../README.md) · [Frozen protocols](../../protocols/README.md)
