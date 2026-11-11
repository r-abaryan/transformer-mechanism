# Encoder-decoder input and loss contract

Freeze before execution. No optimizer updates or trained performance claims. Stop after18 cases: seeds2341/2347/2351 × target/source lengths(3,5),(5,3),(4,4) × target padding absent/present. Batch2; the second source always has one padded position. When target padding is present its second example has one padded position.

Model: separate16×8 source/target embeddings scaled by sqrt(8), fixed sinusoidal positions, one post-norm encoder layer and one post-norm decoder layer, two heads, ReLU FFN16, LayerNorm epsilon1e-5, biases enabled, dropout0, bias-free8-to16 output. Native PyTorch layers, CPU float64, deterministic algorithms. No extra final normalization. PAD=0, BOS=1, EOS=2; content symbols3..15. All weights remain at initialization.

Fixtures: random content symbols, each valid target ends with EOS. Decoder input is BOS followed by earlier target symbols; positions outside the target length are PAD. These are input-contract fixtures, not a learned copy task. Reject empty sources and empty valid-target loss reductions.

Independent equations reconstruct all three attentions, residuals, norms, FFNs and output projection. Compare logits, masked mean token loss, all parameter gradients and source/target feature gradients within1e-10. A separate loss reduction uses only valid positions, while retaining all16 vocabulary classes in each softmax denominator. Check one joint parameter directional central difference per case (epsilon1e-6; error≤1e-7).

For every valid target position compare teacher-forced logits against recomputation using exactly the same target prefix, with no cache. Require error≤1e-10. This verifies fixed-prefix equivalence, not free-running generation accuracy or EOS stopping.

Perturb source padding in both signs before the encoder; valid target logits and loss must stay unchanged within1e-10. Gradients on padded source/target feature rows must be zero within1e-10. Perturb future valid target features and require earlier logits unchanged. Changing padded target logits must preserve the masked loss. All-positive controls use1e-10.

Intentional errors must be detected: omit only the encoder source mask, omit only the decoder memory mask (each has a >1e-6 output effect under at least one perturbation sign), remove the decoder causal mask (>1e-6 prefix leakage), pass unshifted targets (independent index rejection), and include padded rows in the loss (nonzero padded-logit gradients >1e-6, when padding exists). Use both signs for future-input perturbations too. Test explicit empty-source and empty-loss rejection.

Save every case, worst errors, detection counts, environment and source/protocol hashes. Preserve earlier studies. Passing establishes this finite CPU implementation contract. It does not establish trained quality, generalization, GPU/precision behavior, KV caching, left padding, packed sequences or complete generation-loop correctness.

References: [original Transformer, sections3.1/3.5](https://arxiv.org/html/1706.03762v7), [PyTorch2.11 decoder layer](https://docs.pytorch.org/docs/2.11/generated/torch.nn.TransformerDecoderLayer.html), [CrossEntropyLoss](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html).
