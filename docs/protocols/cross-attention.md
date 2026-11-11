# Cross-attention correctness protocol

Frozen before execution. No training, dataset selection or architecture ranking.

Question: does a projected multi-head cross-attention implementation agree with independently expressed equations, and do its masks enforce the stated visibility contract?

Use CPU float64, width8, two heads, encoder-memory width6, batch2, dropout0. Seeds2309/2311/2333; target/source lengths(2,5),(5,3),(4,4); projection biases on/off; source padding absent/present. This fixes36 cases. In padded cases the second batch item has one invalid source row. Memory means supplied encoder output; a complete encoder is outside scope.

Compare PyTorch MultiheadAttention with explicit per-batch/per-head projected matrix equations. Require maximum output, attention-weight and all input/parameter gradient differences ≤1e-10. Use a fixed random scalar output probe. Check one joint directional central finite difference over inputs and every parameter per case, epsilon1e-6, absolute derivative error ≤1e-7.

Padding tests: changing invalid memory rows must preserve outputs and their gradients must be zero within1e-10. Permuting memory rows together with their padding mask must preserve output within1e-10; this does not permute raw source tokens before encoding. Reject all-masked source rows before native attention; no backend-wide behavior claim follows.

For causality, compose a causal target self-attention residual with the tested cross-attention residual. Fix source memory independently of targets. Perturb target inputs after the midpoint; earlier outputs must agree within1e-10. This small assembly omits encoder, FFN, normalization, positional encoding and training.

Intentional errors, detected by output changes >1e-6: ignore source padding, remove decoder self-attention's causal mask, and incorrectly impose a triangular target/source cross-mask. For padding controls perturb the masked memory substantially. Check teacher-forcing input `[BOS,y0,...,y(T-2)]` independently by indices; the unshifted target sequence must fail the check. Cross-attention alone cannot fix wrong target shifting.

Save all36 cases, worst errors, negative-control counts, environment and source/protocol hashes. Stop after these checks. Any numerical failure is investigated and recorded; thresholds are not changed to hide it. Passing supports these finite arithmetic/masking cases, not training quality, all precisions, GPU kernels or full encoder-decoder correctness.

Primary conventions: [Vaswani et al., sections3.1-3.2](https://arxiv.org/html/1706.03762v7); [PyTorch2.11 MultiheadAttention](https://docs.pytorch.org/docs/2.11/generated/torch.nn.MultiheadAttention.html). In this API a true boolean padding-mask entry excludes a key. Do not assume all attention APIs use that convention.
