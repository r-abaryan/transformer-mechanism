# Architecture and mathematical tree

Scope: dense text transformers. This map separates the original encoder-decoder [P01](https://arxiv.org/html/1706.03762v7) from a proposed small decoder-only research baseline. Equations below define the baseline and derive properties of those definitions; they do not describe every deployed model. Dropout is omitted in the mathematical probes.

```text
Sequence model
|-- Input contract
|   |-- text -> token IDs (tokenizer must be specified)
|   |-- shifted input / next-token targets
|   `-- padding and visibility masks
|-- Representation
|   |-- embedding lookup
|   `-- position signal (additive initially; rotary is an alternative)
|-- Repeated computation blocks
|   |-- normalization and residual path
|   |-- token mixing: multi-head attention
|   |   |-- learned Q, K, V projections
|   |   |-- pairwise dot-product scores and scaling
|   |   |-- visibility mask
|   |   |-- row-wise softmax
|   |   `-- weighted values -> head concatenation -> output projection
|   `-- per-token feature mixing: feed-forward network
|-- Output
|   |-- final normalization for our pre-norm baseline
|   `-- vocabulary logits -> next-token distribution
|-- Learning
|   |-- cross-entropy -> chain-rule gradients
|   `-- optimizer updates parameters
`-- Evaluation
    |-- held-out prediction and controlled tasks
    |-- intervention-based mechanism tests
    `-- compute, memory, latency, and uncertainty
```

## Shapes and input

Let batch size be B, sequence length n, vocabulary size U, model width d, heads h, key width k, value width v, and hidden feed-forward width f. For clarity use a single sequence; batching adds a leading B dimension. A token ID is an index, not a scalar semantic measurement.

For token IDs t_i, embedding table E in R^(U x d), and additive position vectors p_i, define X_i = E[t_i] + p_i. A transparent initial position choice is p_(i,2j) = sin(i / 10000^(2j/d)) and p_(i,2j+1) = cos(i / 10000^(2j/d)), for even d. This is a baseline choice, not a proof of length generalization. Tokenization and its data must be fixed before comparing trained models.

## A single attention head

Define learned W_Q and W_K in R^(d x k), W_V in R^(d x v):

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
\quad S=QK^T/\sqrt{k}+M,\quad A_{ij}=\frac{e^{S_{ij}}}{\sum_r e^{S_{ir}}},\quad Y=AV.
\]

Shapes: Q,K are n x k, V is n x v, S,A are n x n, Y is n x v. Q defines a query, K provides comparison features, V provides content to aggregate. These are computational roles, not guaranteed human-readable meanings. Bias terms are optional and must be documented.

For a causal model, M_ij = 0 for j <= i and negative infinity otherwise. Padding also removes invalid keys. Each row must have at least one visible key. Compute softmax as exp(S_ij - max_j S_ij) divided by its row sum to prevent overflow. An all-masked row is undefined, not a uniform distribution.

For h heads, concatenate Y_1,...,Y_h along features and multiply by W_O in R^(hv x d). A head is not necessarily one concept. Cross-attention instead projects queries from decoder states and keys/values from encoder states; its score matrix is n_target x n_source.

## Blocks, normalization, output, and learning

Our proposed pre-norm decoder block is:

\[
U=X+\operatorname{MHA}(\operatorname{LN}(X)),\qquad
X'=U+\operatorname{FFN}(\operatorname{LN}(U)).
\]

For each token row x, LN(x) = gamma * (x - mean(x)) / sqrt(mean((x - mean(x))²) + epsilon) + beta. Statistics are over features, not tokens. Gamma and beta are learned feature vectors. Epsilon > 0 stabilizes the denominator. A simple baseline FFN is max(0, xW_1+b_1)W_2+b_2 with widths d -> f -> d. A residual adds an alternate identity path; it does not ensure perfect gradient flow.

The original model uses post-norm sublayers, encoder self-attention, and decoder masked self-attention plus cross-attention. Pre-norm changes optimization behavior and is a separate experimental choice, motivated for investigation by [P04](https://arxiv.org/abs/2002.04745).

After L blocks, H=LN(X_L); logits Z=HW_U+b, with W_U in R^(d x U). For next-token target y_i, define loss on N valid targets:

\[
\mathcal L=-\frac1N\sum_i\log \operatorname{softmax}(Z_i)_{y_i}.
\]

For one target, dL/dZ_j = probability_j - indicator(j=target). Chain rule propagates this signal into embeddings and all projections. A simple SGD step is theta <- theta - learning_rate * gradient; an Adam-family optimizer adds state and a different update rule. Teacher-forced training supplies known prefix tokens in parallel under a mask; ordinary autoregressive generation repeatedly chooses a token and extends the prefix. Target shifting errors can falsely inflate results.

## Local derivations and falsifiable boundaries

**Scaling.** Suppose q_l and k_l are independent, zero-mean, unit-variance variables, and products are independent across l. Then E[q_l k_l]=0, Var(q_l k_l)=1, and Var(sum_l q_l k_l)=k. Division by sqrt(k) makes variance one. Learned or correlated vectors need not satisfy these assumptions. This motivates scaling, not its universal optimality. Spike S01 checks the stated random model.

**Softmax sensitivity.** For a=softmax(s), J_ij=a_i(delta_ij-a_j). As one probability approaches one, entries approach zero. Saturation can reduce sensitivity to scores. It does not prove that every unscaled trained model fails. S04 compares this Jacobian against central finite differences.

**Permutation.** For permutation matrix P and no position-dependent terms or fixed causal mask, scores transform as P S P^T, row softmax transforms the same way, and attention(PX)=P attention(X). This is equivariance, not invariance. A fixed causal mask breaks the premise. S03 checks both the identity and a positional countercondition.

**Non-unique explanations.** Let scalar values V=(0,1,2)^T. Both attention rows a=(0.2,0.6,0.2) and b=(0.4,0.2,0.4) return value 1. They are strictly positive distributions and are realizable by softmax(log(a)) and softmax(log(b)). Thus distinct attention patterns can yield exactly the same head output. This existence proof says nothing about their frequency in a particular trained model. S05 implements it.

**Cost.** Direct QK^T and AV require O(n²(k+v)) work per head; storing A costs n² elements. Projections add O(ndh(k+v)) up to constant factors; the FFN adds O(ndf). Quadratic score storage is not mandatory: tiling and online normalization can avoid materializing A. [FlashAttention](https://arxiv.org/abs/2205.14135) exploits IO-aware exact attention. Our S06 scalar streaming identity is not its GPU implementation or a reproduction of its speed claims. KV caching stores previous keys/values; dense decoding still reads a growing prefix.

For the streaming identity, maintain a running maximum m, denominator l=sum_j exp(s_j-m), and numerator u=sum_j exp(s_j-m)v_j. For a new finite score s and value v, set m'=max(m,s), l'=exp(m-m')l+exp(s-m'), and u'=exp(m-m')u+exp(s-m')v. Start with m=-infinity, l=0, u=0. Induction preserves both sums; the final result u/l equals the dense softmax aggregate. This trades score storage for repeated accumulation without removing pairwise score computation.

**Associativity boundary.** Without softmax, (QK^T)V = Q(K^TV). Softmax(QK^T)V generally differs from Q(K^TV), so deleting softmax changes the model. S07 supplies both an equality check and a counterexample. Kernelized linear attention introduces its own feature map and normalization; it is not automatically exact softmax attention. See [P05](https://arxiv.org/abs/2006.16236).

The derivations establish narrow statements under explicit assumptions. The numerical probes check our implementation at selected inputs, not universal theorems or training outcomes.
