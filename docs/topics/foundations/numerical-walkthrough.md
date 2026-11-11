# One complete transformer forward pass

This example connects the eight mathematical probes into a **fixed-weight, one-block decoder**. It is not trained. All numbers below come from [the executable](../../../experiments/transformer_walkthrough.py); [the JSON trace](../../../results/transformer_walkthrough.json) contains every weight and intermediate tensor. Displayed numbers are rounded.

## Configuration and path

Vocabulary: A=0, B=1, C=2. Sequence: `A B C A`. Inputs are `A B C`; targets are `B C A`. The model must predict the next token at each input position.

```text
IDs [3]
 -> embeddings + positions [3,4]
 -> LayerNorm [3,4]
 -> two causal attention heads [3,2] each
 -> concatenate and output projection [3,4]
 -> add original input [3,4]
 -> LayerNorm -> linear [3,3] -> ReLU -> linear [3,4]
 -> add attention residual [3,4]
 -> final LayerNorm [3,4]
 -> vocabulary projection [3,3]
 -> probabilities [3,3] -> next-token cross-entropy
```

Batch size 1; model width 4; two heads with key/value width 2; FFN width 3. No padding, dropout or biases. LayerNorm uses gamma=1, beta=0, epsilon=1e-5. These simplifications are teaching choices, not recommendations for training. The architecture follows our [pre-norm baseline definition](architecture.md), not the original paper's full encoder-decoder.

## 1. Embeddings and position

An embedding lookup selects a row by token ID:

```text
E[A] = [1,0,-1,0]
E[B] = [0,1,0,-1]
E[C] = [1,1, 0,0]
```

At position i, add `[sin(i), cos(i), sin(i/100), cos(i/100)]`. This is the width-4 sinusoidal formula. Unlike the original transformer, this fixture does not multiply embeddings by sqrt(model width).

The resulting X is:

```text
A at 0: [1.0000, 1.0000,-1.0000, 1.0000]
B at 1: [0.8415, 1.5403, 0.0100,-0.00005]
C at 2: [1.9093, 0.5839, 0.0200, 0.9998]
```

Token identity and position now both affect later computations.

## 2. Normalize each token's features

For a row x, compute mean mu and variance sigma² across its four features:

\[
\operatorname{LN}(x)=(x-\mu)/\sqrt{\sigma^2+10^{-5}}.
\]

The first row has mean 0.5 and variance 0.75, giving `[0.5773,0.5773,-1.7320,0.5773]`. No token is mixed with another during normalization.

## 3. Two attention heads

Head 1 selects normalized features 1-2; head 2 selects features 3-4. Within each head, Q, K and V share that selector in this fixture. General transformers can learn different projections for all three.

For head 1:

```text
Q = K = V =
[[0.5773, 0.5773],
 [0.3791, 1.4670],
 [1.4956,-0.4270]]
```

Compute S=QK^T/sqrt(2), mask future positions, then apply row softmax. At input B (row 1), scores are `[0.7537,1.6233,-infinity]`, giving weights `[0.2953,0.7047,0]`.

Thus its head-1 output is:

\[
0.2953[0.5773,0.5773]+0.7047[0.3791,1.4670]
\approx[0.4377,1.2043].
\]

The complete attention matrices are:

```text
Head 1                    Head 2
[[1,      0,      0],     [[1,      0,      0],
 [0.2953, 0.7047, 0],      [0.3860, 0.6140, 0],
 [0.1925, 0.1193, 0.6883]] [0.4943, 0.1995, 0.3062]]
```

The heads weight positions differently because they compare different features. This does not imply they represent human concepts.

## 4. Concatenate, project, and add the residual

Concatenate both head outputs. At B this gives `[0.4377,1.2043,-1.2305,-0.3487]`. The output projection W_O is the identity matrix here, so it leaves this vector unchanged. A general W_O mixes head features.

Add it to the original X row:

\[
U=X+\operatorname{MHA}(\operatorname{LN}(X)).
\]

At B, U is `[1.2791,2.7446,-1.2205,-0.3487]`. The residual retains an additive path from the input.

## 5. Feed-forward computation and second residual

Normalize U, then compute F=ReLU(LN(U)W_1)W_2. ReLU replaces negative entries with zero. The fixture's matrices are:

```text
W1 = [[1, 0,-1],      W2 = [[0.2, 0,  -0.2, 0],
      [0, 1, 0],            [0,   0.2, 0,  -0.2],
      [1, 0, 1],            [0.1,-0.1, 0.1,-0.1]]
      [0,-1, 0]]
```

At B:

```text
LN(U)      = [ 0.4371, 1.3995,-1.2046,-0.6320]
LN(U) W1   = [-0.7675, 2.0315,-1.6416]
ReLU       = [ 0,      2.0315, 0]
F          = [ 0,      0.4063, 0,     -0.4063]
U + F      = [ 1.2791, 3.1509,-1.2205,-0.7550]
```

The same FFN acts independently on every token. It mixes features, whereas attention mixes information across tokens. For this fixture, ReLU zeros the entire hidden vector at A and C; that is a property of these chosen weights and inputs, not a general rule.

## 6. Vocabulary prediction

Apply final LayerNorm and multiply by the vocabulary matrix:

```text
WU = [[ 1, 0,-1],
      [ 0, 1,-1],
      [-1, 0, 1],
      [ 0,-1, 1]]
```

At B, final normalized features are `[0.3824,1.4577,-1.0538,-0.7863]`, producing logits `[1.4361,2.2441,-3.6802]`. Logits are unnormalized scores; softmax converts them to vocabulary probabilities.

| Input position | P(A) | P(B) | P(C) | Correct next token |
|---|---:|---:|---:|---|
| A at 0 | 0.90156 | 0.08954 | 0.00889 | B |
| B at 1 | 0.30776 | 0.69039 | 0.00185 | C |
| C at 2 | 0.95472 | 0.04028 | 0.00501 | A |

Vocabulary softmax selects between tokens. Attention softmax selects between visible positions. They use the same function for different purposes.

## 7. Loss and the boundary to learning

For target y_i, loss_i=-log(P_i(y_i)). The three losses are `2.41302, 6.29480, 0.04634`; their mean is **2.91805 nats**. A uniform three-token predictor has loss log(3)=1.09861, so our chosen weights perform worse on this example. They were selected to expose arithmetic, not to predict well.

For mean loss over three positions, the derivative of a logit is:

\[
\frac{\partial L}{\partial z_{ij}}=
\frac{p_{ij}-\mathbf{1}[j=y_i]}{3}.
\]

At B, the target is C, so the logit gradient is approximately `[0.10259,0.23013,-0.33272]`. Gradient descent directly on these logits would lower A/B scores and raise C's. Training model weights requires propagating gradients through every earlier operation; that backward pass and parameter update are not implemented here.

## Executed checks and remaining work

Run `python experiments/transformer_walkthrough.py` from the repository root.

Six checks passed: future edits leave earlier logits unchanged; separate prefix evaluation agrees; removing the mask exposes future influence (maximum earlier-logit change 0.12095); uniform logits give log(3) loss; constant rows normalize to zero; vocabulary probabilities sum to one.

The first two errors were exactly zero in this run. These checks reuse existing arithmetic, so they are not independent library verification. No padding, backward pass, training, or GPU performance was tested.

Next: audit the foundational source claims and independently verify a trainable baseline's forward and backward computations before controlled training. Novel architecture proposals remain deferred.
