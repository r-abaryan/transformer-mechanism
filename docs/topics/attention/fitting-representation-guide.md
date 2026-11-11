# Routing versus the effective decoder

The [diagnostic](fitting-representation-results.md) asks which part of the saved L-BFGS computation carries its fitting advantage. It swaps existing parameters; it does not train a new model.

## Two parts of one computation

| Part | Parameters | Role |
|---|---|---|
| Routing R | Normalized keys/tied queries and beta | Determines how writes interact and how a query reads them |
| Effective decoder D | Value embeddings, classifier weights and bias | Maps the retrieved mixture to six class logits |

Keeping values and classifier together avoids breaking their private coordinate system. This grouping follows an exact local derivation.

Our memory update is $S_t=(I-\beta k_tk_t^T)S_{t-1}+\beta k_tv_t^T$. Write the six value embeddings as rows of $V\in\mathbb R^{6\times8}$. If an event's value ID is $j$, then $v_t^T=e_j^TV$, where $e_j$ is its one-hot basis vector. Starting at zero, induction gives

$$S_t=C_tV,\qquad C_t=(I-\beta k_tk_t^T)C_{t-1}+\beta k_te_j^T.$$

For a query, let $c=C_T^Tq$. Its six components depend on routing and the sequence's value IDs. Then

$$z=W S_T^Tq+b=(WV^T)c+b=Ac+b.$$

$A$ is a 6 by 6 effective class map. This rewrites the existing model exactly; it adds no capacity or new architecture. Components of $c$ can be signed. They are not attention probabilities.

```mermaid
flowchart LR
  H[History and query] --> R[Routing: keys and beta]
  R --> C[Six value-basis coefficients]
  C --> D[Effective decoder: A and bias]
  D --> Z[Class logits]
```

## What the swaps test

| Routing donor | Decoder donor | Meaning |
|---|---|---|
| Negative gradient | Negative gradient | Native comparison endpoint |
| L-BFGS | Negative gradient | Does better routing carry the error improvement? |
| Negative gradient | L-BFGS | Does the decoder carry it? |
| L-BFGS | L-BFGS | Native successful endpoint |

We test both conditional effects, since routing and decoding may interact. A swap can fail because the two parts adapted together. Endpoint effects do not establish what gradient history caused that adaptation.

Large decoder norms do not prove useful decoding. Mean key overlap does not describe every important key relationship. Positive uniform scaling of all logits preserves their ranking, while a changed class map can change predictions. Use prediction counts, CE and complete maps to assess the swaps.

Source inspected: [Yang et al. (2024), section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2) supplies the delta write/read context, with memory orientation transposed here. Our scalar beta and tied queries differ from full DeltaNet. The value-basis factorization and swap design are local derivations, checked against native outputs; the paper does not predict our scores.
