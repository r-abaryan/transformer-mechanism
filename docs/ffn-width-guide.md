# FFN width: what32 versus64 changes

**Question:** does extra FFN hidden capacity improve recall enough to justify its cost? RoPE, ReLU, model width, heads and depth stay fixed. See the [frozen protocol](ffn-width-protocol.md).

## Where the extra units go

```mermaid
flowchart LR
    A["Token representation:16"] --> B[LayerNorm]
    B --> C["Linear:16 to f"]
    C --> D["ReLU: f hidden units"]
    D --> E["Linear: f to16"]
    E --> F[Residual addition]
    A --> F
```

Only `f` changes from 32 to 64, in both blocks. Each token gets the same learned FFN transformation. Context has already entered through attention.

| Change | What it changes |
|---|---|
| FFN hidden width, this test | Number of intermediate nonlinear features per token. Input/output remain 16-dimensional. |
| Model width | Size of the residual representation and associated projections. |
| Head count | Number of attention heads; at fixed model width it also changes head width. |
| Depth | Number of sequential attention/FFN processing rounds. |

The old [depth/width comparison](topics/training/depth-comparison-results.md) changed several dimensions together. It did not isolate this FFN choice.

## Essential math

For normalized token vector z:

`FFN(z) = W2*ReLU(W1*z+b1)+b2`.

Shapes: `W1: f x16`, `W2:16 x f`. Equivalently, the output sums f learned feature contributions:

`FFN(z) = b2 + sum_j W2[:,j]*max(0,W1[j,:]*z+b1[j])`.

More terms allow more nonlinear feature combinations. Width 64 can reproduce width 32 by copying 32 units and zeroing the remaining contributions. This is a capacity statement. Standard random initialization does not perform that copying, and optimization need not exploit the extra capacity.

Within one ReLU activation region, `J_FFN = W2*D*W1`, where D marks active units. Its rank is at most `min(16,f,number_active)`. Both widths already exceed 16. Widening therefore does not double the maximum local input/output rank; it can change the available activation regions and features.

## Cost, explicitly

One FFN has `16*f + f + f*16 +16 =33*f+16` parameters, including biases.

| Quantity | f=32 | f=64 |
|---|---:|---:|
| Parameters per FFN | 1,072 | 2,128 |
| Whole model, two blocks | 5,088 | 7,200 |
| Projection multiply-accumulates per token per block | 1,024 | 2,048 |
| Hidden coordinates per token | 32 | 64 |

That is 2,112 more parameters (+41.51%) and twice the FFN projection arithmetic. Whole-model time and memory need not double. Equal training tokens are not equal compute.

## How to interpret the test

Five paired seeds share non-FFN initial tensors and minibatches. Different FFN shapes require different tensors; they are not identical initial functions. Both use 6,000 updates. This tests a practical width replacement under one initialization/optimizer recipe, not an optimization-free comparison of capacity.

Three-pair data is reused and labelled diagnostic because no unused families remain. Four/six-pair families are fresh relative to the recorded earlier studies. The six-pair keep rule requires all wide seeds >=90%, mean gain >=2 points, no worsening seed and preserved short-length performance. Stop after ten runs and audit.

A gain would support paying for more FFN capacity here. A failure would not prove that 32 is universally sufficient or that wider FFNs cannot learn the task.

## Primary source

[Vaswani et al., Attention Is All You Need, v7, section3.3](https://arxiv.org/html/1706.03762v7) defines the two-projection, position-wise ReLU FFN. Our shape/count equations follow directly from that structure. Our small pre-norm RoPE recall experiment is not a reproduction of its translation benchmarks.


[Completed comparison](topics/ffn/ffn-width-results.md): width 64 gains 5.59 points on average, but one seed worsens and falls below 90%. The fixed adoption rule fails; comparison closed.


For the concrete meaning of pairs and attention vectors, see the [saved-model Q/K/V walkthrough](topics/retrieval/trained-recall-walkthrough.md).
