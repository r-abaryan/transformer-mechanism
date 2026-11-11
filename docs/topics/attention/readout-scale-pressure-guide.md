# Why training can reward sharper answers

The previous [scale control](readout-scale-control-guide.md) lowered held-out loss without changing answers. This diagnostic asks whether the training objective locally rewards expansion of that same classifier direction.

## The measured direction

Keep memory and the read vector fixed. Let C be the class-centered classifier weights and bias. Scale C by exp(lambda), so lambda=0 is the saved model. Increasing lambda makes its score differences sharper; it preserves the winner.

For true class y, probability p and saved scores z, our local derivative is

$$g_i=\left.\frac{\partial CE_i}{\partial\lambda}\right|_{0}=\sum_c p_{ic}(z_{ic}-z_{iy}).$$

| Derivative | Local meaning under scalar gradient descent |
|---|---|
| g<0 | Increasing scale lowers this answer's CE |
| g>0 | Reducing scale lowers this answer's CE |
| Near zero | This direction has little measured pressure |

If the answer is correct, z_y is a largest score. Every difference z_c-z_y is nonpositive, so g<=0. Training can keep rewarding sharper correct predictions after accuracy has stopped changing. Ties and saturated probabilities can yield zero or tiny derivatives.

For a wrong answer, some differences are positive and others negative. Their probability-weighted sum decides the sign. A wrong answer can also favor expansion. The [results](readout-scale-pressure-results.md) include this exception at load 16.

## Why the number of answers matters

A partition's mean derivative describes one typical answer within it. Its contribution to the whole pool is

$$G_{part}=\frac{N_{part}}{N}\,\operatorname{mean}_{part}(g).$$

Many weakly negative correct-answer derivatives can outweigh fewer strongly positive error derivatives. Do not balance the two partitions artificially. The actual recipe cycles equally through four loads, so the training-objective diagnostic averages the four pool derivatives equally.

The [measurements](readout-scale-pressure-results.md) support this local objective incentive at all ten saved snapshots in both seeds. They do not reconstruct Adam's update. Adam uses moments and coordinate-wise scaling, while memory, values and classifier direction also change during training. A causal repair would need a separate training comparison.

Sources inspected: [PyTorch 2.11 CrossEntropyLoss](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html), class-index CE and averaging; [Guo et al. 2017, section 4.2, equation 9](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), positive-temperature scaling with unchanged argmax. The derivative, correct-answer sign proof and contribution accounting above are local derivations. Neither source predicts our gradients; this is not calibration fitting.
