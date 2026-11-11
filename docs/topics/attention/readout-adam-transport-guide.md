# Why Adam can keep expanding an already-correct classifier

The [previous diagnostic](readout-scale-pressure-guide.md) measured the loss's incentive to sharpen scores. This study reconstructs actual optimizer calculations during two original four-load cycles per seed.

## Gradient versus movement

Let C be the class-centered classifier weights and bias. The radial gradient g=<gradient,C> measures scale pressure. Negative g favors expansion under scalar gradient descent.

Adam's actual step also changes direction. Its radial coefficient is

$$\rho=\frac{\langle C,\Delta C\rangle}{\|C\|^2},\qquad \Delta C=\rho C+\Delta C_{perpendicular}.$$

Positive rho is an expansion component. The exact norm change also includes the perpendicular component; it is not exactly rho. We measure both. This is classifier geometry, not a change in attention-head count.

## Where the step comes from

Adam stores a first moment m and a second moment v. For current gradient G,

$$m_t=\beta_1m_{t-1}+(1-\beta_1)G,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)G^2.$$

Using $d_t=\sqrt{v_t/(1-\beta_2^t)}+\epsilon$, split the step into

$$\Delta A_{stored}=\frac{-\eta\beta_1m_{t-1}}{(1-\beta_1^t)d_t},\qquad
\Delta A_{current}=\frac{-\eta(1-\beta_1)G}{(1-\beta_1^t)d_t}.$$

| Term | Meaning |
|---|---|
| Stored term | Earlier gradients carried by the first moment |
| Current term | This minibatch's gradient, with Adam's coordinate scaling |
| Their sum | The reconstructed native Adam step, within floating-point tolerance |

When an easy batch has almost zero gradient, stored m can still move the parameters. Its earlier signals need not come from that same load because all four loads share one optimizer.

Both terms use the same updated denominator. Removing the stored term here is algebra; it does not simulate a momentum reset, which changes subsequent learning. The denominator itself contains gradient history. Preconditioning can also make the current term's radial sign differ from plain gradient descent.

The [results](readout-adam-transport-results.md) show expansion in all 16 replayed steps, dominated by the stored term. Late load-2 batches illustrate movement despite unresolved current pressure. The head also moves perpendicularly, so this is not pure scale training.

Sources inspected: [Kingma and Ba, Algorithm 1](https://arxiv.org/pdf/1412.6980) and [PyTorch 2.11 Adam](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html), moment updates, bias correction and coordinate denominator. Installed torch's single-tensor implementation was checked too. Radial projections and numerator splitting above are local derivations. The papers do not predict these local measurements or establish a training repair.
