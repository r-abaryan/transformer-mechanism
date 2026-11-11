# MoE load-balancing results

Executed 2026-10-03 and closed. No training. [Guide](moe-balancing-guide.md) | [Frozen protocol](../../protocols/moe-balancing.md) | [Source](../../../experiments/moe_balancing.py) | [Raw evidence](../../../results/moe_balancing.json).

## Method and numerical checks

24 CPU float64 cases: seeds 2777/2789/2791, token counts 5/9, experts 2/3, alpha .01/1, width 4. Last token is padding. Compare vectorized loss/autograd with independent token-loop exponentials and counts, analytical logit/input/router gradients, and joint directional finite differences. Both computational forms use PyTorch; analytical derivatives supply a separate check.

| Check | Maximum error | Frozen limit |
|---|---:|---:|
| Loss | 2.22e-16 | 1e-10 |
| Token-loop gradients | 1.39e-16 | 1e-10 |
| Analytical gradients | 1.11e-16 | 1e-10 |
| Central directional difference | 2.37e-10 | 1e-7 |
| Padding perturbation / alpha doubling | 0 / 0 | Exact |

All selections remain unchanged under finite-difference perturbations. Empty valid pools are rejected. Python 3.14.7, PyTorch 2.11.0+cu128, CPU only, one thread. Syntax compilation passed. Raw evidence stores source, dependency and protocol hashes, plus 279 preserved earlier JSON hashes.

## Controlled witnesses

Alpha 1, four valid tokens, two experts, capacity two each. Lowest-index tie rule; first arrivals fill slots. Independent counts agree with the previously audited dispatch implementation.

| Fixture | Assignment fractions f | Mean probabilities P | Loss | Overflow | Max fixed-choice gradient |
|---|---|---|---:|---:|---:|
| Balanced | .5 / .5 | .5 / .5 | 1 | 0 | 0 |
| Concentrated | 1 / 0 | .9 / .1 | 1.8 | 2 | .045 |
| Uniform tie | 1 / 0 | .5 / .5 | 1 | 2 | .125 |
| Uniform mean without ties | .75 / .25 | .5 / .5 | 1 | 1 | .06 |
| Below-one witness | .75 / .25 | .385 / .615 | .885 | 1 | .062475 |

The tie row is an implementation convention, not a finite-difference claim at a boundary. The no-tie witnesses already establish that uniform P does not ensure uniform f, and that loss 1 is not a global lower bound.

## Error controls

Use probabilities [.8,.2], [.7,.3], [.6,.4], [.1,.9], then masked padding [.01,.99].

| Incorrect variant | Loss change | Max gradient change |
|---|---:|---:|
| Omit expert-count multiplier | .525 | .03 |
| Replace hard f with soft P | .04 | .036 |
| Use post-capacity counts / original T | .275 | .03 |
| Include padding | .0732 | .0408 |
| Detach P | 0 | .06 |

All five exceed the 1e-6 detection rule. The detached variant shows why checking only the forward loss is insufficient.

## Paper comparison and conclusion

[Switch Transformers, section 2.2 and routing pseudocode](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf) defines the auxiliary dot product and computes it before capacity filtering. Our equations and gradients agree with that definition. Its uniform-minimization description requires qualification: the explicit no-tie .885 witness is below the uniform value 1. This does not challenge its measured language-model results or establish that balancing is ineffective in training.

The audit verifies arithmetic and diagnostic limits. It does not measure optimization, expert specialization, task accuracy or speed. The previous routing study remains unchanged. Stop here; no training or automatic extension selected.
