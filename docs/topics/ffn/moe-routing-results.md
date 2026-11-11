# MoE routing arithmetic results

Executed 2026-10-03 and closed. No training. [Guide](moe-routing-guide.md) | [Protocol](../../protocols/moe-routing.md) | [Source](../../../experiments/moe_routing.py) | [Raw evidence](../../../results/moe_routing.json).

## Method

24 CPU float64 cases: three seeds, token counts 5/9, expert counts 2/3 and capacity factors 0.5/1.5. Width 4, hidden width 6, bias-free tanh experts; final token is padding. Batched dispatch is compared with a separate token loop using explicit exponentials, scalar contractions and capacity counters. Both use PyTorch autograd; joint directional differences provide another derivative check.

Python 3.14.7; PyTorch 2.11.0+cu128, CPU only, one thread. Source/protocol hashes and 277 unchanged pre-existing JSON hashes are recorded in the raw report. Syntax compilation passed. No Pylance tools were available; runtime evidence uses the explicitly invoked project interpreter.

## Findings

| Check | Measured result | Meaning |
|---|---:|---|
| Maximum output disagreement | 2.78e-17 | Dispatch matches token equations |
| Maximum input/router/expert gradient disagreement | 1.39e-17 | The two computations agree on the tested gradients |
| Maximum directional difference error | 2.88e-12 | Below the frozen 1e-7 tolerance; all selections stayed fixed |
| Padding perturbation | Zero output change in all 24 cases | Masked tokens do not consume slots or affect outputs here |
| Ties and empty pool | Passed | Lowest index wins; an all-padding branch returns zero |
| Overflow residual error | Zero | Skipping the expert leaves x unchanged |
| Unselected expert gradient | Zero in the scalar fixture | That expert receives no output-loss signal from this token |
| Router gradient magnitude | 0.2398875 in the scalar fixture | Selected probability retains a router gradient |

The deterministic fixture has one padding token followed by two identical valid tokens. Both select expert zero; capacity is one. Branch outputs are `[0, 2.0124297215, 0]`.

| Intentional error | Maximum observed change |
|---|---:|
| Select least-probable expert | 1.437855 output |
| Omit probability multiplier | 0.272353 output |
| Ignore capacity | 2.012430 output |
| Let padding consume a slot | 2.012430 output |
| Detach probability | 0.239888 router gradient |

All five exceed the frozen 1e-6 detection threshold. Parameter counts are 104/156 stored versus 56/60 used by one accepted token for two/three experts.

## Conclusion and boundary

Top-one dispatch, probability weighting and declared capacity handling match the independent local equations. This supports the arithmetic contract. It does not establish specialization, successful training, balanced expert usage, language quality or speed. No balancing objective, distributed implementation, routing noise or selection-boundary derivative was tested.

The comparison to [Switch Transformers](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf) concerns routing structure only; the fixture is not a paper reproduction. Stop here. No training or follow-up is automatically selected.
