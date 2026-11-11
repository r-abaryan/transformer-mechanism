# MoE load-balancing protocol

Frozen 2026-10-03 before execution. CPU float64, one thread. No training.

Use Switch auxiliary loss L = alpha E sum_i f_i P_i. f counts top-one choices before capacity filtering; P averages probabilities over valid tokens. Treat f as discrete, with no gradient. Lowest index wins ties. Padding is excluded from both means; reject an empty valid pool.

24 random cases: seeds 2777/2789/2791, token counts 5/9, experts 2/3, alpha 0.01/1. Width 4, logits xW, final token padding. Compare vector loss/autograd with explicit per-token exponentials and counters, and analytical logit/input/router gradients. Joint unit directional differences use step 1e-6 and must preserve assignments. Tolerances: loss/gradients 1e-10, finite differences 1e-7.

Fixtures, supplied probabilities via log probabilities: balanced [.8,.2],[.2,.8] repeated twice; concentrated [.9,.1] four times; uniform [.5,.5] four times; uniform mean with unequal choices [.6,.4] three times and [.2,.8] once; below-one witness [.51,.49] three times and [.01,.99] once. Alpha 1. Capacity factor 1, ceil(T/E), first-arrival policy. Compare independent counts with existing audited dispatch; no expert training. Check balanced f gives zero auxiliary gradient; empty rejection; padding invariance; alpha scaling.

Detect five wrong variants: omit E, substitute soft P for f, use post-capacity counts divided by original T, include padding, detach P. Each must change loss or gradient by >1e-6 on a declared nonuniform fixture. Report failures without changing thresholds. Preserve previous source/reports and JSON hashes. Stop after numerical audit and short educational guide; no optimization, specialization or quality claims.

Primary source: [Switch Transformers](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf), section 2.2 equations 4-6 and pre-capacity auxiliary-loss pseudocode. Rounding, padding and tie policies are local conventions. Counterexamples test the loss as a diagnostic, not the paper's empirical training results.
