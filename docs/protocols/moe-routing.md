# MoE routing arithmetic protocol

Frozen before execution, 2026-10-03. No training or GPU benchmark.

- CPU float64, one thread. Seeds 2741/2749/2753; tokens 5/9; experts 2/3; capacity factors 0.5/1.5: 24 cases. Input/output width 4, expert hidden width 6, bias-free tanh experts. Tanh is a smooth local fixture, not a Switch model reproduction.
- Last token is padding. Capacity is ceil(factor * valid tokens / experts), shared over the flattened token list. Padding uses no slot. Highest logit wins; exact ties choose the lowest expert index. First tokens in input order fill slots. Overflow and padding produce zero branch output; residual remains x.
- Compare batched expert dispatch against an independent token loop using explicit normalized exponentials and counters. Check outputs and input/router/both expert-weight gradients of a fixed random linear output probe. Use a joint unit direction, central difference step 1e-6; verify both perturbations preserve routing and capacity decisions.
- Thresholds: output/gradient <=1e-10; finite difference <=1e-7. Finite random cases do not prove a global derivative at selection boundaries.
- Deterministic fixtures: ties, zero valid tokens, padding invariance, overflow residual identity, unselected expert gradients, router gradient with a fixed selected expert. Detect wrong expert selection, omitted gate weight, ignored capacity, padding consuming slots and detached gate gradients (>1e-6 effect).
- Count all expert plus router parameters, and the parameters used by one accepted token. Counts are not latency or allocated memory measurements.
- Save source/protocol hashes, environment, all cases and fixtures. Preserve existing result JSON hashes. Stop after these checks, report failures without changing thresholds. No tuning, specialization, quality or speed claim.

Source: [Switch Transformers](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf), sections 2.1–2.2 and routing pseudocode. Local rounding, ties, padding and smooth activation are declared conventions. No auxiliary balancing loss, noise, distributed dispatch or full model is included.
