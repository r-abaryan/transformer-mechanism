# Scalar gated-memory arithmetic

Frozen2026-10-03 before evaluation. No training. Audit a deliberately simple normalized positive-kernel memory, not a reproduction of the full GLA architecture.

S_t=g_t*S_(t-1)+phi(k_t)*v_t^T; z_t=g_t*z_(t-1)+phi(k_t); y_t=phi(q_t)^T*S_t/(phi(q_t)^T*z_t). Here phi=ELU+1, g=sigmoid(a), one scalar per batch/head/token. Initial state zero; include current write, no write gate. Reject nonpositive/nonfinite denominators. No masks, projections, positional encoding, GPU or learned gate network.

Independent expanded weights: w_tj=phi(q_t)^T phi(k_j) product(g_l,l=j+1..t), j<=t. Normalize weights over j. Products exclude the write's own gate. No division by cumulative products, so exact zero gates remain testable.

- Seeds2633/2647/2657; lengths3/7/17; widths(m,dv)=(3,2)/(5,4); B2,H2,CPUfloat64,one thread:18 random-normal Q/K/V/gate-logit cases.
- Compare recurrence, explicit expanded weights, and two chunks split after token1 carrying state without detach. Outputs, final state, and all Q/K/V/gate-logit gradients <=1e-10. Joint normalized directional finite difference h1e-6, error<=1e-7. Future perturbation invariance<=1e-10. Gate1 recovers ungated attention; direct gate0 resets history. First gate-logit gradient is exactly zero under zero initial state.
- Intentional controls: omit decay; decay S but not z; include each write's own gate; detach chunk state; omit causal restriction. Each deterministic fixture must differ>1e-6 in output or old-value gradient as relevant.
- Scalar overwrite witness: q=k=0, values2 then8, gates1/.5 gives output6, ungated5; gate0 gives8. Derivative of final output with respect to old value: g/(1+g). Evaluate g=0,.5,1.
- Interference witness with explicitly supplied nonnegative features: keysA=(1,0),B=(0,1), values2,5,8 for A,B,A, gates1/.5/.5. Final global gate0 instead updates A to8 but erases B; B read rejected. Gates1/.5/.5 yield A6.8 and B5. This ideal orthogonal-feature fixture is outside strictly positive finite ELU+1 features and must be labeled.
- State collision at gates1/.5: scalar keys0/1, values(0,1) versus(4,0), both S=2,z=2.5. Same query reads. Gating does not guarantee unique history storage.
- Stop after arithmetic matrix, controls, witnesses and concise educational report. No speed, trained quality, automatic extension or failure-cause claim about earlier retrieval.

Source: Yang et al., [GLA](https://arxiv.org/html/2312.06635v6), sections2.1 and4.1. Paper uses a richer gate and different normalization choices; this local scalar normalized construction isolates forgetting only.
