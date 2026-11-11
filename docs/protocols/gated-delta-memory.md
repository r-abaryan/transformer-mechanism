# Gated delta-memory audit

Frozen2026-10-03 before evaluation. No training. Source: [Yang et al., Gated Delta Networks v3](https://arxiv.org/html/2412.06464v3), section3.1 equation10, transposed into m-by-d_v state convention.

Decay first: P=alpha*S. Correct the decayed read: S_new=P+beta*k*(v-P^T*k)^T. Read after write, y=S_new^T*q. Equivalent transition alpha*(I-beta*k*k^T). Unit L2 keys from nonzero raw keys; alpha and beta are sigmoid logits. Zero initial state. No z, positive features, masking, positions, learned projections or GPU.

- Seeds2693/2699/2707; lengths3/7/17; widths(m,d_v)=(3,2)/(5,4); B2,H2 CPUfloat64 one thread:18 cases.
- Independent explicit ordered matrix-transition expansion versus recurrence and two chunks split after token1. Output/state and gradients through Q/raw-K/V/both gate logits<=1e-10. Joint normalized directional central difference h1e-6<=1e-7; future invariance<=1e-10.
- Boundaries each random case: alpha1 matches archived plain delta; beta0 yields zero from zero state; alpha0 equals beta*k*v^T writes with no old-state influence. First alpha-logit gradient zero. Direct0/1 boundary fixtures are not finite sigmoid outputs.
- Five error controls, each difference>1e-6: subtract undecayed old read, decay new write too, omit decay, detach state (gradient mismatch), read before current write.
- Initial orthogonal A2/B5, new A8: alpha1,beta1 gives8/5; alpha.5,beta1 gives8/2.5; alpha0,beta1 gives8/0; alpha.5,beta.5 gives4.5/2.5; alpha0,beta0 gives0/0. Check each final read and gradients to both old associations. Zero read after reset denotes lost memory, not a valid stored association.
- Check general change delta_y=(alpha-1)*S^T*q+beta*(q^T*k)*(v-alpha*S^T*k) on random states. Correlated q=(.6,.8), A=(1,0), initialS=(2,4.75)^T: old q read5 becomes6.8 at alpha.5,beta1 updating A8.
- Stop after checks, educational guide/results and four-rule synthesis. No training, tuning, model adoption, GPU performance or causal explanation for earlier retrieval errors. Preserve earlier artifacts.
