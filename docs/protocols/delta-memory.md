# Key-directed delta-memory audit

Frozen2026-10-03 before evaluation. No training. Reference: Schlag et al. (2021), delta-update construction; Yang et al., Gated Delta Networks v3 (2025), section2.2. Audit the plain delta rule only, not a complete model or gated-delta combination.

Column-key convention: S has shape m*d_v. old=S^T k; S_new=S+beta*k*(v-old)^T; read y=S_new^T q. Equivalently S_new=(I-beta*k*k^T)S+beta*k*v^T. Initial S=0; inclusive current write. No softmax, positive feature map, denominator z, decay gate, padding, positions or learned projections. Normalize supplied nonzero raw keys by their L2 norm; reject zero keys. beta=sigmoid(a). Q/V supplied directly.

- Seeds2663/2671/2683; lengths3/7/17; (m,d_v)=(3,2)/(5,4); B2,H2,CPUfloat64,one thread:18 cases.
- Compare correction recurrence with explicit ordered transition-product expansion and two chunks split after token1. Check outputs/final state and Q/raw-K/V/beta-logit gradients <=1e-10; joint normalized central finite difference h1e-6 error<=1e-7; future perturbation invariance<=1e-10. Include gradient through key normalization. Zero raw keys rejected.
- Five negative controls, each effect>1e-6: omit old-value subtraction, reverse correction sign, reverse noncommuting transition order, detach carried state (gradient effect), read before current write. Deterministic fixtures with nonparallel keys for transition order.
- Exact overwrite witness: initial unit-key value2, new value8; beta0/.5/1 gives2/5/8 and old-value derivative1/.5/0. Boundary beta0/1 supplied directly, not claimed from finite sigmoid logits.
- Orthogonal keys A=(1,0),B=(0,1), writesA2,B5,A8 with beta1: final reads8,5. Correlated B=(.6,.8), prior S=(2,4.75)^T gives A2,B5; beta1 update A8 changes B by3.6 to8.6. Verify general interference delta read=beta*(q^T k)*(v-S^T k) on random states too.
- Key-norm witness: raw k=2 in dimension1, beta1, target2, S0=0: repeated unnormalized reads8,-16,56. Unit k yields2 throughout. Verify transition eigenvalues1-beta*||k||^2 along k and1 on perpendicular directions. This is a local homogeneous-state sensitivity, not a proof of full-network stability.
- Stop after matrix, controls and concise educational report. No trained quality, GPU speed, architectural superiority or cause attribution to earlier retrieval failures. Preserve all earlier artifacts; no automatic extension.
