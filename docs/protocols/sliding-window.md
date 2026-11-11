# Sliding-window numerical protocol

Frozen2026-10-02 before execution. No training. Question: which arithmetic and information-access properties change when attention is restricted to a fixed local window?

- CPU float64, one thread, deterministic. Seeds2411/2417/2423; lengths5/9; radius r in0/1/2/n-1; bidirectional and causal:48 cases. B=2,H=2,Q/K width3,V width2. Random independent Q/K/V and scalar output probe. No padding, dropout, projections, global tokens or dilation.
- Visible keys: bidirectional abs(i-j)<=r; causal max(0,i-r)<=j<=i. Self is always visible. Compare dense masked softmax against explicit gathered keys with stable exponentiation. Output and all Q/K/V gradients must agree within1e-10; normalized joint directional difference with step1e-6 within1e-7.
- Full attention uses the corresponding causal policy without the local radius. At r=n-1 it must agree; elsewhere report the difference without requiring random cases to differ.
- Deterministic controls: uniform scores and a value outside a query's window expose omitted window masking; a future value exposes missing causal masking; a boundary value exposes an off-by-one radius. Effects must exceed1e-6. r=0 must return each own V.
- Reachability: n=9, r=0/1/2, depth1/2/3, both policies:18 graphs. Compare independent set propagation against Boolean adjacency powers and interval formula. A stack of uniform local averaging layers supplies a constructive witness: input/output gradients positive on reachable pairs and exactly zero outside. No claim that trained models use every available path.
- Count visible query/key pairs by row enumeration and closed formulas for n=5/9/128/1024/4096 and r=0/1/2/4 capped at n-1, both policies. Formula uses clipped radius; bidirectional n(2r+1)-r(r+1), causal n(r+1)-r(r+1)/2. Compare full counts n^2 and n(n+1)/2. No speed or measured memory claim.
- Save per-case results and provenance, preserve previous artifacts, retain any failed attempts. Stop after this matrix, controls and documentation. No training, GPU benchmark or automatic extension follows.
