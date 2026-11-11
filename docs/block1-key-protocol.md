# Block-1 contributions to block-2 value-position K

Fixed before execution. Five frozen models; same 5,120 inspected six-derived crops each. No training. Final Q is compact. Block-2 V, non-value K and recipient residual/FFN execution stay shifted. Replace only K at value slots 1/3/5 after constructing candidate block-1 outputs.

Block 1: u=x+a, f=FFN(LN2(u)), y=u+f. Cache compact and shifted x,a,f. First evaluate every binary choice of compact versus shifted x,a,f in y (eight cells). These are frozen-component sums, not ordinary live block execution. Then evaluate four choices of x/a with f recomputed from the mixed u. This distinguishes direct donor FFN effects from the FFN response to a changed input.

For each candidate y, recompute block-2 LN1 and project K, then insert its value-position K into otherwise shifted attention with compact final Q. Both heads. Never copy candidate Q/V or block-2 residual. Primary sufficiency criterion per condition: every seed >=90%. Report all seeds, accuracy/loss and paired rescue/harm relative to all-shifted K with compact Q. No additive causal percentages.

Verify cached y=x+a+f equals block-2 input (<1e-10); recomputed FFN at unmixed endpoints matches its cache; frozen 000 and 111 reproduce preceding forward-none/all metrics (<1e-10); recomputed endpoints match frozen endpoints; labels, finite outputs, unchanged checkpoints and hashes. Layer normalization is nonlinear, so the K change is not an additive decomposition. Mixed states are artificial, data reused and crops dependent. This tests sufficient construction paths, not unique error origin or necessity.
