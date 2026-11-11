# Global-token mathematical audit

Frozen2026-10-02 before evaluation. No training. Global means selected existing positions have global query and key connections; no extra token is appended.

Visibility: `(abs(i-j)<=r OR i in G OR j in G) AND (j<=i if causal)`. Self is always visible. Shared supplied Q/K/V, no global-specific projections, padding, dropout or dilation. This is a connectivity audit, not a full Longformer implementation.

- Numerical matrix: seeds2473/2477/2503; n5/9; r0/2; G empty, first, middle, first+last; causal/bidirectional.96 cases. B2,H2,Q/K width3,V width2,CPU float64,one thread. Compare dense masked softmax and independently enumerated/gathered keys. Outputs and Q/K/V gradients <=1e-10; central directional difference step1e-6 <=1e-7. Empty G must match the prior sliding-window function.
- Reach: same lengths/radii/global sets/policies, depths1/2/3:96 cases. Independent set propagation versus adjacency powers; uniform local/global averaging Jacobian positive exactly on reachable pairs. Causal reach must contain no future edges. Any nonempty bidirectional G must yield all-pairs reach after two layers. Record individual reachable sets.
- Counts: n5/9/128/1024,r0/2,same four G sets,both policies:64 cases. Independently enumerate visible keys and count by partition: global query rows read all allowed keys; nonglobal query rows read their local interval plus global keys outside it. Compare local/full counts.
- Controls at n9,r1,G={0}: omit global query edges, omit global key edges, or let only global queries read future tokens. Zero Q/K and selected nonzero V must detect each omission/leak above1e-6. For the last case, output4 cannot read input8 in one layer but must leak through global0 after two. Correct causal output4 and its derivative with respect to input8 must stay zero. Record a causal middle-global relay from input0 to output8 in two layers.
- Save per-case evidence, source/protocol hashes and all prior result hashes; preserve failures without threshold changes or replacement cases. End after audit and concise educational documentation. No training or speed claim follows.
