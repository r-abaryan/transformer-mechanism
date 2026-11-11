# Attention-pattern versus output capacity

Freeze before execution. Reuse archived S05 non-uniqueness evidence and the archived A=softmax(I_4) head-width target. Do not retrain or rerun those studies. Add two value fixtures only, CPU float64, tolerance 1e-10.

1. Distinct scalar values V=[0,1,2,3]^T. Construct a width-1 head with K=V. For each target output (AV)_i, solve softmax(q_i*K^T)V=(AV)_i by 80 bisection iterations on [-10,10], after checking the target is bracketed. Verify output agreement <=1e-10 but attention difference >.01. This is a finite algebraic construction, not training or an across-input model. Save Q, K, both probability matrices, outputs, errors and rank([V,1]).
2. Values V=[I_3;0,0,0]. Verify rank([V,1])=4, while rank(V)=3. For every row, the first three attention weights are the output coordinates; recover the last as one minus their sum. Verify exact recovery <=1e-10. Apply the width-1 probabilities from fixture 1 as a negative control and require output difference >.01. State separately why no width-1/2 representation can reproduce this output exactly, using the archived target's log-odds rank 3.

Theory: for normalized rows a,b, equal output requires (a-b)V=0 and (a-b)1=0. Thus full row rank of [V,1] guarantees uniqueness among probability rows. Rank deficiency permits small positive-preserving perturbations for interior rows; boundary rows and restricted Q/K families require care. An arbitrary nullspace perturbation need not be realizable by the same trained head.

Independently audit new probabilities/outputs with 50-digit decimal arithmetic, the archived S05 identity with exact fractions, and the augmented-value determinants/ranks algebraically. Verify hashes and all fixed fixtures. Stop after the two new value cases. No head-count sweep or novel architecture claim.
