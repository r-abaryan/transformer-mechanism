# Causal head-width probe

Freeze before execution. Four tokens, one head, free Q/K, CPU float64. No retraining. Reuse the earlier head-width/output-capacity reasoning; test only its causal extension. Tolerance 1e-10, numerical rank threshold 1e-10, independent decimal audit.

Construct a width-two head: K rows [1,0], [0,1], [0,0], [0,0]; Q rows [0,0], [0,0], [sqrt(2),0], [0,sqrt(2)]. Apply the standard lower-triangular causal mask including self. Query rows 3 and 4 jointly see keys 1,2,3. Relative to key 3 their 2x2 log-odds matrix must be I_2. Its rank two forbids exact width-one realization despite causal masking. Save Q/K, scores, probabilities and singular values. Check masked zeros, positive visible weights and row sums.

Independently define expected visible scores as [0], [0,0], [1,0,0], [0,1,0,0]; verify the width-two construction exactly reproduces them within tolerance. All finite causal diagonal weights are positive, so the probability matrix is full-rank triangular; this alone says nothing about minimum Q/K width.

Two fixed value cases: (1) V=[0,1,2,3]^T, construct width-one K=V and per-row queries via 80 bisection iterations on [-10,10] over the visible prefix. First-row query is zero because only one key is visible. Verify output error <=1e-10 with probability difference >.01. (2) V=[I_3;0,0,0], recover A from AV and normalization; require width-one output mismatch >.01 as a negative control. The rank-two log-odds obstruction then also forbids exact width-one output reproduction for these fixed values.

Audit saved artifacts with 50-digit decimal arithmetic, explicit expected score rows, exact determinant of the I_2 certificate and value reconstruction. Stop after this single causal pattern and its two value cases. Common-visible-key rank gives a necessary condition, not a general sufficient characterization of causal attention; shared factor compatibility across overlapping prefixes is untested.
