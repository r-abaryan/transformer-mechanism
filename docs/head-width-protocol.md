# Head-width representation probe

Freeze before execution. One head, four tokens, CPU float64. No trained-model sweep. Distinguish rank of scores S, probabilities A, and log odds B. All numerical equalities use 1e-10 tolerance; numerical ranks use singular values >1e-10 and retain the full spectrum.

1. Set Q=K=[0,1,2,3]^T (width 1). Verify rank(S)=1, but rank(softmax(S))=4. Independently verify its determinant from the Vandermonde formula for exp(i*j), followed by row normalization.
2. Define desired probabilities A=softmax(I_4), and B_ij=log(A_ij/A_i4) for j=1,2,3. Verify B equals [I_3; -1,-1,-1], rank 3, and B^T B=I_3+ones(3,3). Its singular values are 2,1,1: no width-1 or width-2 free-Q/K head can exactly represent this positive unmasked attention pattern.
3. Compute best rank-1 and rank-2 SVD approximations of B. Verify Frobenius errors sqrt(2) and 1. These are log-odds approximation errors, not output-error lower bounds. Construct exact width-3 Q/K with K=[I_3;0], Q=sqrt(3)*B; verify desired probabilities.
4. Mask zero scores causally. Verify lower-triangular attention has rank 4 and determinant 1/24. Do not apply positive unmasked log-odds analysis across masked zeros.

Save fixtures, spectra, probabilities and hashes. Independently audit with 50-digit decimal softmax/determinants and exact integer Gram matrix checks. Stop after these four fixture families. Necessary representability constraints do not establish training failure, optimal head count, or a new architecture.
