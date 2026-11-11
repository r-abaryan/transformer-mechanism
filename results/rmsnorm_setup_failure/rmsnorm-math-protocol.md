# LayerNorm versus RMSNorm: frozen mathematical protocol

Question: which function and gradient properties change when centering is removed? This follows the closed normalization-placement study; it does not repeat its training comparison.

CPU float64, existing PyTorch2.14, one thread, deterministic operations. No trained checkpoint changes, retraining, timing or architecture adoption. Check the original RMSNorm and LayerNorm papers and installed library definitions. Set epsilon explicitly; do not compare unlike defaults.

For dimensions4 and16 use five vectors: ordinary x_i=i+1; centered x_i=i+1-(d+1)/2; constant x_i=2; near-zero x_i=(-1)^i(i+1)*1e-6; zero x_i=0. Use epsilon0 and1e-5; skip epsilon0 whenever the respective denominator is zero. Use gain gamma=1 and gamma_i=.5+(i+1)/d. LayerNorm bias is explicitly zero, matching gain-only RMSNorm. This isolates centering without conflating bias removal. Expected68 valid cases.

For each case record inputs, gains, outputs and Jacobians. Compare explicit formulas with native `F.layer_norm`/`F.rms_norm`, autograd and central finite differences. Difference step is `max(max|x|,1e-8)*1e-5`. Native-forward error <=1e-10; analytical/autograd relative maximum error <=1e-10; finite-difference relative maximum error <=1e-7. Relative maximum error divides by `max(1,max|J|)` so large near-zero derivatives are not silently compared on an unsuitable absolute scale.

Test common feature shifts+3, positive scaling2 and negative scaling-1. Record output differences rather than assuming invariance. At epsilon0 positive scale preserves defined outputs; negative scale reverses the bias-free output. At epsilon>0 scaling need not be exactly invariant. LayerNorm's common-shift invariance holds with either gain; RMSNorm has counterexamples, not a claim that every shift changes every vector. Allow1e-8 absolute error for floating-point shifted outputs: adding3 to near-zero inputs loses relative precision before normalization. Independently verify the exact symmetry with Decimal arithmetic.

Derive LayerNorm J=diag(gamma)[P/r-cc^T/(d*r^3)], with P=I-11^T/d, c=Px, r=sqrt(mean(c^2)+epsilon). RMSNorm J=diag(gamma)[I/r-xx^T/(d*r^3)], with r=sqrt(mean(x^2)+epsilon). Check LayerNorm's exact mean-direction null and both radial epsilon terms. With unit gains verify eigenvalues: LayerNorm {0,epsilon/r^3,1/r repeated d-2}; RMSNorm {epsilon/r^3,1/r repeated d-1}. At constant/zero inputs epsilon>0 makes the repeated eigenvalues coincide appropriately. No eigenvalue claim applies unchanged to nonuniform gains.

Include two explicit witnesses. At a zero-mean point outputs agree but their mean-direction derivatives differ. For residual y=x+N(x), a common shift remains through the LayerNorm skip path although the normalization branch removes it. These are local structural examples, not trained-model diagnoses.

The independent standard-library auditor verifies fixture coverage, source/protocol hashes and records. Recompute scalar normalization with70-digit Decimal arithmetic and Jacobians through central differences with step1e-22, independently of the analytical/autograd implementations. Audit recorded shift/scale witnesses, radial/mean directions, unit-gain characteristic directions and residual examples. Require normalized discrepancies <=1e-10; explicitly verify that the false universal shift/positive-scale invariance claims have counterexamples.

Freeze this protocol and both scripts before execution. Retain one report, one audit and one educational document with math, results tables and sources. Close after all checks. No claim that centering is useless, RMSNorm learns better, or fewer reductions guarantee faster GPU execution.
