# Majority: frozen task-transfer protocol

Question: can the selected recall architecture learn a different task and transfer to longer inputs? This is from-scratch training, not reuse of learned recall weights or an architecture comparison.

For odd n, input is n binary tokens followed by query marker 2. Label y = 1[sum(bits) > n/2]. Read only the final marker's two logits. Majority requires an aggregate; a single bit is an imperfect shortcut. Margin m = |2 sum(bits) - n| measures closeness to a tie. Report m=1 separately.

## Fixed architecture and budget

Use the existing MQA implementation: width16, two causal pre-norm blocks, two query heads of width8, one K/V head, RoPE offset4, ReLU FFN32, learned LayerNorm gains/biases, epsilon1e-5. Replace embedding with3x16 and bias-free readout with16x2. Train all4,016 parameters from scratch. Query exposure from recall has no key-query analogue here; every string receives one final-marker label.

Seeds1613,1627,1657,1669,1693. CPU float64, one thread, deterministic algorithms. Adam lr.001, default betas/epsilon, no weight decay or clipping. Batch64, alternate9/11 bits for6,000 updates, no warmup/dropout/early stopping. Draw both length streams every step using separate generators seeded seed+1000/2000; only the alternating length is trained. Save state/optimizer/generators and validation at0/1000/.../6000. Final checkpoint only is tested. Per run:384,000 examples,4,224,000 input tokens including markers;46,848,000 dense attention matrix entries per layer/head (counting masked entries). This budget differs from recall because lengths differ.

## Data reservation

Data RNG20270111. A family is the distinct orbit {x, reverse(x), complement(x), reverse(complement(x))}. Keep entire families in one split. Each orbit has2 or4 strings and balanced labels. This protects against exact, reversed and complemented training/test overlap. It does not hold out bit counts or all permutations.

Enumerate all binary strings at9/11 bits, sort canonical family representatives, shuffle independently using the same sequential RNG. Family counts are136/528, from (2^n + 2^((n+1)/2))/4 for odd n. Split9 into80/20/36 train/validation/test families; split11 into350/50/128. Expand every orbit to its distinct strings. Train uniformly over those strings with replacement.

For15 and21 bits, sample uniform independent bitstrings and retain the first200 distinct canonical families at each length; expand the orbits. Thus longer tests are sampled families, not exhaustive support or exactly uniform expanded strings. All test labels are balanced. Generate the fixed artifact before training, but do not evaluate any model on test until all five attempts finish. Record first-bit, last-bit, constant-class and exact-count baselines on each test. Report class-wise, family-all-correct, cross-entropy and margin-stratum accuracy.

## Predeclared decision and audit

Pass task transfer only if all five attempts have finite loss/gradients and complete6,000 updates; every seed scores>=99% at9 and11 bits; every seed scores>=90% overall AND>=90% at m=1 at15 and21 bits. Thresholds are research targets, not paper-derived guarantees. The same fixed tests are shared across seeds; orbit variants and seed/example outcomes are not independent datasets.

Before training, compare full outputs and all parameter gradients against the existing independently written SDPA/norm/RoPE forward, tolerance1e-10. Check causal prefix invariance when a later bit changes. Check labels against a signed-sum definition. Wrong last-bit and flipped-label controls must fail on enumerated inputs.

After training, independently check labels/family closure/disjointness/capacities/balance; replay initialization and index hashes; verify update counters/budget/provenance. Recompute all20 test evaluations using the independent forward; maximum logit disagreement<=1e-10, identical argmax, independently computed metrics within1e-10. Preserve every checkpoint, failure and prediction. Failed attempts fail the decision; no replacement seed, longer training, changed threshold or tuning on opened tests. Close after these checks.

## Source and scope

[Hahn and Rofin2024](https://aclanthology.org/2024.acl-long.800.pdf), sections2-3, empirical discussion and appendixE/Table1, motivates separating expressivity from learnability and using majority as a lower-sensitivity aggregation task. Their study examines sensitivity/sharpness and uses different models/training. This protocol measures classification transfer, not their sharpness theorem or a reproduction. Architecture success on two toy tasks would still not prove language-model superiority.
