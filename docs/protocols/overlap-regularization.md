# Learning lower overlap at the original state size

2026-10-04. User authorized one bounded matched training comparison. Question: does discouraging key overlap improve retention without increasing the 16x8 delta state? No novel architecture claim.

## Fixed objective and budget

Reuse the association-load task: 32 events, loads 2/8/16/32, 32 known key symbols, six consistent value classes. Unchanged LoadReader: 615 parameters, unit keys, learned values/classifier/scalar beta. Task CE supervises one selected final query per example. Do not add all-query supervision or alter event order.

Let U contain the 32 normalized keys. Use `R=mean_{i!=j} (u_i dot u_j)^2`, averaging 992 ordered pairs. Objective `task_CE+lambda*R`, lambda 0/1/10. Evaluate task CE separately from the regularized objective. R measures geometry, not accuracy. Its minimum is 1/31, not zero, for 32 unit keys in 16 dimensions.

Every run: Adam .003, 3,000 updates, batch 64, CPU float64, one thread; cycle loads 2/8/16/32. 192,000 selected-query targets and 6,144,000 events per run, 48,000 targets per load. Match initialization and minibatch streams per seed. Perform disposable complete CPU/CUDA float64 optimizer steps before locking formal execution; report peak allocated/reserved GPU bytes including optimizer states. Formal training is CPU; profiles are not speed measurements.

Development seeds 3407/3413, all three weights: six runs. Train/validation pools per load: 768/192 examples, 128/32 families with six cyclic renamings. Generate data with seed 2026100441, excluding old association-load development/confirmation families.

## Selection, confirmation and stopping

Require baseline and candidate selected-query training accuracy >=99% in every load and development seed. Candidate validation loads 2/8 must stay within two points of matched baseline in every seed. Among eligible positive weights, select highest mean validation all-query accuracy at load 32, then lowest common task CE, then lower weight. Selection uses declared validation; no test access. If neither candidate qualifies, stop without confirmation.

Otherwise lock the selection and run baseline/selected penalty pairs on fresh seeds 3433/3449/3457 and fresh data seed 2026100442, excluding all old and development families. Same train/validation counts; test pools have 384 examples per load (64 families). Complete all six confirmations before opening test metrics. Maximum formal budget: 12 runs, 36,000 updates, 2,304,000 supervised queries and 73,728,000 events. No checkpoint selection, LR search, replacement seeds or extension.

Full training/validation/test selected-query accuracy/CE at every load. Also evaluate every query from one final state at load 32 in each split, with all-query CE/accuracy and complete-map accuracy. These additional query labels are evaluated, not supervised. Poor all-query training scores limit a pure transfer interpretation even if supervised training fit passes.

Frozen confirmation criteria, reported separately: >=99% supervised training accuracy in every run/load; >=10 percentage-point all-query test-32 gain in every matched seed; loads 2/8 preserved within two points; >=99% all-query train-32 accuracy in both arms; penalty complete-map test-32 accuracy >=95% in every seed. `bounded_benefit` requires supervised fit, gain and low-load preservation. `simultaneous_retention_supported` additionally requires all-query training adequacy and complete-map reliability. Neither criterion may be revised after evaluation. Report every seed, selected-query/all-query distinction, penalty and beta; no success from R alone.

## Verification

Five reusable preflight methods cover independent penalty value/normalized-key gradient/directional difference, deliberate zero-gradient and diagonal-inclusion errors, a 1/31 tight-frame fixture, actual initialized saved evaluator/audit, missing groups and bad labels/scalars, selection eligibility/coverage and inverse decision controls. Data labels/families use independent dictionary checks. Final audit checks all logits with matrix equations, task CE/complete-map counts with Python reductions, penalty with explicit scalar dot products, optimizer counters, initial states and complete streams. Selection/decision replay shares producer functions and uses tested inverse controls; it is not independent policy code.

Preserve old sources/protocols/data/reports/checkpoints. Record all attempts and blocked confirmation if development is ineligible. Report profiles, elapsed execution times and peak GPU memory without inferring performance ranking. Stop at the declared budget and audit.

## Sources and local derivation

[Yang et al., DeltaNet 2024, section 2.2](https://arxiv.org/html/2406.06484v2) defines the correction rule. [Casazza, Fickus and Mixon, Auto-tuning unit norm frames, 2010, introduction](https://arxiv.org/pdf/1009.5562) discusses squared inner-product frame potential and its minimizers. Older defining mathematics is directly relevant; no claim of a latest exhaustive review.

Local bound: `||U U^T||_F^2=||U^T U||_F^2>=32^2/16` by Cauchy-Schwarz on the nonnegative eigenvalues. Subtract 32 unit diagonal terms and divide by 32*31 to obtain R>=1/31. `[I;I]` attains it. This does not imply equal pairwise overlaps or optimal retrieval. Our joint Adam objective is not the paper's frame-only algorithm or its convergence theorem. No source predicts local classification gains. [Saved geometry evidence](../topics/attention/key-geometry-retention-results.md).
