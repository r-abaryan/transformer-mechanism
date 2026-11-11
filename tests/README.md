# Reusable correctness checks

Exposure-trace classifier control: `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_copy_exposure_readout_control.py -v`. Two methods pass, covering saved initialized native traces, symmetric/ordered identities, classifier-only/state-only/interaction boundaries, different terminal rivals, common logit-shift invariance, empty groups and deliberate missing/corrupted terms. Fixtures use unique ignored folders and zero training. This checks diagnostic arithmetic; it does not establish a learned FFN mechanism.

Copy positional sites: `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_copy_position_sites.py -v`. Three methods passed: twelve initialized output/gradient cases, exact ordinary/combined predecessor checks, explicit positional-site wiring, a wrong-site rejection, supplied logits/ties/absent groups, decision controls and the actual saved training/evaluation/audit entry. Five deliberate artifact faults are rejected. Fixtures use eight updates in unique ignored folders. Direct execution writes the one-time preflight marker; use discovery for repeats. Reference structure/projection helpers are shared, and formal training gradients are not replayed.

Copy positional exposure: `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_copy_position_exposure.py -v`. Three methods pass: six output/gradient cases, native/translation/order checks, supplied logits/ties/absent groups, initialized saved training/evaluation/audit and four deliberate artifact faults. Fixtures use four updates in unique ignored folders; formal runs remain untouched. Use discovery for repeat checks; the direct entry writes the one-time preflight record. Audit reference equations share attention/projection utilities and do not replay every formal training update.

Copy-source visibility checks: `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_copy_source_visibility.py -v`. Three methods pass, covering 18 initialized output/gradient cases, native and gap-marker equivalence, both signs of masked perturbations, saved producer/audit entry, supplied ties/absent groups and six deliberate artifact defects. Unique ignored fixtures use zero training and do not overwrite formal evidence. The direct script entry writes a one-time preflight marker; use unittest discovery for later checks.

Map-exposure checks exercise actual initialized saved training/evaluation/audit, new labels at fixed key order, exact archived-control replay, paired streams, independent gradients/scalar Adam, ties and missing groups. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_map_exposure.py -v`. Two methods passed before formal training (15.136 seconds; 17.282 including discovery/imports). Deliberate data/order/gradient/parameter/stream/label/logit/count/decision/missing-cell faults fail. Unique ignored fixtures use 24 synthetic updates, separate from formal exposure. The audit checks every recorded update and score cell; the training helper and decision reducer are shared.

Fitting-trajectory checks exercise initialized saved checkpoints, the actual producer/auditor, supplied ties/empty groups, scale invariance and decoder counterexamples. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_fitting_trajectory.py -v`. Two methods passed in 4.619 seconds before formal scoring. Deliberate coefficient/logit/label/count/decision/missing-cell/missing-group/CSV faults fail. Fixtures use unique ignored directories and zero training; this entry does not rescore formal trajectories. The audit separately reconstructs matrix writes, margins, quantiles, cohorts and primary signs; CSV extraction is shared.

Cross-block training checks exercise the initialized saved producer/evaluator/audit, exact native quadratic controls and independent dense block directions. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_fitting_transport_training.py -v`. Two methods passed in 9.651 seconds before formal execution; deliberate gradient/direction/counter/loss/label/logit/missing-group/decision faults fail as intended. Unique ignored fixtures preserve old evidence. Formal execution/audit are serial CPU float64 and require no multiprocessing pipe permission. Native controls replay exactly; all trial gradients and score cells are checked. The line-search replay and some reducers are shared.

Transport review checks use saved synthetic traces, an independent dense inverse-BFGS recurrence, the initial zero-transport case and deliberate direction/partition/missing-step/metric faults. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_fitting_transport.py -v`. Two methods passed before formal analysis; no neural training or prediction evaluator is involved. The actual review reuses archived gradients, audits all 306 calls and reconstructs all 189 L-BFGS operators separately. Arithmetic checks do not establish transport's learning benefit.

Decoder scale/shape checks exercise the initialized saved evaluator and actual scorer/audit, scalar weight projections, scaled-classifier forwards, ties/absent groups and deliberate projection/logit/label/count/decision/missing-cell faults. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_decoder_scale_shape.py -v`. Two methods passed in 4.510 seconds before formal scoring, using ignored unique fixtures and zero training. The formal producer transforms cached logits; the audit independently recomputes source forwards, projection arithmetic, scalar metrics and the primary rule. The full contrast reducer is shared.

Representation checks use initialized saved donors/evaluator/audit, exact value-basis arithmetic, private decoder-basis permutations, tied logits/absent groups and deliberate donor/logit/label/count/decision/missing-cell faults. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_fitting_representation.py -v`. Two methods passed in 3.504 seconds before formal evaluation, using ignored unique fixtures and zero training. The formal diagnostic evaluates saved hybrids, not a training recipe. Matrix logits, scalar metrics and sufficiency error counts are independently checked; contrast reductions and most parameter-statistic reducers are shared.

Direction checks exercise initialized saved training/evaluation/audit, independent matrix gradients, an exact native quadratic control, identical first steps, later direction divergence, ties/absent groups and deliberate gradient/direction/counter/loss/label/logit/missing-pool faults. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_fitting_direction.py -v`. Two methods passed in 12.519 seconds before formal execution; unique ignored fixtures preserve old evidence. Formal training and audit use three native Windows workers and need pipe permission inside a restricted sandbox. The intervention audit shares the native line-search engine, while directions and gradients receive separate checks.

Fresh fitting confirmation checks exercise three native Windows CPU workers, initialization/warm-start lineage, actual saved warm-up/fitting/evaluator/audit paths, tied logits/absent groups and deliberately corrupted labels/logits/gradients/source steps/counts/decisions/missing cells/loads/family leakage. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_full_pool_fitting_confirmation.py`. Two methods pass; the fixture producer makes 51 optimizer calls and 57 gradients, with verification replays separate. Fixtures use unique ignored paths and do not rerun formal confirmation. Windows multiprocessing needs native pipe permission when invoked inside a restricted sandbox.

Full-pool fitting checks exercise actual initialized saved Adam/L-BFGS execution, evaluator/audit, complete-objective weighting, exact fit boundary, missing groups, ties/absent groups and deliberately corrupted labels/logits/loss/gradients/parameters/optimizer/counts/decisions. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_full_pool_fitting.py`. Two methods pass using unique ignored fixtures. The fixture producer makes 13 optimizer calls and 15 gradient evaluations; audit replays and additional gradient checks are separate. This command does not rerun or overwrite formal training. Full-objective optimizer replay shares PyTorch's engine, while gradients and two-loop directions receive separate checks.

Classifier norm-cap checks exercise actual initialized saved training/evaluation/audit with an active cap, paired streams, gradients/scalar Adam, null/contraction projections, ties/absent groups and deliberate label/logit/parameter/moment/gradient/stream/count/decision faults. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_readout_norm_cap_training.py`. Two methods pass; each invocation uses eight synthetic training updates in unique ignored fixtures. It does not rerun or overwrite the four formal training runs. The separately measured two profile steps are not part of this test command.

Beta/key checks exercise the initialized saved factorial evaluator/audit, identical-donor null effects, fixed-value/head scope, common key rotation, ties/absent groups and deliberate source/label/logit/coefficient/count/contrast defects. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_beta_key_transport.py`. Two methods pass using unique ignored fixtures and zero training/solvers; formal source evidence remains untouched.

Writing-stage checks exercise an initialized saved evaluator/audit, known aligned-key damage versus orthogonal preservation, tied logits/absent cohorts and deliberate label/logit/coefficient/position/cohort/summary/count defects. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_write_interference_stages.py`. Two methods pass using unique ignored fixtures, zero training/solvers. Formal stage evidence and executed sources remain untouched.

Training-stage checks exercise initialized saved evaluator/audit, singular reads remaining unresolved, supplied ties/absent groups, exact certificate controls and deliberate read/label/coordinate/head/step/count/CSV defects. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_readout_conflict_stages.py`. Two methods pass using unique ignored fixtures and zero training. Evaluator model reloads preserve global RNG via `fork_rng`; fixtures do not rerun formal stages or overwrite frozen sources.

Shared-classifier recovery checks cover full-coordinate mapping, cancellation-sensitive products, exact conflicts, singular rejection, missing groups, initialized saved recovery/audit and deliberate transformation/head/certificate/count defects. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_shared_readout_recovery.py`. Three methods pass using unique ignored fixtures, pinned analysis dependencies and zero training. Fixture solver calls do not count as formal recovery runs and do not overwrite earlier evidence.

Shared-classifier checks cover separable/conflicting features, free negative weights, exact certificate/tie controls, supplied absent groups, initialized saved evaluator/audit and deliberate read/label/head/group/count defects. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_shared_readout_feasibility.py`. Three methods use unique ignored fixtures and the pinned `requirements-analysis.txt` dependencies. Fixture solver calls are separate; this entry does not solve formal training pools, train or overwrite evidence.

Single-query checks exercise individual-loss autograd, coefficient factorization, different-value/zero/absent-cohort controls, initialized saved evaluator/auditor and deliberate coefficient/label/mask/summary/CSV defects. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_single_query_write_opposition.py`. Two methods use unique synthetic fixtures and zero optimizer calls; formal analytic runs are not launched or overwritten.

Write-history checks exercise known within/between/same-position partitions, shared-value mapping, zero/orthogonal/aligned/opposing controls, and the actual saved-artifact reduction/audit entry. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_write_history_opposition.py`. Two methods use one initialized primary fixture with synthetic forwards/backwards, zero optimizer calls, and deliberate summary/CSV/missing-cohort/input defects. Formal evidence is never overwritten.

Write-credit checks exercise initialized saved evaluation/audit, tied logits/absent cohorts, event finite differences, analytic temporal transport and deliberate label/logit/mask/gradient/group/summary defects. Run `.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_write_credit_assignment.py`. Three methods use unique ignored fixtures, zero optimizer calls, and do not overwrite formal evidence.

Preservation checks exercise actual initialized saved replay/audit, zero-start Reader agreement, known destructive writes, orthogonal/zero-beta controls, local error gradients, ties/absent groups and deliberate target/state/logit/residual/step/group/summary defects:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_shared_state_preservation.py
```

Three methods pass. Unique ignored fixtures use two small preflight feasibility solves; this entry does not replay formal saved histories or train.

Shared-state checks exercise actual initialized saved-model evaluation and audit, exact dual fixtures, supplied ties/absent groups and deliberately corrupted labels, states, logits, groups and summaries:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_shared_state_feasibility.py
```

Three methods pass. Requires the pinned additions in `requirements-analysis.txt`. Fixtures use unique ignored `results/test_runs` folders; this entry does not solve formal histories, train or overwrite frozen evidence.

First-moment checks exercise actual saved training/evaluator/audit and a complete CPU step profile:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_memory_first_moment_reset.py
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_memory_first_moment_audit_recovery.py
```

The original two methods check paired streams, scoped reset, scalar Adam, matrix scores/gradients, supplied ties/absent groups and inverse moment/label/step/RNG defects. Recovery adds ordered nested criteria and deliberately missing/reversed lists/decisions, plus the real saved fixture under that comparator. Each fixture invocation makes 12 synthetic optimizer calls in unique test directories. Formal frozen evidence is not overwritten. For the recorded study's audit, use the separate recovery entry; preserve the original stopped auditor.

Adam memory/error checks exercise saved-state evaluator/audit, private finite effects and probes:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_adam_memory_error_transport.py
```

Three methods cover actual matrix/scalar auditing, supplied ties/absent errors, fixed membership and corrupted labels, masks, gradients, steps, numerator terms and probe signs. A unique initialized synthetic fixture reconstructs four optimizer steps for preflight. This entry does not read formal replay results, extend training budgets or launch the formal diagnostic.

Memory-gradient checks exercise actual initialized saved evaluator/audit and numerical probes:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_memory_gradient_competition.py
```

Three methods check independent matrix gradients, weighted correct/error sums, fixed-mask probes, ties/absent partitions and synthetic opposition with both beneficial/harmful totals. Corrupted labels, gradients, counts, signs, load weighting and missing loads must fail. This entry writes unique initialized fixtures; it does not run formal checkpoint diagnostics or training.

Target-only checks exercise initialized saved forwards and the independent audit entry:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_target_only_history.py
```

Three methods cover actual filtering/query order, variable retained lengths without padding, supplied ties/missing groups, clamped nonunit keys, and corrupted counts, labels, logits and competing writes. This entry writes unique initialized fixtures and launches no formal checkpoint evaluation or training.

Adam head-transport checks exercise initialized and warmed saved replay with independent matrix/scalar auditing:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_readout_adam_transport.py
```

Three methods check exact resumed versus uninterrupted native states, all-coordinate Adam, current/stored radial sums, supplied missing groups, and synthetic momentum agreement/opposition. Wrong moments, parameter steps, RNG draws, labels, signs and historical beta records must fail. Fixtures replay short synthetic optimizer cycles; this entry does not launch the formal original-budget reconstructions or extend historical training.

Readout-scale pressure checks exercise initialized saved forward/gradient paths and independent matrix audits:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_readout_scale_pressure.py
```

Three methods cover both wrong-answer derivative signs, ties, absent groups, offset invariance, equal-load weighting, scalar autograd, two finite differences and saved native/matrix W/b gradient projections. Deliberate wrong signs, counts, labels, logits and gradients must fail. This entry writes unique initialized fixtures and performs no formal checkpoint analysis or training.

Readout-scale checks exercise initialized saved evaluators, independent matrix audits, unchanged memory and deliberate errors:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_readout_scale_control.py
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_readout_scale_threshold.py
```

Three control methods and one threshold-regression method cover norms, offset invariance, bias scaling, unchanged winners, absent groups and count/label/head/scale/memory corruption. The threshold fixture demonstrates the original count mismatch and verifies 60-digit versus 90-digit recovery. Formal evidence stays preserved; its successful audit entry is `experiments/recover_readout_scale_audit.py`. The original auditor documents the uncorrected discrete-count failure.

Saved-score loss decomposition checks exercise all transitions and actual initialized evaluator/audit reloads:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_feedback_loss_decomposition.py
```

Three methods cover signed contribution sums, scalar probabilities/margins, absent groups, tied winners, uniform-logit-shift invariance, initialized saved matrix-equation audits and deliberate wrong labels/counts/contributions/probabilities/example identities. They write unique fixture folders and perform no training or formal endpoint analysis.

Exhaustive-feedback continuation checks exercise exact resume and actual saved evaluator/audit paths:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_query_feedback_continuation.py
```

Three methods cover every model/optimizer tensor against uninterrupted training, initialized/intermediate reloads, raw gradients, first-fit and cap stopping, supplied logits, absent groups and wrong moments/RNG states/scores/labels. Fixtures write unique test-run folders. This entry does not launch formal continuation or open historical held-out pools.

Matched query-feedback checks exercise both query policies and actual saved evaluation/audit paths:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_query_feedback_training.py
```

Three methods check all-parameter gradients and CE normalization, two short training runs, initialized/intermediate evaluator reloads, stream/exposure reconstruction, missing groups and deliberate wrong scores/maps/labels/counts. This entry does not launch formal training or open the reserved test data.

Query-gradient variability checks use initialized checkpoints and deliberately corrupted fixtures:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_query_gradient_variability.py
```

Three methods cover population variance, native finite differences/autograd reload at four loads, missing groups, supplied logits and wrong derivative/variance/label/metric controls. No formal checkpoint analysis or training runs through this test entry.

Retention replay preflight exercises the actual original/instrumented training loops, initialized/intermediate checkpoint audits, scalar Adam reconstruction, absent groups and deliberate corruptions:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_retention_learning_replay.py
```

Two methods cover both arms and 48 saved objective cells. They write only unique test-run folders and do not launch formal reproduction.

Retention endpoint gradients use fresh initialized checkpoint fixtures:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_objective_beta_pressure.py
```

Four methods exercise labels, eight actual evaluator/audit reload cells, supplied logits, missing groups, tangent/finite differences and deliberate errors. They do not run formal checkpoint evaluation or training.

Saved language-model context checks use fresh initialized checkpoint fixtures, with CPU and available CUDA evaluation:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_language_context_use.py
```

Four methods cover labels/permutations, actual evaluator/audit reload, rotary positions and missing-group metrics with deliberate errors. They do not invoke the frozen formal run.

Run from the workspace root:

```powershell
.venv-gpu/Scripts/python.exe tests/run_tests.py
```

Each invocation creates fresh fixtures and a summary in `results/test_runs/`. It reuses the three archived copy modules: 12 methods and 63 initialized equation/gradient cases, plus three runner-safety methods. No training occurs. These tests cover specific copy implementations, not the whole project.

The archived test files are hashed by completed studies. Keep their bytes unchanged. Their standalone entry points and direct unittest discovery can write fixed historical checkpoint names; the alignment entry also overwrites its preflight. Use the launcher above for routine checks. It imports their cases, redirects fixture paths and never invokes those recording blocks.

Fixtures remain available for inspection because Windows temporary-directory cleanup previously failed. Their folder is listed in `.gitignore`. `--run-name NAME` permits a single fresh folder name; an existing name or path traversal is rejected before test execution. Nothing is recorded in a frozen study folder.

Future arithmetic checks belong in reusable tests. Create a study only when it addresses an empirical or explanatory research question.

Compact delta/softmax reader checks are separate and also use fresh retained fixtures:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_delta_retrieval.py
```

This runs five methods, including 18 initialized equation/gradient cases, six directional checks and deliberate errors. The closed study's initialized/final audits are recorded in its [report](../docs/topics/attention/delta-retrieval-results.md).

The checkpoint-only recency intervention has three focused methods, including inverse and permutation controls:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_recency_ablation.py
```

Native-CUDA reader resource checks execute actual timing, retained-state, gradient and disposable optimizer paths. They require the six saved delta-retrieval checkpoints, which are ignored by Git:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_retrieval_gpu_resources.py
```

The command updates only disposable in-memory copies. Formal three-process measurements and audit are recorded in the [resource report](../docs/topics/attention/retrieval-gpu-resources-results.md).

Association-load correctness checks cover the larger key table, no-recency buffer, all-parameter gradients, labels/families, actual initialized checkpoint audit and cross-write/decision controls:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_association_load.py
```

Fresh initialized fixtures are retained under ignored `results/test_runs/`. Closed training evidence is in the [association-load report](../docs/topics/attention/association-load-results.md).

History-repetition checks cover whole-block order, softmax invariance, deliberate partial/wrong duplication, labels/scalars, paired categories, decision boundaries and actual initialized saved-checkpoint audit:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_history_repetition.py
```

These four methods create fresh ignored fixtures and perform no training. Saved-reader findings are in the [repetition report](../docs/topics/attention/history-repetition-results.md).

Query-write order checks cover the fixed event map, relative order, labels/coverage, prefix eligibility, softmax invariance, a known suffix harm, deliberate frozen-suffix/trace/scalar errors and actual initialized checkpoint audit:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_query_write_order.py
```

Four methods, no training; fresh ignored fixtures. [Saved-reader trace results](../docs/topics/attention/query-write-order-results.md).

Key-geometry checks cover Gram/padding contracts and inverse errors, all-query labels, complete-map failures, actual initialized checkpoint audit, scalar/logit/count errors, original Reader agreement and the orthogonal direct-read/reversal identities:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_key_geometry_retention.py
```

Four methods, no training; fresh ignored fixtures. [All-query retention results](../docs/topics/attention/key-geometry-retention-results.md).

Overlap-regularization checks cover scalar/gradient/directional penalty equations, the nonzero bound fixture, zero-gradient/diagonal errors, actual initialized checkpoint evaluator/audit, labels/scalars and selection/decision inverse controls:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_overlap_regularization.py
```

Five methods, no training; fresh ignored fixtures. [Bounded training comparison](../docs/topics/attention/overlap-regularization-results.md).

Query-coverage checks cover independent labels, one-query versus finite-average loss/gradients, actual initialized saved evaluator/audit at all four loads, absent groups, wrong logits/scalars/map counts and inverse objective-fit/confirmation gates:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_delta_query_coverage.py
```

Five methods, no training; fresh ignored fixtures. [Matched training results](../docs/topics/attention/delta-query-coverage-results.md).

Beta-swap checks cover beta-only/self-copy contracts, first-write labels/positions, actual initialized checkpoint evaluator/audit at every load, wrong scalars/logits/map counts, analytical and numerical beta derivatives, and inverse directional/coverage controls:

```powershell
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_beta_swap.py
```

Five methods, no training; fresh ignored fixtures. [Frozen intervention results](../docs/topics/attention/beta-swap-results.md).
