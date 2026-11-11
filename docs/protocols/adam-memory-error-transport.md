# Adam memory movement and remaining errors

Question: do selected original Adam memory steps improve or worsen the errors present immediately before each step?

The [partition diagnostic](../topics/attention/memory-gradient-competition-results.md) found local opposition but helpful joint raw descent. Adam need not follow that direction. Use the already audited original-step traces, without replaying another optimizer call.

| Item | Frozen choice |
| --- | --- |
| Steps | Seeds 3461/3463; 3001–3004 and 11001–11004; 16 steps total |
| Source | Existing readout_adam_transport traces, bound to its passing audit and report/checkpoint hashes |
| Data | Full existing training load-32 pool, 768 histories × 32 queries = 24,576 answers per step |
| Mask | Errors under that step's before-model; fixed during gradients, after-model evaluation and probes |
| Conditions | Before model; after memory parameters with before classifier; complete recorded after model |
| Parameters | Keys, values, alpha and joint memory; classifier/full-model projections for context |
| Comparators | First-order conditional error CE along actual Adam movement and hypothetical -lr × recorded minibatch gradient; the latter is never applied |
| Adam terms | Stored/current first-moment numerator contributions using the actual updated second-moment denominator; not a reset counterfactual |
| Metrics | All/unique/repeat accuracy/CE and all-query correctness in all three conditions; fixed before-error CE; rescues/harms; projected CE contributions per block |
| Counts | 393,216 baseline targets; 1,179,648 condition answer evaluations; at most 1,572,864 additional probe evaluations |
| Numerical probes | Unit actual joint-memory movement; central spacings 1e-5/5e-6, both signs on private copies; skip norm ≤1e-10 or absent errors |
| Precision | CPU float64, one thread; projection sign unresolved within ±1e-10 |
| Verification | Saved initialized evaluator and actual audit entry; supplied tied logits/absent error groups; independent labels, matrix gradients/forwards, scalar Adam and projection sums; deliberate gradient/label/mask/step/term defects |
| Tolerances | Logits 1e-9; gradients 1e-9+1e-7 absolute reference; Adam/term reconciliation 1e-11; finite differences 1e-7+1e-4 absolute reference |
| Stop | Exactly these 16 saved steps, conditions and probes. No optimizer call, extra snapshot, reset, training extension, tuning or repair experiment |

Each step gets its own before-error subset. These conditional means cannot be combined into one unchanged error cohort across steps. All queries at load 32 are unique; repeat groups are empty/null. No held-out effect or whole-objective gradient is measured. The late steps start at 11000, not the earlier diagnostic's 12000 endpoint.

Memory-only after states are private mixtures of recorded before/after parameter blocks. They isolate a finite parameter intervention on these inputs, not an additive causal decomposition of the full update. Full-model first-order projections add across blocks; finite losses need not. Directional improvement need not change accuracy.

Primary sources inspected: [Kingma and Ba, Algorithm 1](https://arxiv.org/pdf/1412.6980) and [PyTorch 2.11 Adam](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html), first/second moments, bias correction and coordinate update. Projection and fixed-error construction are local derivations. Earlier trace auditing establishes selected replay fidelity, not directly logged historical head/memory movements or complete-history attribution.
