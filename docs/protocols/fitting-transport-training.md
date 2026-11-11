# Cross-block curvature action during fitting

Frozen after runtime preflight and complete-step profiling, before formal training, 2026-10-05. One question: does removing routing/decoder curvature transport worsen the observed finite-pool fitting quality?

The [saved review](../assessments/fitting-transport-review.md) finds transport and cancellation, but cannot establish benefit. The primary prediction is **strictly more final all-load training errors after removal in every seed**. Publish each seed's error difference and the conjunction before held-out scoring. Equal or fewer errors in any seed fail the all-seed benefit rule. CE and complete maps remain required measurements, not replacement criteria. No recipe selection, extra settings, seeds or budget extensions.

## Locked intervention

Full direction: $d=-Hg$. Removal: $d=-(P_RHP_R+P_DHP_D)g$, where R contains write_logit/key.weight (513 parameters, tied queries included), and D contains value.weight/output.weight/output.bias (102). Apply the existing two-loop operator twice to masked gradients and retain only the matching output blocks.

Both branches keep global secant pairs, initial scalar H_diag, memory size and the native line search. Each branch forms curvature from its own subsequent trajectory. This is not two independent optimizers, frozen historical curvature or an effective-decoder scale intervention. Positive-curvature inverse-BFGS updates preserve positive definiteness; their principal blocks give a descent direction for nonzero g. That does not establish convergence or fitting benefit.

| Factor | Both branches |
|---|---|
| Starts | Saved Adam-12000 weights, seeds 3473/3479/3481 from fitting confirmation |
| Model/data | Unchanged 615-parameter zero-start delta reader; exact previously opened confirmation pools |
| Labels/splits | Latest assigned value; constant value per key within each history; train/validation/test 768/192/384 histories per load, with six correlated cyclic value shifts |
| Loads/objective | 2/8/16/32 keys; equal mean of four exhaustive-query CEs; 3,072 histories and 44,544 answers per gradient |
| Cap | At most 128 closures/gradients per arm, including rejected trials; stop with fewer than two remaining |
| Solver | lr 1, max_iter 1, max_eval min(26, remaining minus 1), history 20, gradient tolerance 1e-10, change tolerance 1e-12 |
| Search | Installed strong-Wolfe implementation, c1=1e-4/c2=0.9; original initial trial-length policy |
| Runtime | CPU float64, deterministic, one thread, serial native Windows execution |
| Reporting | Training at start, crossed 32-gradient milestones and final; final validation/test for all loads. CE/accuracy/complete maps, all/unique/repeated groups and explicit absences |

First direction/search must be identical. Full controls must exactly reproduce all historical native trials, final weights and optimizer states. Later step lengths, trial counts, updates and curvature may diverge as intervention consequences. The cap matches feedback allowance, not equal updates or wall time. Same-length held-out scores are descriptive; no length-transfer test is added. Shifted histories and optimizer calls are not independent seeds.

## Execution checks and stopping

The installed optimizer has no public direction hook. A hash-bound source adapter changes only the direction assignment immediately before g dot d. The original trainer, objective, scorer and snapshot path are reused; executed sources are preserved separately.

Two preflight methods passed before formal execution. They exercise an initialized saved model through both actual training/evaluator/audit paths, a native quadratic null control, separate dense block-direction arithmetic, supplied tied logits/empty groups and intentional gradient/direction/counter/loss/label/logit/missing-group/decision faults. Independent matrix gradients/labels, native state/trial replay and every score cell are required in the final audit; independently count final errors from logits. Measure accepted-step Wolfe inequalities, including failed conditions if any.

Two complete optimizer calls per arm were profiled: 3 then 2 gradients each, 1.18..3.53 seconds per call. Optimizer tensor payload was 9,840 then 19,704 bytes; this is not peak process memory. Profiling is excluded from formal exposure. CUDA allocated/reserved memory is inapplicable because execution is on CPU; no GPU throughput claim follows. Serial execution avoids the previously documented restricted Windows multiprocessing path.

Stop after six capped runs and their audit. An implementation/audit failure requires preserved evidence and a separate diagnosis, not silent overwrite. This comparison tests a procedure contribution on reused starts/pools; it does not establish a generalization repair, universal transport necessity, an architecture change or why all earlier training chose particular representations. RL stays parked.

Inspected [PyTorch 2.11 documentation](https://docs.pytorch.org/docs/2.11/generated/torch.optim.LBFGS.html) and [versioned implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/lbfgs.py), including secant updates, two-loop application and strong-Wolfe search. [Liu/Nocedal (1989), author abstract](https://users.iems.northwestern.edu/~nocedal/Abstracts/limited-memory.html) supplies limited-memory/convex-theory scope. The partition, intervention and error prediction are local; sources do not forecast these scores or prove neural convergence.
