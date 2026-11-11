# Where positional-exposure training helps copying

Protocol date: 2026-10-07. Execution is authorized only after complete-step profiling, initialized evaluator/audit preflight and a source/data hash lock. Question: does the previously observed long-copy repair require exposure in self-attention, cross-attention, or their combination?

## Four conditions

| Arm | Encoder and decoder self-attention IDs | Cross-attention query/key IDs |
|---|---|---|
| ordinary | Consecutive | Consecutive |
| self_only | Sampled | Consecutive |
| cross_only | Consecutive | Sampled |
| both | Sampled | Sampled |

Draw one sorted subset of n+1 IDs from 0..10 per batch. All four arms consume identical position draws; ordinary ignores them. Enabled sites share the same draw. Source content+EOS and decoder BOS+content use corresponding IDs. Causal masks follow token order. Evaluation uses consecutive IDs everywhere; fixed-layout fit probes activate only each arm's trained sites. For probe generation beyond the supplied positions, extend by unit increments.

Keep the prior 1,888-parameter marked RoPE model: width 8, two heads, one encoder/decoder layer, postnorm, ReLU FFN16, no dropout. Reuse labels, tokenizer and metrics. Do not edit the frozen predecessor. Forward control is separate only to expose independent positional sites; shared helpers and reference-path limits are disclosed.

[Ruoss et al. (2023), section 3](https://arxiv.org/html/2305.16843v1#S3) was inspected for sorted sampling without replacement once per batch, including RoPE. The method motivates the exposure manipulation, not local score predictions. Our site ablation, small range and ordinary test IDs are a local adaptation, not a reproduction or novelty claim. The rotation identity is `(R_p q)^T(R_r k)=q^T R_(r-p)k`. Selecting a training site still changes gradients throughout the network; it does not freeze other components.

## Data, fit and budget

Reuse short train/validation/test3/4/5/6 and historical uniform3/4/5/6/8/10 pools. These are opened evidence. Reserve 128 new random sequences per long length 8 and 10, seed 2026100721, excluding every exact sequence in the original marker pools and prior positional-exposure pools. Length 8 has 64 distinct/64 repeated cases; length 10 has 128 repeated cases and no distinct group. All confirmation seeds share these pools. Four fixed sampled training layouts per length use seed 2026100722; apply each to the complete training pool. They represent positional exposure, not exhaustive layout coverage.

One development quartet, seed 3761. Candidates: 6,000 updates, then continue ALL four to 12,000 only if any training gate fails. Select the first budget passing every arm. No learning-rate, seed or position-range alternatives. If both stages fail, stop and leave long tests unused. Confirmation: three fresh quartets, seeds 3767/3779/3793, same locked budget from scratch. Never replace a failed confirmation arm or extend it. Save all twelve checkpoints and a pretest manifest before any reserved test scoring.

Fit gate: >=99% full generated copying in every ordinary train3–6 distinct/repeated group, all arms. Each enabled arm must also reach >=99% in every corresponding fixed-layout training group. Ordinary layout cells are native duplicates and gate only once. Validation losses/accuracy are descriptive. Final confirmation includes full training, validation, short/long test and uniform scores even if a confirmation gate fails; such failure blocks a clean site attribution.

Adam lr .001, betas .9/.999, epsilon 1e-8, no decay/clipping/schedule. Cycle lengths 3/4/5/6 with batches 210/168/140/120 and equal distinct/repeated examples. Paired initial tensors, token stream and sampled position stream are identical within each quartet. Every update supervises 840 tokens including EOS; source tokens also total 840. Effective batch equals physical batch, no accumulation. A 6,000-update run uses 957,000 sequences/5.04M target tokens. First-stage success gives 16 runs, 96,000 formal updates, 15.312M sequences and 80.64M target tokens; a 12,000 selection doubles these exposures. Record all stages, optimizer/RNG state and minibatch loss every 1,000 steps.

CPU float64, one thread, native Windows. Before freezing, profile three complete forward/backward/Adam steps for every arm/length: 4×4×3=48 discarded steps. Record times and optimizer bytes. GPU unused, allocated/reserved memory null; CPU peak memory unmeasured. Profile timings are not architecture speed findings.

## Measurements and decision

Use existing teacher-forced CE, token/content accuracy, generated exact/content exact accuracy, EOS rates, first-token accuracy, and first-source outranking in either cross-attention head (tolerance 1e-10). Decode all 16 classes, own prefixes, EOS stop, cap n+3. No oracle route or supplied content. Report every group; absent groups are null.

Primary cells: three confirmation seeds × (length8 distinct, length8 repeated, length10 repeated) = nine groups per arm. First require all-arm fitting and no arm losing more than 5 percentage points on any short-test group. Replication requires both to reach >=99% full copying and improve copying AND outranking by >=10 points over ordinary in every primary cell. If replication fails, report mixed outcomes without assigning a unique site.

Conditional single-site sufficiency: the single arm reaches >=99% full copying, stays within 5 points of both, and improves copying/first-source outranking by >=10 points over ordinary in every cell. Evaluate self_only and cross_only separately. If both pass, either tested intervention is sufficient; if only one passes, localize the effective intervention; if neither passes but combined does, only the combined recipe meets this rule. Do not infer universal necessity from threshold failures. Report the continuous accuracy interaction `A_both − A_self − A_cross + A_ordinary` in each cell, not a significance test or additive causal share.

## Verification and stop

Preflight: twelve initialized output/all-parameter-gradient cases, native and combined predecessor equality, explicit calls verifying each site's IDs, wrong-site error control, independent label shifting, supplied logits/ties/absent groups, actual saved initialized training/evaluation/audit entry and deliberate metric/missing-cell/stream/selection/arm faults. Isolated fixtures use eight updates.

Each development arm has 8 ordinary train/validation pools + 4 lengths × 4 layouts = 24 cells. Each confirmation arm has 20 ordinary pools + 16 layouts = 36 cells. First-stage success audits 4×24 + 12×36 = 528 cells; second-stage success adds 96 = 624. Development failure audits 8×24 = 192 cells. Each cell contains three group slots, some absent; evaluations are not independent seeds.

Audit independently rebuilds token/position streams, fit/decision reductions, greedy generation, scalar metrics and explicit rotation/normalization/softmax reference calculations. Require <=1e-9 score/metric error and zero prediction differences. Check source/protocol/data hashes, fresh-long disjointness, paired starts, checkpoint hashes, Adam counters, selection and pretest binding. Forward structure/projection helpers are shared; full training gradients are not replayed. Preserve previous evidence. Stop after the declared development, confirmation, audit and educational report; consolidate the copy findings next, without automatically adding a range or head sweep.
