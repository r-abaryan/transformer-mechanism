# Learning with a source-end marker

2026-10-03. One fixed-recipe learning diagnostic, not fresh confirmation. Earlier marker insertion used checkpoints trained without source EOS. This run supplies the marker during every update. Both sinusoidal and RoPE arms receive it; no paired unmarked arm is trained, so historical comparisons cannot isolate the marker's training effect.

## Frozen task and recipe

Three fresh paired seeds: 3163/3167/3169. Exactly 6,000 Adam .001 updates per model (betas .9/.999, eps1e-8), no decay, clipping or schedule. One candidate per arm, no selection, replacement seed or automatic extension. The retained recipe previously met both random-pool training-fit gates. If fitting fails here, record a recipe failure rather than infer an architecture or pure transfer limit.

Reuse opened `results/copy_length_exposure/data.json`. Training cycles content lengths 3/4/5/6 with identical sampled indices and equal distinct/repeated minibatch counts. Exact-input splits share symbols/subsequences. Length 10 has no distinct group. Encoder input is **content+EOS**; decoder input remains **BOS+content**; targets remain **content+EOS**. The source marker is not another content target: do not add a second EOS label. All 16 output classes remain available.

Models retain 1,888 learned parameters: width 8, two width 4 heads, one post-norm encoder/decoder, ReLU FFN16, dropout zero. Both models start from identical tensors. Sinusoidal positions and the declared RoPE self/cross rotations retain their previous definitions and origin zero. Encoder/decoder embedding tables are separate. The new adapter changes source formatting and extends the source padding mask by one visible token; frozen evaluators still receive content lengths.

Profiled CPU float64, deterministic native Windows, one thread; GPU unused and GPU memory peaks unmeasured. Complete-step profile includes allocated Adam states and is discarded. Batches 210/168/140/120 keep 840 target rows/update. Each run samples 957,000 sequences, 4,083,000 content targets and 957,000 terminal targets, totaling 5.04M supervised targets. It also reads 5.04M source tokens, including 957,000 markers. Visible attention work is 71,820,000 pairs before the two-head factor. Both arms match these counts; this is greater source/attention work than the unmarked historical setup, not a speed comparison.

## Measurements and rules

At updates 1,000/3,000/6,000 evaluate all full training/validation pools and short uniform probes without selection. Save all six final checkpoints and a pretest snapshot before scoring test pools or longer uniform probes. Ordinary greedy generation stops at EOS with cap n+3; oracle content excludes only EOS and generates n symbols using its own prefix. No forced routing, oracle stopping in the ordinary metric or target feedback during generation.

Report full exactness, teacher CE/accuracy, content accuracy/CE, EOS stopping rates, terminal accuracy/CE and oracle content per seed/length/group. Fit requires >=99% ordinary full exactness in every training group/seed. Seen-length reliability requires that fit plus >=95% every test3/4/5/6 distinct/repeated group. Extrapolation at each length8/10 requires seen reliability plus >=90% every eligible test group/seed. RoPE benefit requires both fit gates, >=10-point ordinary exact gains at both longer lengths in every pair and no >5-point regression in any seen-length group. Preserve descriptive results when these rules fail.

Separately evaluate eight uniform sources per length3/4/5/6/8/10. Do **not** add them to training. Record exact overlap with each original split. Short uniform-copy rule is >=90% ordinary full exactness at each short length in every seed, separately per arm. These partly seen/previously inspected challenge cases are not a fresh held-out split or an additional training-fit gate. Longer uniform copying is descriptive and does not replace random-pool transfer metrics.

## Preflight, audit and stopping

Before training, all 3 x 3 x 2 =18 initialized cases at content lengths3/6/10 exercise equations, all parameter gradients, checkpoint reload, final evaluator and the actual independent audit cell. Check supplied correct logits and absent length10 groups. Masking the marker is an intentional wrong input condition; check source/target boundaries and an erroneous duplicate EOS target. Verify a nonzero gradient reaches the source-marker embedding. Independently reconstruct criteria and inject a fit error. Counts are derived from declared factors.

Final audit: 3 seeds x 2 arms x 20 pools =120 cells, every teacher logit and ordinary/oracle sequence, independent scalar metrics, initial states, streams, optimizer steps, exposure and criterion decisions. Tolerance1e-9 and zero prediction differences. Explicit rotations/manual attention and separate layer equations share projection/shape utilities with the producer. Relevant hashes must remain unchanged. Stop after six runs, audit and concise educational documentation. No longer budget, marker grid or extra recipe is selected automatically.

The [saved-marker diagnostic](../topics/benchmarks/copy-source-end-marker.md) motivates this learning question. [RoFormer, section 3](https://arxiv.org/html/2104.09864v5) supplies the existing rotation identity. Neither source establishes the new training outcome; no paper reproduction or universal positional winner is claimed.
