# Copy positional representation: fixed diagnostic

Question: does replacing additive sinusoidal positions with RoPE improve the observed long-copy failure under this recipe? Earlier RoPE results cover decoder-only retrieval, not encoder-decoder copying. This is an opened-data diagnostic, not fresh confirmation or a novel architecture proposal.

## Frozen design

Three paired seeds: 3119, 3121, 3137. Sinusoidal and RoPE arms start with identical learned tensors and use identical sampled indices. One candidate per arm, 6,000 Adam updates (.001, betas .9/.999, eps 1e-8), no clipping, schedule, decay, checkpoint selection or replacement seeds. Nonfinite training stops and is preserved. No automatic extension or recipe search. Historical baseline training fit motivates retaining its recipe; if either new arm misses fitting, report that limit rather than infer a pure transfer failure.

Reuse `results/copy_length_exposure/data.json`, with independent labels, exact-input split and simple baselines checked again. Pools are already opened; symbols and subsequences are shared. Train cycles lengths 3/4/5/6, equal distinct/repeated sampling. Length 10 has only a repeated group. Teacher inputs are BOS+source, targets source+EOS, all 16 classes available. Both objectives are mean CE over all content and terminal rows.

Both models have 1,888 learned parameters: width 8, two heads of width 4, one post-norm encoder and decoder layer, ReLU FFN16, dropout zero. Sinusoidal adds the existing position vectors to source and target embeddings. RoPE removes those additions and rotates projected Q and K in encoder self-attention, decoder self-attention and cross-attention. V, projections, masking, FFNs and normalization retain the same definitions. Removing additions and introducing rotations together defines the treatment; neither component is separately isolated.

For head width d=4, frequencies are (1, .01). Source and decoder positions each start at zero; decoder row j predicts source symbol j. This shared origin is an explicit copy-task convention, not a claim about translation alignment. The terminal decoder row has position n. No length scaling, frequency tuning or forced routing.

CPU float64, one thread, deterministic native Windows. The discarded full-step profile includes backward and allocated Adam states for both arms at every training length. GPU unused, GPU peaks unmeasured. Batch sizes 210/168/140/120 give 840 supervised rows per update. Each run sees 957,000 sampled sequences, 4,083,000 content targets, 957,000 terminal targets and 5,040,000 total targets. Record 57,657,000 visible attention pairs before the two-head factor. Equal exposure/parameters do not imply equal runtime; this custom implementation is not a speed benchmark.

## Evaluation and decisions

Evaluate every full training and validation pool at updates 1,000/3,000/6,000 without selection. Save all six final checkpoints and the pretest snapshot before scoring model test pools. Report per seed, length and eligible group:

| Measure | Meaning |
|---|---|
| Ordinary full exactness, teacher loss/accuracy, stopping rates | Greedy generation with learned stopping, cap n+3 |
| Oracle-length exactness/token accuracy | Exclude EOS, generate n symbols with the model's own prefix |
| Teacher content accuracy/CE, terminal CE/accuracy | Separate content from the terminal prediction |

Ordinary full exactness is primary. Oracle length is privileged and only diagnoses content. Do not suppress other output classes or force alignment.

Training fit: >=99% ordinary full exactness in every training group/seed, separately per arm. Seen-length reliability adds >=95% every test3/4/5/6 group. Scoped RoPE benefit requires both fit gates, >=10-point ordinary full-exact gains at **both** test8 and test10 in every pair, and no >5-point loss in any seen-length group. RoPE extrapolation requires its seen-length reliability plus >=90% every eligible group/seed at test8 or test10, separately. Preserve gains and threshold failures together; no pooled seed selection.

## Preflight and independent checks

Before training, run reusable tests on initialized models: explicit rotation matrices/manual exponential attention versus native SDPA, separate normalization/FFN equations, all parameter gradients and a directional difference. Check common-position-shift invariance, future exclusion, source/memory masks and intentional wrong K-rotation/frequency/mask controls. Exercise the actual final evaluator and audit cell after checkpoint reload at lengths 3/6/10, including absent distinct groups and supplied known-answer logits. Record the successful preflight separately; these are implementation tests, not trained-quality findings.

Audit every saved teacher logit, ordinary and oracle sequence with independent equations, scalar metrics and Python argmax; tolerance 1e-9 and zero prediction differences. Projection/shape utilities are shared between native and reference RoPE; the checks are not wholly disjoint implementations. Verify labels/splits, stream hashes, initial states, optimizer counters, exposure, criteria and deliberate criterion corruption. Hash relevant dependencies before/after. Finish after six runs, independent audit and short educational documentation.

## Paper boundary

[Su et al., RoFormer, section 3](https://arxiv.org/html/2104.09864v5) derives paired Q/K rotations and relative-position scores. Its section 4.1.2 describes rotary **self-attention** for translation. Rotating our cross-attention with a shared origin is a local application of the identity, not a reproduction of that experiment. Common-shift invariance does not imply longer-sequence success: extra keys change relative distances and attention normalization.
