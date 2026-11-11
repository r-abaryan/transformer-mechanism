# Sinusoidal versus RoPE: comparison protocol

Selected before implementation, data generation and training. One baseline and one candidate; ten runs total. No alternative selection from test outcomes. See the [learning guide](positional-comparison-guide.md).

## Architecture and execution

Both arms: vocabulary19, two pre-norm blocks, model width16, two heads of width8, FFN32/ReLU, final LayerNorm, causal attention, dropout0. Preserve all projection biases and initialization behavior of the existing model. Pair every learned tensor exactly at initialization. Expected parameter count 5,088 in both arms; verify before training.

Baseline adds existing sinusoidal PE once to embeddings. RoPE removes that addition and rotates projected Q/K after projection including bias, in both blocks. Rotate adjacent coordinate pairs, all eight head coordinates; frequencies `10000^(-2r/8)`, r=0..3. Do not rotate V. No frequency scaling, interpolation or learned positional parameters. Position indices start at +4 for both arms.

Use one common explicit attention implementation for both arms to avoid a kernel confound. CPU float64, one thread, deterministic algorithms. This is a learning comparison, not a GPU benchmark. Same Adam .001, betas .9/.999, eps1e-8, no weight decay, clipping, loss scaling or warmup. Seeds 907/929/953/977/997. Batch64; exactly 6,000 updates. Both arms alternate three-pair batches on odd steps and four-pair batches on even steps. Identical paired minibatches; no six-pair training/validation. Equal examples/tokens, but rotations change arithmetic cost. Record parameters, tokens, elapsed time and the arithmetic difference; do not claim equal FLOPs.

## Fresh data and evaluation

Use the distinct-value associative-recall task and mapping/order expansion from the length-exposure study. New generator seed 20270103. Exclude all full mappings in that study's exclusion manifest plus every mapping in its train/validation/test datasets. Before generation, check for any later task-data artifacts and include their mappings if present. Save the exact exclusion inventory and hashes. Familiar smaller submappings may remain; do not call those novel.

Mapping counts: three pairs train1,600/validation200/test400; four pairs train1,605/validation210/test420; six pairs test420 only. Balance value subsets as before. Six orders per mapping, all queries per order. Both arms use the same splits. Advance independent three/four index streams every update, seeded model_seed+1000/+2000; use the appropriate stream at each step. Hash streams.

Save model/Adam/RNG states and validation metrics at steps0/1000/.../6000. Final checkpoints only. Open test predictions after all ten training attempts. Nonfinite training is a failed attempt; no changed recipe or replacement seed. Report each seed's accuracy, cross-entropy, queried-rank accuracy and all-variant mapping correctness at three/four/six pairs. Predictions sharing a mapping are dependent.

## Verification before interpretation

Before training: baseline outputs/gradients must match the existing library implementation within 1e-10/1e-9 on fixed synthetic fixtures. Check RoPE pair rotations against explicit rotation matrices, norm preservation and the relative-position identity within 1e-10. Check future-token masking and global-offset invariance of the RoPE-only model within 1e-10. Use finite differences on a small synthetic gradient fixture, tolerance 1e-6. No held-out task data for implementation checks.

Before accepting results: audit mapping exclusion, label/rank balance, paired initial tensors and streams, unchanged protocol/source hashes, final checkpoint metrics and saved logits using an independent forward calculation. A demonstrated implementation bug must be documented and corrected before a valid experiment; never silently amend a completed run.

## Fixed decision and stop

Keep RoPE for this task/budget only if every seed reaches six-pair accuracy >=90%, mean paired six-pair gain is >=5 percentage points, no seed loses six-pair accuracy, every RoPE seed reaches three-pair accuracy >=99%, and mean three-pair accuracy drops by at most .5 points. Otherwise reject this fixed recipe while retaining any partial gains. These are engineering thresholds, not statistical significance tests.

Stop after ten attempts and the audit. No extra seeds, temperature changes, frequency tuning, longer training or switching to ALiBi after seeing the test. A success is a scoped comparison of established positional representations, not architectural novelty or general LLM superiority.
