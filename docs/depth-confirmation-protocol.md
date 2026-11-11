# Depth confirmation protocol

Fixed before execution. Replicate the depth-versus-width advantage with new initialization/minibatch seeds and regenerated splits. This is a local replication, not an independent implementation or new task distribution.

| Choice | Fixed value |
|---|---|
| Models | Deep: 2 blocks, width 16, FFN 32, 4,896 parameters; wide: 1 block, width 24, FFN 40, 5,152 parameters; both 2 heads |
| Training | Adam 0.001, betas 0.9/0.999, epsilon 1e-8, no decay; 3,000 updates, batch 64; CPU float64 |
| Model seeds | 137, 163, 191, 223, 257; minibatch seed = model seed + 1000 |
| Three-pair splits | Train 8,192 / validation 1,024 / test 1,024; generator seeds 20261101/102/103 |
| Transfer | Four and six pairs, 1,024 each; seeds 20261104/105; exclude previous four/six-pair evaluation identities |
| Selection | Final checkpoint only; validation every 500 updates; no tuning |
| Primary criterion | Every deep seed >=90% training and three-pair test accuracy, and mean deep-minus-wide test advantage >=10 percentage points |
| Separate transfer criterion | Every deep seed >=90% accuracy at each longer length, reported separately by length |

The task and existing generator are unchanged: unique keys 0..7, values 8..11, marker 12, query one present key. Only the final query receives cross-entropy loss. Canonical identity is (sorted mapping, query); exclude identities across current splits, including reordered copies. Different queries about the same mapping may occur in different splits: this does not test unseen mappings. Independently check labels with key lookup, target counts and query-position counts. Record frequency-value (ties choose smallest value), copy-first and copy-last baselines.

The three-pair identity universe contains C(8,3)*4^3*3 = 10,752 cases, already extensively explored. Regenerated data cannot be called globally untouched. Report overlap of every new split with original training/validation/test identities; the earlier complement test also covered remaining identities. New test data are held out from these new models' training, not from all previous research. Four/six-pair confirmation excludes the original corresponding evaluation sets.

Reuse verified model and evaluation helpers. Check canonical disjointness, labels, model counts and existing forward/gradient/causal checks before training. Preserve split data/hashes, source/protocol hashes, final checkpoints and matched minibatch hashes. Report every seed, losses and failures. Approximate parameter control is not equal compute. No new architecture or binding-mechanism claim follows from passing accuracy alone.
