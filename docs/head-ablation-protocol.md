# Head contributions: bounded diagnostic

Fixed before execution. Earlier head-token experiments patched donor Q/K in shifted block-2 inputs. This study instead removes head outputs from ordinary inference in both blocks.

Use all five pre-norm constant-rate checkpoints at 6,000 updates from norm-extension, seeds 601/631/659/683/719. Reuse its 400 mappings, 7,200 variants, offset 0. These examples are already opened. This is exploratory, with no training, model selection, pruning claim or fresh confirmation.

Seven conditions per seed: ordinary; remove head 0, head 1, or both in block 1; the same three removals in block 2. Remove the selected head's AV tensor at every query position before output projection. Retain the shared output bias, residual, normalization and FFN. Downstream computation remains live. No rescaling of surviving heads.

Report accuracy, cross-entropy, all-variant mapping correctness, paired prediction changes/rescues/harms and accuracy by queried pair rank. Report descriptive loss interaction L(both)-L(h0)-L(h1)+L(base). It is not a causal share. Head indices have no aligned semantic meaning across seeds. Do not label a head dispensable from accuracy alone.

Inspect ordinary attention only: per head, mean final-query distribution over eight positions, target-value mass/top frequency, mean projected contribution L2 at final query; also each value-row's attention mass on its immediately preceding key. None alone establishes a mechanism. Target positions are label/query-aware diagnostics.

Verify explicit ordinary logits against saved baseline (<1e-10). Verify head contribution sum plus shared bias equals attention output. Independently implement removals by zeroing matching columns of output projection in a reloaded library model and compare every example's logits (<1e-10). Audit saved logits/metrics and unchanged checkpoint hashes. Stop after 35 conditions; no adaptive head search.
