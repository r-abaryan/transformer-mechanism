# MHA versus MQA: frozen training comparison

Question: can one shared K/V head halve cache payload while preserving recall quality under this fixed training recipe? Five paired seeds: 1411, 1423, 1451, 1481, 1499. No conversion of trained checkpoints.

## Architecture and initialization

Both arms: vocabulary 19, model width 16, two pre-norm blocks, two query heads of width 8, value width 8, RoPE at offset 4, ReLU FFN width 32, final LayerNorm, dropout zero. MHA has two K/V heads; MQA has one. Keep both query heads and the same output projection. Expected parameters: 5,088 versus 4,544. Expected K/V payload halves at equal batch/length/precision.

Initialize a common untrained MHA template for each seed. Copy embeddings, norms, FFNs, Q and output projections exactly into both arms. MQA takes the first template K/V head, fixed before any training. This preserves each initialized element's marginal distribution and avoids the variance reduction of averaging independent heads. It is a from-scratch paired initialization recipe, not the paper's pretrained mean-pooling/uptraining procedure. Record full/common/retained-KV hashes.

Separate projection modules avoid inactive parameters. Verify the MHA function against the archived dense implementation, and both arms' outputs/parameter gradients against independent matrix-rotation/SDPA arithmetic at <=1e-10 before training. Change a future query token as a causal control: earlier logits must remain unchanged; final logits must differ by >1e-6 in each arm. Measure compact K/V shapes/bytes directly.

## Data and exposure

Generator seed 20270108. Exclude the union of repeated-key, GELU, FFN-width and query-exposure families. Reserve fresh four-pair 800/100/200 train/validation/test families and six-pair 200 test families before training. Reuse the exhausted three-pair splits unchanged (160/20/40 families); their evaluation is diagnostic. Exactly one duplicate key and distinct values; 36 order/cyclic-value variants per family. Six-pair data is test only.

Expand each context to queries for all distinct present keys, duplicated key first then sorted unique keys. Labels are the latest value for the queried key. Both arms receive identical expanded-row indices, inputs and labels. Freshness concerns whole canonical families; smaller familiar fragments are allowed. Audit exclusions, variant structure and labels independently.

## Budget, evaluation and fixed decision

CPU float64, one thread, deterministic algorithms. Adam lr .001, betas .9/.999, eps 1e-8, no weight decay/warmup/clipping. Batch 64, 6,000 updates; alternate three-pair odd/four-pair even. Both index streams advance every step using seed+1000/+2000. Each run processes 3,456,000 tokens and the same attention-pair count. MQA has fewer projection operations/parameters, so this is equal exposure/steps, not equal FLOPs. Explicit K/V broadcast temporaries are excluded from compact cache payload; no speed claim.

Save latest model/Adam/RNG state, validation at 0/1000/.../6000, stream hashes, query counts and loop times. Finish all ten attempts before opening final tests. Preserve nonfinite failures without replacement. Final checkpoints only. The archived runner is adapted in a new file to preserve executed sources; existing family identity, row expansion and metric helpers are reused.

Report accuracy/CE separately for repeated and unique queries at 3/4/6 pairs, overall and all-query-context accuracy, every seed and paired differences. Adopt MQA for this recipe only if all conditions hold:

- Parameter count is 4,544 and compact K/V payload is exactly half MHA.
- Every MQA seed reaches >=99% repeated and unique accuracy at 3/4 pairs; three-pair evidence remains diagnostic.
- Every MQA seed reaches >=90% repeated and unique accuracy at six pairs.
- Mean paired MQA-minus-MHA accuracy is >=-1 percentage point for each query type at four and six pairs.
- No paired seed loses more than 5 points for either six-pair query type.

These are engineering acceptance thresholds, not significance tests. A failed rule does not erase savings or establish a universal capacity limit. One split and five initialization seeds do not estimate every source of variability.

Independently audit source/protocol/data/checkpoint hashes, initial matching, stream/count regeneration, 6,000 Adam updates, payload and all 30 final checkpoint/length evaluations at <=1e-10; save logits. Close after ten runs and this audit. No new seeds, head-selection tuning, budget extension or automatic GQA sweep.

Context: [Shazeer 2019](https://arxiv.org/html/1911.02150v1), [GQA 2023 v3](https://arxiv.org/html/2305.13245v3), and [our shared-KV mathematical findings](shared-kv-results.md). Their benchmark speed/quality claims are not reproduced by this toy-task comparison.
