# Broader query exposure: frozen protocol

Question: does asking about every present key during training improve query transfer without an architecture change? Two conditions: repeated-only versus all-keys. Both use RoPE/ReLU/FFN32, two pre-norm blocks, width16, two heads, vocabulary19, dropout0 and5,088 parameters. No architecture changes.

## Data and matching

Generator seed20270107. Reuse exactly the three-pair context splits from the GELU study (160/20/40 families); their tests are exposed diagnostics. Generate fresh four-pair800/100/200 train/validation/test families and six-pair200 test families, excluding the union of repeated-key, GELU and FFN-width data. Available before generation:16,500 four-pair and237,000 six-pair families. Same 36 order/value variants per family. Exactly one duplicated key, all values distinct. No six-pair training/validation.

For every context, expand queries in this order: repeated key first, then all unique keys sorted numerically. All-keys training samples uniformly from that expanded list. Repeated-only training samples the identical index, but resets the query and label to the repeated key and its latest value. Thus contexts, sampling frequency, sequence lengths and batch count match exactly. Query/label coverage changes. The all-keys arm allocates only1/(n-1) of updates in expectation to repeated queries at n pairs; this tradeoff is part of the tested recipe, not extra training. Save exact query counts.

## Training and evaluation

Seeds1301/1321/1327/1361/1373. Identical learned initialization within pairs. Adam.001, betas.9/.999, eps1e-8; no decay, clipping or warmup. CPUfloat64, one thread, deterministic algorithms. Batch64,6,000 updates; alternate3-pair odd/4-pair even. Both index streams advance every update using seed+1000/+2000. Same3,456,000 tokens per run. Save latest model/Adam/RNG state, validation at0/1000/.../6000, batch hashes and elapsed loop time. Final checkpoints only; open tests after all ten attempts. Preserve nonfinite failures without replacements.

Evaluate all query types in both arms. Report accuracy and CE separately for repeated and unique queries at3/4/6 pairs, plus overall accuracy and all-query-context correctness. Report paired gains, not comparison to older different-data studies. Context variants and queries within a family are dependent.

Adopt broader query exposure only if every all-keys seed scores >=90% unique-query accuracy at both4 and6 pairs, mean paired unique-query gains >=5 percentage points at both lengths, no seed loses unique-query accuracy at either length, every all-keys seed has >=99% repeated accuracy at3/4 pairs, and mean repeated-query losses are <=.5pp at3/4 and <=2pp at6 pairs. These fixed engineering thresholds are not significance tests. Three-pair preservation is diagnostic only. Stop after ten runs and audit; no tuning, extra seeds or budget extension.

## Verification

Reuse verified unchanged forward math after hash checks. Before training audit context-family exclusions, reused splits, query expansion, labels by reverse scan, and paired input identity outside the last token. Ensure baseline rewrite changes unique queries/labels and leaves repeated queries unchanged. After training independently regenerate streams and exposure counts; check initialization, protocol/source/data/checkpoint hashes,6,000 Adam steps and all30 test evaluations with independent matrix-rotation/SDPA attention within1e-10. Save logits. Failure may reflect coverage tradeoffs or finite-budget learning, not architecture incapacity.
