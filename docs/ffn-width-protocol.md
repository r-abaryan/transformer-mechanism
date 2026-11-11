# FFN width32 versus64: frozen protocol

## Question and data boundary

With RoPE and ReLU fixed, does widening only the two FFN hidden layers improve latest-value recall? The earlier depth/width study changed model/head width as well; this comparison isolates FFN-width choice within a fixed recipe, including its necessary parameter/initialization changes.

The three-pair family space is exhausted:1,320 total, all used. Reuse exactly the train/validation/test three-pair datasets from `gelu_comparison_data.json` (160/20/40 families), preserving their split. The three-pair test is an exposed diagnostic, not fresh confirmation. Do not promote its test rows to training.

For four and six pairs, exclude every family from both `repeated_key_data.json` and `gelu_comparison_data.json`; retain their hashes. Available before this run:17,600 four-pair and237,200 six-pair families. New data seed20270106. Generate four-pair800 train/100 validation/200 test families and six-pair200 test only. Same single-duplicated-key task,36 variants per family, query always that key; retain all prior limitations on query dependence and familiar fragments. Check finite support before sampling; use a bounded sampler. Test labels/reversals/split isolation are audited before training; learned test predictions open only after all ten attempts finish.

## Models and training

Two pre-norm blocks, width16, two width8 heads, vocabulary19, standard RoPE offset+4, ReLU, final LayerNorm, dropout0. Baseline FFN hidden32 has5,088 parameters; hidden64 has7,200. Both embedding/output, attention and normalization tensors start identically within each seed. Construct the existing baseline first; in the wide arm replace both linear1 and linear2 with standard newly initialized Linear(16,64) and Linear(64,16), then convert to float64. FFN tensors are not identical or function-matched across shapes. Audit shared initialization separately from full model hashes.

Seeds1201/1217/1237/1259/1277. CPU float64, one thread, deterministic algorithms. Adam lr.001, betas.9/.999, eps1e-8; no decay, clipping or warmup. Batch64,6,000 updates, alternating three-pair odd/four-pair even. Advance both batch-index streams every update with seed+1000/+2000. Save validation at0/1000/.../6000 and latest model/Adam/RNG checkpoints. Final weights only; failed attempts remain without replacement.

Equal3,456,000 tokens per run, not equal compute. Each FFN has `2*d*f+f+d` parameters. Across two blocks, added parameters are2,112 (+41.51%). The FFN projection multiply-accumulates per token per block are `2*d*f`,1,024 versus2,048. FFN hidden activations double in size; whole-model runtime and memory do not necessarily double. Record elapsed loop time without a speed benchmark claim.

## Verification, criterion and stop

Before training: narrow forward/gradients must match the prior RoPE/ReLU implementation within1e-10/1e-9. Wide forward/gradients must match independent matrix-rotation/SDPA attention within1e-10/1e-9. Check causal masking and parameter counts. A separate zero-padding witness must reproduce the narrow model in the wide shape within1e-10; this demonstrates containment, not the actual training initialization. Check shared initial tensors for all paired seeds.

After training: audit unchanged hashes, data exclusions/reuse, regenerated streams, initial tensors,6,000 optimizer steps and30 independent test-logit evaluations within1e-10. Report accuracy/CE at3/4/6, six-pair family/reversal accuracy, obsolete-value rate and latest-position accuracy.

Keep width64 only if all wide seeds reach six-pair >=90%, mean paired six-pair gain >=2 percentage points, no seed loses six-pair accuracy, every wide seed reaches >=99% at both3/4 pairs, and mean loss relative to32 is <=.5 points at each short length. The3-pair preservation check is diagnostic because its test is reused. Engineering thresholds, not significance tests. Stop after ten attempts and audit; no extra seeds, widths, budgets or tuning on these tests.

[Learning guide](ffn-width-guide.md) explains the math, what extra hidden units mean and why width differs from heads or depth.
