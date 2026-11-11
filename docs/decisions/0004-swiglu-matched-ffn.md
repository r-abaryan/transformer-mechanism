# Decision0004: SwiGLU at matched FFN projection cost

Date: 2026-10-01. Frozen before new model-quality evaluation.

## Question and control

Does a SwiGLU FFN improve fixed-budget word prediction over ReLU at equal FFN weight count and projection multiply-adds? This is a package comparison (gating plus narrower hidden width), not isolation of gating from width.

Use the pinned WikiText-2 encoding and evaluation windows from [Decision0003](0003-wikitext2-matched-baseline.md) and its [execution protocol](../wikitext2-execution-protocol.md). Keep width16, two pre-norm blocks, MQA with two query/one K/V heads, RoPE offset4, LayerNorm gain/bias/epsilon, untied vocabulary matrices and causal softmax fixed.

Both FFNs are bias-free. ReLU uses hidden33, SwiGLU hidden22: `2*16*33 = 3*16*22 = 1056` weights and forward projection MACs per token/block. SiLU, the elementwise product, activation storage and backward operations differ; full compute/runtime is not equal. This changes the earlier biased width32 control in both arms; historical scores are not a matched baseline. The original recipe is not overwritten.

## Initialization and budget

Three new paired seeds:1783,1801,1823; six attempts total. Instantiate the existing external MQA template with the same seed and vocabulary, preserving all non-FFN tensors exactly within each pair. Replace FFNs using a separate CPU generator seeded `seed+3000`. Draw gate/input projection, content projection and output projection in that order from uniform `[-1/sqrt(fan_in),1/sqrt(fan_in)]`. For ReLU omit the content draw. FFN initializations differ in shapes and random draws; they are not function-matched.

Each arm:6,000 Adam updates, lr.001, default betas/epsilon, batch8/context64,3,072,000 next-word targets. Replay paired training starts with generator `seed+1000`. No dropout, warmup, clipping, weight decay, tuning or checkpoint selection. Alternate which arm runs first by seed. Use native Windows CUDA FP32, deterministic settings, TF32 off and one CPU thread. Record loss/gradients/updated-parameter finiteness at every update, source/data hashes, sampled-start hashes, optimizer counters and final checkpoints. Retain failed attempts; they fail adoption.

## Metric and stopping rule

Primary metric: **final validation** mean next-word NLL; lower is better. Acceptance requires all six finite completed attempts, both arms at least.05 nats/token better than the training-count add-one unigram validation baseline, geometric mean SwiGLU/ReLU validation PPL ratio<=.98, and every paired seed ratio<=1.00. The2% threshold is a research target, not a significance claim. No best checkpoint or favorable-seed selection.

Freeze the decision after all six validation results. Only then score the official test split descriptively. Both official validation/test were used in an earlier architecture comparison: they are reused benchmark evidence, not a fresh confirmation set. Test results cannot change adoption. No local recall retention claim is made; this does not automatically replace the original recall recipe.

Stop after one independent audit and one result document, regardless of outcome. No extra seeds, width sweep, budget extension or tuning on opened splits.

## Numerical verification

Before training, check SiLU/product analytical derivatives against autograd and finite differences; check full FFN input/weight gradients, cost counts, negative/amplifying gate witnesses and deliberately wrong sigmoid-only/omitted-product controls. Independent Decimal scalar arithmetic supplements tensor checks.

All three paired actual-vocabulary models must pass an independent CPU-float64 forward/gradient reference and CUDA FP32 preflight: double output/gradient absolute error<=1e-10; GPU output absolute<=1e-4 and relative L2<=1e-5; GPU gradient absolute<=1e-5 and relative L2<=1e-4; causal prefix error<=1e-6. A biased width32 ReLU compatibility fixture must agree with the previous immutable implementation. Reference attention uses native SDPA and independently expanded normalization/RoPE, not the producer's explicit softmax.

Retain every final validation/test token loss and prediction plus the first two complete blocks and any partial final block's logits. Independently recompute **all** validation/test losses using CPU float64 and saved FP32 weights: max token NLL error<=1e-4, mean<=1e-5, fixture logits<=1e-4. Count argmax differences descriptively. Replay initialization/starts, recompute unigram, exact coverage, resource counts and the validation-only decision. A failed implementation check is archived/resolved; it is not a quality finding.

## Sources and boundary

[Shazeer2020, equations2/6 and sections2-3](https://arxiv.org/html/2002.05202v1) defines bias-free gated FFNs and two-thirds width matching; its larger T5/C4 denoising results motivate, but do not predict, this comparison. [PyTorch2.11 SiLU](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.silu.html) specifies `x*sigmoid(x)`. Gated FFNs act within a token; they do not replace attention across tokens or guarantee better gradient flow. No exhaustive literature survey, paper reproduction, speed claim, convergence claim or novel architecture is intended.
