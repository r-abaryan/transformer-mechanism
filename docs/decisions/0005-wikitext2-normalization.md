# Decision0005: normalization transfer to WikiText-2

Date: 2026-10-01. Fixed before new model-quality evaluation.

## Question and control

Does gain-only RMSNorm improve final word-prediction validation perplexity over gain-only LayerNorm under the established fixed training budget?

Use Decision0003's pinned encoding and nonoverlapping evaluation windows. Both arms keep width16, two pre-norm blocks, MQA (two query heads, one KV head), RoPE offset4, biased ReLU/FFN32, untied vocabulary matrices and causal softmax. Replace all five normalization sites together. Both learn gain only, initialized to one, with epsilon1e-5. LayerNorm subtracts the feature mean; RMSNorm uses the uncentered second moment. Both have `32*vocabulary+3856` parameters (1,068,752 for vocabulary33,278). This follows the previous gain-only recall comparison. Both omit the original external baseline's80 normalization bias parameters; their removal is not separately tested or adopted. Historical baseline scores are not paired controls.

## Initialization, budget and stop

Three new paired seeds1861,1877,1901; six attempts. Instantiate the existing external MQA template with each seed, then replace normalization modules. Every initial parameter value must match within a pair. Both receive identical sampled training starts (CPU generator seed+1000).

Each:6,000 default Adam updates at lr.001, batch8/context64,3,072,000 targets. Native Windows CUDA FP32, deterministic settings, TF32 off, one CPU thread. No warmup, clipping, dropout, weight decay, tuning or favorable checkpoint selection. Alternate arm order by seed. Check every update's loss, gradients and updated parameters for finiteness. Preserve failures, optimizer/RNG states, checkpoints, source/data hashes and sampling hashes. Do not extend training after a miss.

Primary outcome: final validation mean NLL. Adoption requires all six finite completed attempts, each at least.05 nats/token better than the train-count add-one unigram, geometric mean RMS/LN validation PPL ratio<=.98 and every paired ratio<=1.00. The2% margin is a research decision, not a significance claim. Freeze this decision before scoring the official test split. Both official splits were used previously; test is descriptive reused benchmark evidence, not fresh confirmation. No automatic recall-recipe replacement.

Stop after the six attempts, one independent audit, one educational guide and one result report. No bias/epsilon/site/seed sweep or budget extension. Failure is a completed finding.

## Verification and evidence

Reuse the previously audited normalization mathematics, checking archived report/source hashes. Before training, all six actual-vocabulary models must pass independent CPU-float64 manual normalization/rotation and native SDPA outputs/gradients (absolute<=1e-10). CUDA output absolute<=1e-4, relative L2<=1e-5; gradients absolute<=1e-5, relative<=1e-4. Causal prefix error<=1e-6 with a changed final-token effect>1e-6. Deliberately applying centering to RMS and omitting centering from LN must change logits>1e-4. The gain-only LN arm must match the original forward with only its norm biases removed.

Retain every final validation/test token NLL, prediction and first-two-block/tail logit fixtures. Independently recompute all losses from saved FP32 weights using CPU float64: token error<=1e-4, mean<=1e-5, fixture logits<=1e-4, PPL relative<=2e-5. Count argmax differences, do not demand bitwise equivalence. Replay initial states, sample streams and optimizer counters; independently recompute coverage, unigram and the validation decision.

The orchestration follows the frozen SwiGLU runner in a separate source because editing executed sources invalidates their evidence. Reuse existing data loading, original ReLU forward/evaluation and template model. The auditor independently expands normalization and uses SDPA rather than producer softmax. No dependencies are changed.

## Sources and limits

[Zhang/Sennrich2019](https://arxiv.org/html/1910.07467v1) motivates removal of centering; its benchmark quality/runtime does not predict this task. [PyTorch2.11 RMSNorm](https://docs.pytorch.org/docs/2.11/generated/torch.nn.RMSNorm.html) defines the operation; explicitly matching epsilon avoids differing defaults. [Jiang et al.2023](https://arxiv.org/abs/2305.14858) studies equivalence through transformed pre-norm networks, not unconditional equality from swapping modules with unchanged weights. This comparison is not that construction. All five sites change together; a quality difference does not isolate a responsible site. No convergence, whole-model speed, general superiority or novelty claim.
