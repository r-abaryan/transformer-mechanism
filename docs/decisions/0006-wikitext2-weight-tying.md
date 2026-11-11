# Decision0006: shared input/output word embeddings

Date: 2026-10-01. Fixed before new quality evaluation.

## Question and initialization

Can tying input/output word matrices save parameters while keeping final validation perplexity within a predefined tolerance?

Use Decision0003's pinned word-level WikiText-2 encoding, target shift and nonoverlapping64-token evaluation windows. Keep the original MQA recipe: width16, two pre-norm blocks, two query/one KV heads, RoPE offset4, biased ReLU/FFN32, LayerNorm gain/bias and epsilon1e-5, bias-free readout. No normalization or FFN alternative is adopted.

Three new paired seeds1931/1949/1973. Instantiate the original external template. In both arms copy its initial output matrix into the embedding. These entries originate from the template's output `Linear` initialization (uniform +/-1/sqrt(16)). The untied control keeps equal-valued matrices as two separate parameters; the tied control sets `output.weight=embedding.weight`. All initial tensor values and initial predictions match within a pair. Both controls change the original embedding initialization (previously Embedding's distribution). This is an explicit shared-start design, not comparison against historical scores or proof for all initializations. The output matrix in the untied arm is not recopied after training starts.

Untied parameters `32V+3936=1,068,832`; tied `16V+3936=536,384`, at V33,278. Saving532,448 parameters (49.82%). Shared Adam receives one accumulated gradient and one update, not two independent updates. Matrix projection arithmetic remains: parameter savings do not imply half compute, training memory or runtime.

## Budget and decision

Six attempts,6,000 default Adam.001 updates each, batch8/context64,3,072,000 targets/run. Same CPU sampling generator seed+1000 within each pair; alternate arm order by seed. Native Windows CUDA FP32, TF32 off, deterministic algorithms, one CPU thread. No dropout, warmup, clipping, weight decay, output scale, extra projection, tuning or checkpoint selection. Verify finite loss, every gradient and updated parameters at every update. Archive checkpoints, Adam/RNG state, streams and source/data hashes. Retain failed attempts.

Primary metric: final validation mean next-word NLL. Accept the parameter/quality tradeoff only if all six finite attempts complete, all models beat the add-one training-count unigram validation NLL by at least.05 nats, parameter saving is at least45%, geometric mean tied/untied validation PPL ratio<=1.05, and every seed ratio<=1.10. These are research tolerances, not statistical significance or equality claims. Report any losses even if acceptance passes.

Freeze the validation decision before official test scoring. Validation and test have been used previously; test is descriptive reused benchmark evidence, not fresh confirmation. Test cannot revise acceptance. No automatic change to the recall recipe. Stop after six attempts, one all-token audit, educational guide and result report regardless of outcome. No additional seeds, initialization alternatives or training extension.

## Verification

First check a small nonlinear vocabulary model. For tied E, `dL/dE=G_input+G_output` evaluated at equal-valued untied matrices. Verify analytical gradients against double autograd, central finite differences and independent Decimal forward/loss arithmetic. Include repeated input IDs to test scatter accumulation. Omitting either gradient role must fail. Equal initial outputs do not imply equal subsequent training trajectories.

Before training all six actual-vocabulary controls must pass independent CPU-double manual normalization/RoPE plus native SDPA outputs/gradients (absolute<=1e-10). CUDA logits absolute<=1e-4 and relative L2<=1e-5; gradients absolute<=1e-5 and relative<=1e-4. Causal prefix<=1e-6, changed final token>1e-6. Check initial output identity and the full-transformer tied gradient equals the sum of untied input/output gradients. Check true alias identity after CPU-double/CUDA conversion and checkpoint reload, one optimizer registration, and distinguish an equal-valued clone that ceases to share updates. Independent gradients are compared by unique parameter name, avoiding duplicate state-dict entries.

Keep every final validation/test NLL and prediction, first two full-block logit fixtures and any tail. Independent CPU-double audit recomputes all token losses from saved FP32 weights, token error<=1e-4, mean<=1e-5, fixtures<=1e-4, PPL relative<=2e-5. Count argmax differences descriptively. Independently replay initial states, starts, RNG/Adam counters, data coverage, unigram and the validation-only decision. Check tied checkpoint vocabulary entries are equal and sharing survives load. Failed implementation checks are archived, not interpreted as quality findings.

Reuse the immutable original data loader, ReLU forward/evaluator and model template. The separate orchestration preserves executed-source provenance. The independent auditor uses separately expanded arithmetic rather than producer evaluation.

## Sources and boundary

[Press/Wolf2017](https://aclanthology.org/E17-2025/), gradient-role analysis and weight tying, and [Inan et al.2017](https://arxiv.org/html/1611.01462v3), reuse framework and recurrent WikiText-2 experiments, motivate the constraint. Their larger models, objectives, regularization and training differ. No paper replication, novel architecture, general quality superiority, convergence or speed claim is intended.
