# Which activation replacements rescue the shifted model?

**Replacing the second block's attention output nearly restored compact accuracy.** This locates a sufficient rescue point in these tests. It does not identify where the original error begins.

## Short explanation

We ran each crop twice: normally and with every position shifted by four. We saved intermediate values from the normal run. Then we inserted selected saved values into the shifted run. The input tokens and weights stayed fixed. No training occurred.

Attention brings information between tokens. The FFN transforms each token's representation. The residual path carries the representation forward by addition. Replacing one branch lets us test whether its normal value can rescue predictions while other computations continue.

## Method and math

The [protocol](../../activation-protocol.md) fixed all replacements before execution. Use all 5,120 six-pair-derived crops for each of five frozen models. Each crop contains three pairs and eight tokens. Shift +4 and these examples were chosen after prior diagnostics; this is exploratory.

For each block: a=Attention(LN1(x)), u=x+a, f=FFN(LN2(u)), y=u+f. Donors are x,a,f from the compact run of the same crop. Replace one branch in one block, or the same branch in both blocks. Replace every token's branch value. For a residual replacement, compute attention from the live x but replace x only in u. Downstream FFNs are recomputed unless explicitly replaced.

## Results

Means across five seeds. The criterion requires every seed to reach 90%, not just the mean.

| Compact donor replacement | Accuracy | Cross-entropy, nats | Every seed >=90% |
|---|---:|---:|---|
| None: shifted baseline | 60.60% | 2.2016 | No |
| Block 1 residual operand | 60.97% | 2.2523 | No |
| Block 1 attention output | 92.12% | 0.3694 | No |
| Block 1 FFN output | 65.01% | 1.9182 | No |
| Block 2 residual operand | 58.71% | 2.3533 | No |
| Block 2 attention output | 99.984% | 0.0015 | Yes |
| Block 2 FFN output | 90.76% | 0.2863 | No |
| Both residual operands | 60.98% | 2.2461 | No |
| Both attention outputs | 99.992% | 0.0013 | Yes |
| Both FFN outputs | 91.87% | 0.2543 | No |
| Unmodified compact reference | 99.996% | 0.0005 | Yes |

| Seed | Shifted | Block 1 attention donor | Block 2 attention donor |
|---|---:|---:|---:|
| 137 | 65.74% | 90.96% | 100% |
| 163 | 58.61% | 98.09% | 100% |
| 191 | 61.07% | 90.62% | 99.94% |
| 223 | 54.98% | 86.27% | 100% |
| 257 | 62.60% | 94.67% | 99.98% |

Block 2 attention replacement rescued all 10,086 original errors across model/crop evaluations, but harmed four previously correct predictions. These counts share episodes and models; they are not independent trials. Both attention replacements harmed two. Neither result is perfect.

## What we can conclude

The compact second-attention output is sufficient for near-complete rescue despite retaining a shifted residual stream. That is stronger evidence than merely observing attention weights. Yet the donor contains information computed through the compact first block. The fault may originate upstream. This replacement bypasses the live computation; it does not prove that block 2 alone caused the failures.

FFN donors also rescue many cases. Residual-only donors do not reliably rescue. Neither finding makes those paths irrelevant. The interventions mix values from different runs and can be unnatural. Effects are not additive causal shares. No necessity claim, novel architecture or paper benchmark result follows. The [existing literature comparison](../training/depth-comparison-results.md#paper-comparison-and-remaining-question) remains applicable.

## Verification and next question

[Script](../../../experiments/activation_paths.py) and [raw report](../../../results/activation_paths.json) preserve per-seed metrics, controls and hashes. Explicit compact reconstruction matched library logits exactly. Shifted reconstruction differed by at most 5.82e-14. Full compact-donor and shifted-donor sham reconstructions had zero error. Saved baseline accuracy/loss reproduced within 1e-12. Labels, finite outputs, checkpoint integrity, source hashes and paired-count arithmetic passed. This uses the same PyTorch modules, not an independent framework.

Follow-up completed: [reverse and unrelated-donor controls](donor-controls-results.md). Same-answer unrelated donors also rescue, and different-answer donors usually transfer their own answer. The matching-example specificity criterion failed. Separate Q/K routing from V content next; do not equate whole-output replacement with fixing recipient retrieval.
