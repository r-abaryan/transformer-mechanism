# Depth versus width protocol

Fixed before training. Question: does an additional sequential attention block improve associative learning beyond the original model and a wider one-block parameter control?

Compare original one-block d=16, h=2, FFN=32; two-block d=16, h=2, FFN=32; and one-block d=24, h=2, FFN=40. Vocabulary 13 and all other model choices unchanged. Actual parameters: 2,672 / 4,896 / 5,152. The wide model has approximately 5.23% more parameters than the deep model; this is an approximate parameter control, not equal FLOPs, head width, memory or wall time.

Implementation correction: the initial protocol incorrectly estimated the specified wide model as 5,024 parameters. Direct counting found 5,152 after deep training began and before wide training. Keep every dimension, criterion and training choice unchanged; correct only the count/assertion. Preserve the original protocol/source in `results/depth_run_provenance/` and resume using verified completed deep checkpoints, without retraining them.

Reuse the completed Adam baseline (0.001, betas 0.9/0.999, epsilon 1e-8, no decay) rather than rerun it. Train the two new configurations for 3,000 updates, batch 64, seeds 11/29/47/73/101, original data and minibatches (seed+1000). CPU float64; no scheduler, clipping, tuning or checkpoint selection. Configurations have different initializations due to shape/depth; identical seed is not identical weights across architectures.

First verify one-block new implementation equals the old implementation with copied parameters in both outputs and gradients; verify two-block causality. Primary criterion: every deep-model seed achieves at least 90% train and validation accuracy. Depth-over-width criterion: deep validation accuracy beats paired wide accuracy by at least 10 percentage points on average. Report all outcomes, not just a winner.

Use original validation diagnostically, since already inspected. Do not claim new held-out confirmation. No original test evaluation in this experiment. A successful depth result would justify a separately designed fresh confirmation, not a universal claim about transformers. Record counts, curves, hashes, checkpoints, minibatch correspondence and descriptive times. Add failures to the register; no automatic tuning follows.
