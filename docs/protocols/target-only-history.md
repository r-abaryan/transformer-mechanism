# Target-only histories: frozen diagnostic

Question: can the unchanged reader and classifier decode a queried key after removing every competing-key write?

The amplification branch stops here. This check concerns readout sufficiency, not another optimizer or scale intervention. No training or tuning is authorized by this protocol.

| Item | Frozen choice |
| --- | --- |
| Models | Exhaustive-feedback delta reader, seeds 3461/3463, steps 3000/12000; 615 parameters |
| Data | Existing full train/validation/test pools: 768/192/384 histories per load; no new data |
| Loads | 2/8/16/32 distinct keys in 32 writes; stable value per key; sorted queries cover every present key |
| Intervention | Keep only writes whose key equals the query, in original order; no padding writes |
| Parameters | All embeddings, global beta, classifier weight/bias fixed; CPU float64, one thread |
| Metrics | Original/filtered accuracy and CE, all/unique/repeat; correct-both, wrong-both, rescues, harms; whole-history all-query correctness |
| Counts | 2 seeds × 2 endpoints × 3 splits × 4 loads = 48 cells; 311,808 answers; 688,128 retained events |
| Shortened length | 16/4/2/1 target writes for original loads 2/8/16/32 |
| Verification | Real evaluator and audit on initialized saved model; supplied logits, missing groups, ties; corrupt labels/counts/logits/retained writes; independent closed form and matrix recurrence |
| Stop | Evaluate exactly these 48 cells once, audit and report every group. No added filter, scale sweep, training or replacement seed |

Original metrics/logits come from already audited endpoints. New filtered logits must be produced by actual model forwards. Hash-bound sources, protocols, checkpoints and prior reports remain unchanged. Two seeds are the independent training runs; repeated endpoints and answer counts are not extra seeds.

Each query gets a separate filtered memory. Therefore all-query correctness measures whether every separate diagnostic answer is correct, not whether one shared memory stores a complete map. Empty groups are reported as count zero and metric null. No accuracy threshold is used to select further work.

Filtering uses the query and task knowledge, without reading the answer label. It changes input length and distribution. Recovery demonstrates sufficiency on the clean input; it does not establish a deployable repair, exact historical learning cause or architecture superiority. Gradients are unnecessary for this forward-only intervention.

Source inspected: [Yang et al. (2024), section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2), delta correction and memory read. Their full model uses input-dependent beta; our toy reader uses one learned scalar and tied normalized Q/K. The clean-read closed form is derived locally in the guide. The paper supplies no prediction for these local scores.
