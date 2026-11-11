# Classifier-only terminal feedback does not preserve stopping

2026-10-08. **Keeping terminal targets at the classifier while blocking their new gradients into earlier layers fails to repair complete copying.** The weak model reaches 128/128 correct fifth-token answers but zero complete long copies. Primary repair and preservation criteria both fail.

[Protocol](../../protocols/copy-terminal-gradient-route.md) · [Report and audit](../../../results/copy_terminal_gradient_route/report.json) · [All groups and losses](../../../results/copy_terminal_gradient_route/metrics.csv)

## What differs from deleting EOS loss

All targets remain, with the original loss denominator. During training only, detach the final hidden state for the terminal prediction before applying the classifier. Content hidden states remain connected. At fixed weights, logits and scalar CE are unchanged.

For terminal hidden state h, the classifier receives `W * stop_gradient(h)`. Terminal CE updates all classifier rows through softmax, including competing classes. It contributes no new gradient to attention, embeddings, normalization or FFNs. Content CE still trains everything. Earlier EOS-trained parameters and Adam moments are retained, so this does not erase all historical terminal influence.

Each of four update-500 checkpoints has a fresh native and classifier-only continuation for 500 updates. Identical restored states, batches and position streams make these paired training comparisons. All native endpoints exactly reproduce the original update-1,000 weights, optimizer and generators. Inference uses the unchanged model without detachment hooks.

## Complete copying and stopping

Long results are out of 128 length-10 test examples. Content exactness scores the first ten generated symbols; complete copying also requires the correct EOS. No EOS blocking or supplied content is used in generation.

| Model | Native full copies | Classifier-only full copies | Classifier-only content exact | On-time EOS | Missing EOS |
|---|---:|---:|---:|---:|---:|
| Cross-only 3767 | 1 | 0 | 125 | 0 | 128 |
| Cross-only 3779 | 128 | 0 | 128 | 0 | 128 |
| Cross-only 3793 | 124 | 2 | 126 | 2 | 125 |
| Combined 3767 | 126 | 0 | 120 | 0 | 127 |

The last two intervention models each stop early on one additional example. None stops late. Missing EOS means no EOS within the fixed n+3 generation limit. Combined content loses six exact sequences in aggregate, despite perfect fifth-token scores.

## The failure already affects training inputs

| Model | Native train6 full | Classifier-only train6 full | Native test6 full | Classifier-only test6 full |
|---|---:|---:|---:|---:|
| Cross-only 3767 | 256/256 | 0/256 | 125/128 | 0/128 |
| Cross-only 3779 | 256/256 | 0/256 | 128/128 | 0/128 |
| Cross-only 3793 | 256/256 | 8/256 | 128/128 | 7/128 |
| Combined 3767 | 256/256 | 0/256 | 128/128 | 0/128 |

Train6 common evaluation CE rises from 0.032/0.018/0.020/0.021 to 0.671/0.639/0.577/0.754 in table order. The intervention no longer fits complete short copying under this budget. It cannot be interpreted purely as an extrapolation failure.

The frozen weak test10 >=95% full-copy and >=95% on-time-stopping rule fails. Preservation also fails: 101 of 159 declared model/pool/group checks lose more than five percentage points. These checks overlap and are not independent experiments. Every training, validation, test, layout and uniform group remains visible in the report/CSV.

## Why the supporting diagnostics are insufficient

All four intervention models answer the supplied-prefix fifth-token test correctly on 128/128 examples. The weak head-0 short/long marker-mass gap shrinks from 0.805 to 0.180. Neither improvement predicts complete copying or correct stopping here.

This rules out the narrow expectation that classifier-only new terminal feedback would preserve stopping under this continuation. Retaining terminal targets and their classifier gradients was insufficient. It does not prove a universal need for upstream terminal gradients, a capacity limit or convergence. Changes to the rest of the model still arise from content gradients and stored optimizer history; why this restricted procedure fails to fit stopping remains unresolved.

The previous loss-deletion result is a historical reference, not an extra paired arm in this comparison. Both interventions damage stopping, but they differ in objective and gradient flow. No causal share is assigned to attention, classifier or optimizer from their similarity.

## Verification, budget and source

Eight runs, 4,000 updates, 638,000 sequences, 3.36M computed and active target rows. CPU float64, one thread, Adam .001; 840 targets/update, batches210/168/140/120, no accumulation. Original complete-step profiling is reused, with actual initialized intervention steps additionally timed and optimizer storage recorded. CPU peak memory remains unmeasured; GPU memory is inapplicable.

Preflight verifies unchanged logits, native classifier gradients and content-only gradients for every other parameter, with the original denominator. Deliberately removing detachment fails the scope check. Saved continuation/evaluation/reference paths and prior label/denominator/missing-group checks pass. Native checkpoints and both sampling streams reproduce exactly. All generated outputs and ordinary/paired metrics pass reference audits, maximum discrepancy 2.443e-13; 95 protected files remain unchanged. Shared forward/optimizer helpers limit independence. No failed empirical threshold was revised.

[Vaswani et al. (2017), sections 3.1/3.4](https://arxiv.org/html/1706.03762v7#S3), previously inspected, describes the decoder and final classifier. Gradient detachment and the bounded continuation are local diagnostics, not a method proposed or predicted by that source. Stop this intervention; no learning-rate, EOS-weight or budget sweep is queued.
