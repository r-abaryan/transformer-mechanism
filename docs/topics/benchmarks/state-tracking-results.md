# State tracking: recent resets help; sustained updates fail

**Closed.** All three fixed runs finish, but the all-seed reliability rule fails. Reset-free and long-tail accuracy stays around50%, including at training lengths. This is not merely a failure to extrapolate to longer inputs.

[Learn the task and math](state-tracking-guide.md) · [Frozen protocol](../../protocols/state-tracking.md).

## Method

Existing MQA baseline, trained from scratch: width16, two blocks, two Q/one KV head, RoPE, ReLU FFN32 and biased LayerNorm. Only task vocabulary/readout change;4,080 parameters. Seeds2003/2011/2027, each3,000 Adam.001 updates, batch64, alternating8/12 operations. Final QUERY cross-entropy; no intermediate state supervision. CPU float64, PyTorch2.11.0+cu128, one thread. The CUDA-enabled package does not mean these runs used the GPU.

Equal training exposure to three groups: no resets; one initial SET followed by a long KEEP/FLIP tail; mixed operations with at least two SETs and a SET in the second half. Complement pairs keep labels exactly balanced and remain in one split. Train/validation/test sequences are disjoint. Test scores were computed only after all attempts finished. All three seeds share the same evaluation examples; paired examples and audited predictions are not independent training replications.

Success required every seed/group to reach99% at8/12 and90% at16/24. No group/length passes that all-seed rule. These are predefined practical thresholds, not statistical significance tests.

## Results

Test accuracy (%). Counts per cell are64/256/512/512 examples at8/12/16/24 operations. All averages below weight seeds equally.

| Operations | Group | Seed2003 | Seed2011 | Seed2027 | Mean | Ignore-FLIP baseline |
|---|---|---:|---:|---:|---:|---:|
|8|Mixed resets|100.00|85.94|100.00|95.31|87.50|
|8|No resets|50.00|42.19|48.44|46.88|65.62|
|8|Long tail|48.44|43.75|56.25|49.48|50.00|
|12|Mixed resets|98.44|86.72|96.88|94.01|66.41|
|12|No resets|49.61|50.78|44.53|48.31|50.78|
|12|Long tail|50.00|50.39|50.39|50.26|47.66|
|16|Mixed resets|97.66|82.23|96.09|91.99|74.22|
|16|No resets|49.80|47.46|49.41|48.89|50.00|
|16|Long tail|50.00|49.61|49.02|49.54|45.31|
|24|Mixed resets|95.31|79.49|91.02|88.61|75.78|
|24|No resets|49.80|50.59|49.41|49.93|49.22|
|24|Long tail|49.80|49.80|47.07|48.89|47.27|

The ignore-FLIP baseline predicts the last SET value, or the initial state if there is no SET. It can exceed50% despite balanced labels because it uses input information. A constant prediction scores exactly50% in every cell. The exact symbolic solver scores100% by construction.

## What this teaches

| Observation | Supported conclusion | Not established |
|---|---|---|
| Mixed-reset scores are much higher | This task distribution is easier for this trained recipe | The learned model actually implements the last-reset algorithm |
| No-reset and long-tail cells remain near50% | Reliable held-out state tracking fails, even at seen lengths | An architectural impossibility, optimization cause or exact internal failure |
| Two seeds reach100% on eight-operation mixed resets; one reaches85.94% | A good mean can hide a weak run | Reliable success across seeds |
| Symbolic rules agree exhaustively | Our finite ground-truth implementations match | The neural model uses those rules |

A SET erases earlier history. Long strings with frequent SETs can therefore have short relevant dependencies. This is why mixed-reset accuracy alone would be misleading. Sequence length and dependency span both increase in the no-reset/long-tail tests; their separate causal effects are not identified.

## Follow-up: did the models fit their training examples?

Authorized after the original study closed. Evaluate all training-pool examples using the same three final checkpoints, without updates. Validation/test values below are the previously audited results, not new independent evaluations. This descriptive diagnostic does not change the original acceptance rule.

Mean accuracy across seeds (%):

| Operations | Group | Training | Validation | Test |
|---|---|---:|---:|---:|
|8|Mixed resets|99.74|96.88|95.31|
|8|No resets|58.07|48.96|46.88|
|8|Long tail|62.76|46.35|49.48|
|12|Mixed resets|96.03|92.97|94.01|
|12|No resets|50.65|48.44|48.31|
|12|Long tail|52.15|50.00|50.26|

**Interpretation:** twelve-operation no-reset/long-tail performance is low even on the training pool. This is not simply successful fitting followed by poor generalization. At eight operations there is partial fitting and a train/test gap, but the training task is still not mastered. These observations do not separate optimization difficulty, capacity, supervision or training-distribution effects.

Per-seed training accuracy, ordered2003/2011/2027: no-reset8 =50.00/57.81/66.41%; long-tail8 =62.50/57.03/68.75%; no-reset12 =50.00/52.34/49.61%; long-tail12 =50.20/52.54/53.71%. Mixed-reset8 =100.00/99.22/100.00%; mixed-reset12 =97.85/93.75/96.48%. Seed2011 also has a mixed-reset generalization gap at8:99.22% train versus85.94% test. There is no single uniform failure pattern.

Verification:18 new training cells,5,760 predictions. Independent forward and last-reset/parity labels agree; maximum logit error6.04e-14, loss error1.11e-16, zero argmax differences. Original report, data, audit and checkpoint hashes remain unchanged. [Diagnostic and per-label scores](../../../results/state_tracking_fit.json), [saved logits](../../../results/state_tracking_fit_scores.pt), [reproducible script](../../../experiments/state_tracking_fit.py). Training pools contain128 examples/group at8 and512 at12. These are three seeds, not5,760 independent runs.

This diagnostic is complete. Any next training comparison needs a separate question and fixed protocol; none was run here.

Subsequent authorized work: [fresh matched supervision comparison](state-supervision-results.md), executed under a separate protocol. Earlier runs and this diagnostic remain unchanged.

## Verification and evidence

| Check | Result |
|---|---|
| Sequential, composed-affine and last-reset/parity rules |43,690 exhaustive cases pass, lengths0 through7 and both initial states |
| Deliberately omit FLIPs |10,922 disagreements; the ground-truth check detects this error |
| Three initial models, independent logits/gradients |Maximum errors3.33e-16; causality checks pass |
| All final validation/test cells |54 evaluations,13,824 independently recomputed predictions |
| Independent manual normalization/RoPE plus SDPA |Maximum logit error7.95e-14; loss error8.71e-16; zero argmax differences |
| Provenance and accounting |Source/data/checkpoint hashes, initial states, sampling streams, optimizer steps, split separation and decision pass |

Training loops total118.52seconds, including periodic validation but excluding preflight, final evaluation and audit. This is elapsed execution information, not a controlled speed benchmark. All losses, gradients and updated parameters were checked finite. No failed training attempt was replaced.

[Raw report](../../../results/state_tracking.json), [audit](../../../results/state_tracking_audit.json), [data](../../../results/state_tracking_data.json), [saved scores](../../../results/state_tracking_scores.pt), [runner](../../../experiments/state_tracking.py), [independent verifier](../../../experiments/audit_state_tracking.py). Per-label scores and validation curves remain in the raw report.

## Papers and stopping decision

[Liu et al.2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/510ad3018bbdc5b6e3b10646e2e35771-Abstract-Conference.html) motivate testing binary memory over long dependencies. Our parity-changing FLIPs and final-state task differ from their flip-flop language benchmark. [Li, Guo and Andreas2025](https://proceedings.mlr.press/v267/li25r.html) motivate distinguishing state-tracking behavior from its learned mechanism. Both were abstract-screened here; we do not reproduce their experiments or claim the same mechanism.

Keep this as a documented capability boundary for one small recipe and fixed budget. No extra seeds, longer training, architecture sweep or activation patching follows automatically. No novel solution or general transformer limitation is established.
