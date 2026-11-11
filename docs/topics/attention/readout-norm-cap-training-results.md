# Classifier norm cap: no fitting repair

Completed and independently audited, 2026-10-05. Preventing further classifier growth **worsened training fitting in both seeds**. It slightly reduced held-out load-32 cross-entropy (CE), with mixed accuracy effects. The declared benefit and validation-preservation rules both fail.

## Method

Two development seeds, 3461/3463, start from exhaustive-feedback step 3000 with all Adam states and streams restored. Control and capped arms each execute 1,000 updates. The cap projects the centered classifier norm back to its starting radius after Adam; it leaves memory parameters and optimizer moments untouched during projection. See the [short mathematical guide](readout-norm-cap-training-guide.md) and [frozen protocol](../../protocols/readout-norm-cap-training.md).

Both controls exactly reproduce the historical step-4000 model, optimizer and RNG states. This isolates the intervention in this window. It does not isolate every subsequent parameter change.

The 615-parameter reader uses 16 by 8 shared memory, tied normalized Q/K, one learned beta, six value embeddings and a six-class affine head. Adam uses LR .003 and no schedule/decay. Each update has 64 histories of 32 events, with exhaustive queries; loads 2/8/16/32 cycle equally. Four runs execute **4,000 updates, 256,000 histories, 8,192,000 events and 3,712,000 answer targets**. CPU float64, one thread; summed run time is 144.85 seconds, including trace collection/saving but excluding setup, evaluation and audit. This is not a speed comparison. Complete restored steps were profiled; GPU memory is unmeasured/null because GPU was unused.

## Whole-pool results

Tables show accuracy (%) / mean CE. Each load has 768/192/384 training/validation/test histories. A complete map requires every queried key in one history to be correct. Pools were opened previously; test scores are exploratory. All histories remain 32 events, so this is not a length-transfer test.

| Seed | Arm | Keys | Train: % / CE | Validation: % / CE | Test: % / CE | Complete train maps / 768 |
|---|---|---:|---:|---:|---:|---:|
| 3461 | control | 2 | 100.0000 / 1.29227e-14 | 100.0000 / 5.14623e-14 | 100.0000 / 1.48611e-14 | 768 |
| 3461 | control | 8 | 100.0000 / 0.000211754 | 100.0000 / 0.000697339 | 99.7721 / 0.00423364 | 768 |
| 3461 | control | 16 | 99.9919 / 0.00804127 | 98.6003 / 0.0602913 | 98.0143 / 0.0703814 | 767 |
| 3461 | control | 32 | 95.5485 / 0.152411 | 87.3372 / 0.387969 | 86.7513 / 0.432982 | 184 |
| 3461 | cap | 2 | 100.0000 / 1.4333e-12 | 100.0000 / 2.61191e-12 | 100.0000 / 1.51951e-12 | 768 |
| 3461 | cap | 8 | 100.0000 / 0.000239644 | 100.0000 / 0.000905787 | 99.8047 / 0.00385481 | 768 |
| 3461 | cap | 16 | 99.9837 / 0.0098721 | 98.6003 / 0.0605554 | 97.9004 / 0.0705896 | 766 |
| 3461 | cap | 32 | 95.3288 / 0.166602 | 87.2396 / 0.38573 | 86.6292 / 0.428352 | 165 |
| 3463 | control | 2 | 100.0000 / 1.26635e-16 | 100.0000 / 8.09538e-18 | 100.0000 / 1.61908e-17 | 768 |
| 3463 | control | 8 | 100.0000 / 0.000224131 | 99.6094 / 0.0162026 | 100.0000 / 0.000269471 | 768 |
| 3463 | control | 16 | 99.8535 / 0.00785036 | 98.5677 / 0.0444512 | 97.4121 / 0.102102 | 750 |
| 3463 | control | 32 | 96.1751 / 0.132926 | 86.8652 / 0.442727 | 85.0505 / 0.499291 | 229 |
| 3463 | cap | 2 | 100.0000 / 2.84758e-14 | 100.0000 / 1.69107e-14 | 100.0000 / 1.81111e-14 | 768 |
| 3463 | cap | 8 | 100.0000 / 0.000286616 | 99.6094 / 0.0147423 | 100.0000 / 0.0004256 | 768 |
| 3463 | cap | 16 | 99.8698 / 0.00944071 | 98.4701 / 0.0481591 | 97.4284 / 0.0991596 | 752 |
| 3463 | cap | 32 | 95.7153 / 0.14667 | 86.9954 / 0.431212 | 85.0830 / 0.485107 | 200 |

Starting train-32 accuracy / CE / complete maps was 94.4499% / 0.192271 / 149 for seed 3461 and 94.9259% / 0.175781 / 181 for seed 3463. Both arms improve from that start, but the cap improves less. Final cap-minus-control training gains are **-0.2197/-0.4598 percentage points**, CE changes **+0.014191/+0.013744**, and map changes **-19/-29**. Neither arm reaches 99% accuracy at every training load.

Contradictions matter: seed 3461 loses one extra train-16 answer under the cap. In seed 3463, both arms increase train-16 errors from one initially to 18/16, despite lower CE. Validation-32 CE improves under the cap in both seeds, but validation-32 accuracy decreases in 3461 and validation-16 accuracy decreases in 3463. Therefore validation preservation fails in both.

[Full group export](../../../results/readout_norm_cap_training/evaluation.csv) includes all 72 cells and all/unique/repeated-key counts, losses, accuracies and map counts. Empty groups have no score. Load 32 contains only unique keys; loads 2/8/16 contain only repeated keys. These groups are not separate causal comparisons because their loads also differ.

## Old errors and parameter changes

Rescues and harms use each seed's fixed step-3000 train-32 mask, not a newly selected final-error set. There were 1,364/1,247 starting errors. Lower conditional error CE is helpful, but does not establish whole-pool repair.

| Seed | Arm | Old errors rescued | Previously correct answers harmed | CE on original errors |
|---|---|---:|---:|---:|
| 3461 | control | 466 | 196 | 1.494219 |
| 3461 | cap | 414 | 198 | 1.547176 |
| 3463 | control | 520 | 213 | 1.444021 |
| 3463 | cap | 443 | 249 | 1.512651 |

| Seed | Stage | Centered head norm | Value norm | Beta | Train-32 centered logit RMS |
|---|---|---:|---:|---:|---:|
| 3461 | baseline | 7.9432 | 16.7227 | 0.322128 | 3.7909 |
| 3461 | control | 9.4413 | 19.2034 | 0.276918 | 4.5710 |
| 3461 | cap | 7.9432 | 19.5422 | 0.296389 | 4.1523 |
| 3463 | baseline | 9.2939 | 16.4414 | 0.297427 | 4.0498 |
| 3463 | control | 11.2077 | 19.0507 | 0.251938 | 4.9435 |
| 3463 | cap | 9.2939 | 19.4288 | 0.271512 | 4.4700 |

The cap activates on all 1,000 updates in each capped arm and holds the head norm at its radius. Value norms grow slightly more than in controls; logit RMS still grows from the start. These are measured joint adaptations, not proof of intentional compensation. The constraint did not hold total logit scale fixed.

## Verification and meaning

Two preflight methods exercised initialized saved training, the real evaluator/audit, an active cap, null/tie/absent-group cases and deliberate label/logit/gradient/moment/parameter/stream/count/decision defects. The first attempt exposed a dict-only comparator failing on seed lists. It was fixed before formal training by reusing the existing list-capable comparator; the failed source, log and [development record](../../../results/readout_norm_cap_training_development/record.json) remain preserved. The passing fixture used eight updates; two profiled steps are separate from the 4,000 formal updates.

Independent audit checks every formal gradient, scalar Adam update and projection, paired streams, final states, all labels/matrix logits, summaries and decisions. It passes **72 cells / 467,712 evaluation answers**, with maximum errors 1.88e-15 (gradient), 4.44e-16 (Adam) and 2.13e-14 (logits); 32 scoped protected files remain unchanged. See [raw report](../../../results/readout_norm_cap_training/report.json), [audit](../../../results/readout_norm_cap_training/audit.json) and [preflight](../../../results/readout_norm_cap_training_preflight.json). Decisions were saved before test evaluation.

**Supported:** this fixed-radius training intervention costs fitting quality in both tested seeds. A confidence-loss benefit on opened held-out pools does not become a reliable learned repair. Earlier frozen scaling and expansion-pressure findings remain valid.

**Unknown:** why joint optimization leaves harmful competing reads, whether earlier intervention or a different constraint would help, and whether value/head scaling contributes elsewhere. Two reused seeds and this window cannot establish a universal result, convergence or architectural impossibility. Stop at the frozen budget; no radius sweep, continuation or fresh confirmation follows this failed gate. RL remains parked.

## Sources inspected

[Guo et al. (2017), section 4.2/equation 9](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) supports positive scaling with unchanged argmax. It studies post-hoc temperature calibration, not this training cap. [PyTorch 2.11 Adam](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html) provides the moment/bias-correction equations checked by the audit. [Yang et al. (2024), section 2.2](https://arxiv.org/html/2406.06484v2) provides delta-write/read context; our scalar-beta toy reader is not full DeltaNet. The projection and gates are local designs. No paper predicts these local scores.
