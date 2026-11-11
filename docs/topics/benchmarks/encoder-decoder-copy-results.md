# Encoder-decoder copy: fitting succeeds, transfer is limited

Executed2026-10-02. Closed at6,000 updates per seed. Every training sequence is copied exactly, including EOS. Held-out short sequences have errors; all six/eight-symbol outputs fail exact copying. The fixed every-seed/group training-plus-validation gate fails.

[Short learning guide](encoder-decoder-copy-guide.md) ? [Frozen protocol](../../protocols/encoder-decoder-copy.md)

## Methods

The previously audited1,888-parameter encoder-decoder is trained from scratch with seeds2371/2377/2381. One width8 post-norm encoder/decoder layer, two heads, ReLU16, sinusoidal positions, dropout0. CPU float64, one thread, native Windows, PyTorch2.11.0+cu128. Adam0.001, batch64,6,000 updates, no clipping/schedule. GPU memory is not used or measured.

Input symbols3..10 are copied in order, followed by EOS. All16 output classes remain available. Each run sees384,000 sampled sequences and1,728,000 target tokens, alternating lengths3/4 and balancing distinct/repeated symbols. Unique training examples total384. All896 source sequences are disjoint by exact input across pools; symbols and smaller patterns are shared. One split is used for all seeds.

Protocol wording: "No answer tokens enter the encoder" is imprecise for copying. The source intentionally contains the content to copy; no additional target sequence or decoder prefix is appended. The implemented task and labels follow the protocol's explicit copy definition.

Full train/validation evaluations occur at1,000/3,000/6,000 updates. All final attempts and a pretest snapshot precede model test scoring. Greedy generation uses only BOS, the source and its own predictions; each example stops at first EOS or source length+3 new tokens. EOS is never forced.

## Mean findings

Equal-weight means over three seeds. Exact accuracy includes every content symbol and EOS. Token/content accuracy and loss use teacher-forced prefixes. Rounded half-up; decisions use unrounded counts.

| Pool |Generated exact |Teacher token |Teacher content |Token loss |
|---|---:|---:|---:|---:|
|train_3 |100.00% |100.00% |100.00% |0.000210 |
|train_4 |100.00% |100.00% |100.00% |0.000439 |
|validation_3 |97.40% |99.35% |99.31% |0.042098 |
|validation_4 |93.23% |98.65% |98.70% |0.044537 |
|test_3 |96.88% |99.22% |99.13% |0.027437 |
|test_4 |90.63% |98.13% |98.05% |0.061242 |
|test_6 |0.00% |59.78% |63.06% |5.189719 |
|test_8 |0.00% |35.04% |38.54% |7.711027 |

Training fitting succeeds on both lengths for every seed and group. The lower held-out scores establish a generalization gap on this split. They do not prove memorization or identify its cause. Short held-out reliability also fails, so longer-input failure is not the only limitation.

## Per-seed and per-group results

Each group cell is **generated exact % / teacher-forced token % / token loss**. Full content scores, EOS rates and counts for every group remain in the raw report.

| Seed | Pool |Distinct |Repeated |
|---|---|---:|---:|
|2371 |train_3 |100.00 / 100.00 / 0.000199 |100.00 / 100.00 / 0.000166 |
|2371 |train_4 |100.00 / 100.00 / 0.000327 |100.00 / 100.00 / 0.000329 |
|2371 |validation_3 |100.00 / 100.00 / 0.000428 |90.63 / 97.66 / 0.149801 |
|2371 |validation_4 |90.63 / 98.13 / 0.037954 |90.63 / 98.13 / 0.101744 |
|2371 |test_3 |100.00 / 100.00 / 0.004843 |93.75 / 98.44 / 0.021864 |
|2371 |test_4 |90.63 / 98.13 / 0.037190 |84.38 / 96.88 / 0.051296 |
|2371 |test_6 |0.00 / 57.59 / 5.202546 |0.00 / 59.38 / 5.180514 |
|2371 |test_8 |0.00 / 34.90 / 7.806509 |0.00 / 39.76 / 7.006584 |
|2377 |train_3 |100.00 / 100.00 / 0.000190 |100.00 / 100.00 / 0.000211 |
|2377 |train_4 |100.00 / 100.00 / 0.000277 |100.00 / 100.00 / 0.000273 |
|2377 |validation_3 |100.00 / 100.00 / 0.005171 |96.88 / 99.22 / 0.011900 |
|2377 |validation_4 |90.63 / 98.13 / 0.039168 |93.75 / 98.75 / 0.034602 |
|2377 |test_3 |100.00 / 100.00 / 0.000580 |96.88 / 99.22 / 0.036220 |
|2377 |test_4 |100.00 / 100.00 / 0.002098 |87.50 / 97.50 / 0.058552 |
|2377 |test_6 |0.00 / 56.70 / 5.382158 |0.00 / 63.17 / 4.836107 |
|2377 |test_8 |0.00 / 28.13 / 8.081977 |0.00 / 31.08 / 8.068691 |
|2381 |train_3 |100.00 / 100.00 / 0.000253 |100.00 / 100.00 / 0.000242 |
|2381 |train_4 |100.00 / 100.00 / 0.000499 |100.00 / 100.00 / 0.000927 |
|2381 |validation_3 |100.00 / 100.00 / 0.000527 |96.88 / 99.22 / 0.084761 |
|2381 |validation_4 |100.00 / 100.00 / 0.002322 |93.75 / 98.75 / 0.051431 |
|2381 |test_3 |93.75 / 98.44 / 0.094510 |96.88 / 99.22 / 0.006607 |
|2381 |test_4 |96.88 / 99.38 / 0.057946 |84.38 / 96.88 / 0.160371 |
|2381 |test_6 |0.00 / 59.60 / 5.683701 |0.00 / 62.28 / 4.853291 |
|2381 |test_8 |0.00 / 37.15 / 7.909721 |0.00 / 39.24 / 7.392684 |

## EOS behavior

Early/late compares the first EOS position with the expected source length+1. Missing means no EOS before the fixed limit. On-time EOS can still follow incorrect content.

| Test length |Early |On time |Late |Missing |
|---|---:|---:|---:|---:|
|3 |2.08% |97.40% |0.52% |0.00% |
|4 |4.17% |94.27% |1.04% |0.52% |
|6 |98.96% |0.00% |0.26% |0.78% |
|8 |99.48% |0.00% |0.00% |0.52% |

Early EOS is the predominant observed long-sequence failure mode. This does not establish why the model stops early or that changing EOS alone would restore correct content. Teacher-forced token accuracy also falls at longer lengths.

## Learning curve and frozen decision

Four-symbol validation exact accuracy at the scheduled full-pool checkpoints:

| Seed |1,000 updates |3,000 |6,000 |Final validation4 loss change |
|---|---:|---:|---:|---:|
|2371 |20.31% |84.38% |90.63% |-0.035713 |
|2377 |23.44% |87.50% |92.19% |-0.035705 |
|2381 |35.94% |89.06% |96.88% |-0.061922 |

All six validation-length loss comparisons improve from3,000 to6,000. This does not establish convergence. Final checkpoints were fixed by budget, not chosen by validation score.

| Criterion |Result |
|---|---|
| Training fit on both lengths/groups |100% for every seed |
| Frozen combined training/validation ?95% in every group |Fail: validation groups range90.63% to100% |
| Short held-out reliability, conditional on combined gate |Not qualified; test4 mean90.63% |
| Length6 / length8 transfer, conditional on combined gate |Not qualified; observed exact accuracy0% at both lengths |

The raw decision field `short_fit=false` denotes the combined training/validation criterion. It must not be read as failure to fit the training pool. Every seed has a validation group below95%, including seed2381 whose overall validation4 score is96.88%.

## Simple baselines

Baselines return the stated content followed by EOS. The copy oracle checks task/label consistency. The other functions are fixed shortcuts; similar scores would not prove a model learned one.

| Test length |Copy oracle |Repeat first symbol |Sort input |Reverse input |
|---|---:|---:|---:|---:|
|3 |100.00% |0.00% |23.44% |20.31% |
|4 |100.00% |0.00% |7.81% |1.56% |
|6 |100.00% |0.00% |0.00% |0.00% |
|8 |100.00% |0.00% |0.00% |0.00% |

## Independent verification and scope

All24 full pools, 2,688 generated sequences and14,976 teacher-forced targets pass independent equations and metrics. A separate per-example decoding loop reproduces all12,686 generated tokens with zero differences. Maximum logit error1.15e-13; metric error3.55e-15, below1e-9.

Input/split/label checks, scripted immediate-EOS, prefix-dependent continuation and absent-EOS cases, continued-after-EOS error injection, stream hashes, optimizer steps and the independent decision pass. All227 recorded prior JSON artifacts and171 frozen source/protocol files retain their hashes. The prior full-model gradient audit is reused; this final audit independently checks trained forward arithmetic and generation.

Elapsed training plus scheduled evaluations and checkpoint saving is 30.69 to 32.78 seconds per seed. This excludes preflight and final test scoring; it is not a controlled throughput comparison.

This is a synthetic copy baseline with exact-input separation and familiar symbols. It establishes finite training success and held-out/length limitations for one recipe. It does not compare architectures, reproduce translation results, identify a copying mechanism, prove impossibility or invalidate the earlier graph studies. Autoregressive/teacher-forcing conventions follow the [original Transformer](https://arxiv.org/html/1706.03762v7), as explained in the guide.

The study is closed. Preserve the failed reliability criteria and early-EOS measurements. No longer training, replacement seed, decoding repair or follow-up experiment is selected.

[Raw report](../../../results/encoder_decoder_copy.json) ? [Data](../../../results/encoder_decoder_copy_data.json) ? [Audit](../../../results/encoder_decoder_copy_audit.json) ? [Pretest snapshot](../../../results/encoder_decoder_copy_pretest.json) ? [Training code](../../../experiments/encoder_decoder_copy.py) ? [Independent audit code](../../../experiments/audit_encoder_decoder_copy.py)
