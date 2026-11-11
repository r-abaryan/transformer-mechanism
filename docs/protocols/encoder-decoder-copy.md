# Encoder-decoder copy: fixed learning study

Freeze before evaluation. Question: does the verified small encoder-decoder learn exact copying, including EOS, and transfer to longer sequences? One recipe, no architecture comparison. Stop after seeds2371/2377/2381 each complete6,000 Adam updates, or record a nonfinite failure. No tuning, replacement seeds or extension.

Reuse the1,888-parameter post-norm model from `experiments/encoder_decoder_contract.py` unchanged: width8, two heads, one encoder/decoder layer, FFN16, separate embeddings, sinusoidal positions, dropout0. CPU float64 preserves the audited arithmetic path; GPU memory is not used or measured. Adam lr0.001, betas0.9/0.999, epsilon1e-8, no weight decay, clipping or scheduling. Batch64. Native Windows, one CPU thread, deterministic algorithms.

## Task, splits and exposure

Content symbols3..10; PAD0/BOS1/EOS2. All16 model output classes remain available, including unused symbols. Input is a sequence of content symbols. Correct output is the identical sequence followed by EOS. Teacher-forcing input is BOS plus that sequence. No answer tokens enter the encoder.

Generator seed2026100203. Exact input sequences are disjoint across pools. Symbols and shorter subsequences are shared; this does not test unseen vocabulary or a stronger structural-family split. Each pool is equally divided between all-distinct symbols and at least one repeated symbol. Sample without replacement within each pool, with at most100,000 proposals per group; do not change the generator after seeing scores.

| Pool | Examples | Distinct/repeated |
|---|---:|---:|
| Train3 / train4 |128 /256 |64+64 /128+128 |
| Validation3 / validation4 |64 /64 |32+32 each |
| Test3 / test4 / test6 / test8 |64 /64 /128 /128 |Half each |

Alternate length3/4 updates. Sample32 examples from each group with replacement using CPU generator `seed+1000`. Each run sees384,000 sequences and1,728,000 supervised tokens, including EOS. Loss is mean cross-entropy over every target position. These length-homogeneous batches need no padding; previous padding evidence remains separate.

## Evaluation and generation

Record full training/validation metrics at1,000/3,000/6,000 updates, minibatch loss and elapsed time. Keep the final checkpoint, optimizer state and stream hash. Finish all three attempts and save a pretest snapshot before model test scoring. Evaluate every full pool at the end; retain teacher-forced logits and generated token lists.

Greedy generation starts from BOS and receives only source IDs and its own generated prefix. Stop each example at its first EOS. Maximum new tokens = source length+3; do not force EOS or suppress PAD/BOS/unused vocabulary outputs. Report exact sequence accuracy including EOS, teacher-forced token accuracy/loss, content-only teacher-forced accuracy, and early/on-time/late/missing EOS rates. Early/late compare first EOS position with source length+1. A correctly timed EOS alone does not establish correct content.

Report per seed, length and distinct/repeated group. Baselines: direct-copy oracle, repeat first input symbol, sorted input and reversed input, each followed by EOS. Check the oracle independently through token-list equality. Corrupted reversal labels must fail on at least one fixture; palindromes are not negative examples.

## Verification and decision

Verify splits, labels and input shifting independently. Exercise the generation controller with scripted immediate, later and absent EOS, and a prefix-dependent continuation. Detect a controller that continues after EOS. Reuse the prior model numerical audit; independently reconstruct every final teacher-forced logit/loss and every free-running sequence from saved weights. Maximum logit/loss error≤1e-9, zero generated-token differences. Recompute metrics, thresholds and stream/optimizer hashes independently.

Short fitting passes only if **every seed and each distinct/repeated group** reaches≥95% exact generated-sequence accuracy on both training and validation lengths. Short held-out reliability additionally requires≥95% in every test3/4 group. Length6 and length8 transfer are evaluated separately, conditional on short fitting, with≥90% in every seed/group. Retain descriptive long-length results even if fitting fails. No convergence or learned-algorithm conclusion follows from accuracy.

This tests a synthetic copy recipe. It does not reproduce translation, compare architectures, establish real-world generation quality or measure GPU throughput. Sources and teaching explanation will distinguish teacher forcing from model-generated prefixes.
