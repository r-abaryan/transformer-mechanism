# Repeated-key recall: frozen comparison protocol

## Question and task

Does RoPE's advantage survive a change from unique-key lookup to latest-write lookup? Train new models from scratch. This tests transfer of the architectural choice, not zero-shot transfer of the previous weights.

Input is n key/value pairs followed by marker18 and query key q. Keys are 0..11; values 12..17. Exactly one key appears twice, all other keys once. Query only the repeated key. All n values differ, so its two answers conflict. The correct answer is `v_j`, where `j=max{i:k_i=q}`. Example: `A:red B:blue A:green -> query A -> green`.

## Data fixed before generation

Generator seed20270104. For n=3 and n=4, create 800 training, 100 validation and 200 test families. For n=6, create 200 test families only. A family is an unordered multiset of key/value pairs modulo cyclic renaming of the six value labels. Keep families disjoint across splits. This prevents moving the same bindings to another order or cyclic value naming across splits. Earlier unique-key datasets cannot contain these repeated-key multisets. Other relabelings and smaller fragments can remain familiar.

Each family supplies three sampled orders and their reversals, all distinct, with all six cyclic value renamings: 36 examples. Reversal swaps which repeated value is correct. This gives 28,800 train / 3,600 validation / 7,200 test examples per training length; six-pair test has 7,200. Label counts are exactly balanced; position counts need not be. Save inputs and family identities. Independent audit reconstructs labels and split identity, checks reversal pairs and rejects the wrong first-match rule on every example.

## Matched training

Reuse the verified positional forward implementation, model and optimizer recipe. Two pre-norm blocks, width16, two width8 heads, FFN32/ReLU, vocabulary19, dropout0, final LayerNorm: 5,088 learned parameters. Sinusoidal addition versus standard adjacent-pair RoPE on projected Q/K only, base10000; position offset+4. CPU float64, one thread, deterministic algorithms. No GPU speed claim.

Seeds1013/1031/1051/1069/1091, paired learned initialization and minibatches. Adam lr.001, betas.9/.999, eps1e-8, no decay, clipping or warmup. Batch64, exactly6,000 updates. Alternate three pairs on odd and four pairs on even steps. Advance both streams every update with seeds model_seed+1000/+2000. Equal 3,456,000 tokens per run; rotations add arithmetic. Save stream hashes, validation curves at0/1000/.../6000, latest resumable model/Adam/RNG checkpoint and elapsed loop time. Freeze source and protocol hashes. Retain failed attempts; no replacements. Final checkpoint only for test evaluation, after all ten attempts.

## Metrics, decision and stop

Report each seed's accuracy and cross-entropy at3/4/6 pairs; six-pair accuracy by latest occurrence position; both-order reversal correctness; all36-example family correctness; rate of choosing the obsolete first value. Also report the trivial always-last-pair heuristic. These are not independent observations within a family.

Keep the claim that RoPE's advantage transfers only if all five RoPE seeds reach >=90% six-pair accuracy, mean paired gain >=5 percentage points, no seed loses six-pair accuracy, every RoPE seed reaches >=99% at both three and four pairs, and mean loss relative to baseline is <=.5 points at each short length. Engineering thresholds, not significance tests. Failure does not overturn the previous unique-key result.

Before interpretation, independently reload all checkpoints using library attention / explicit rotation matrices and SDPA; require logits and metrics within1e-10, unchanged hashes, split isolation and paired stream/weight verification. Reuse prior mathematical verification only if its source hash still matches. Stop after ten attempts and audit. No tuning, extra seeds or alternate designs on this opened test set.

## Source and explanation

[RoFormer section3](https://arxiv.org/html/2104.09864v5) defines the rotation identity `R(i)^T R(j)=R(j-i)`. Relative position alone does not implement the latest-write rule: the model must combine matching-key content with ordering. This synthetic repeated-key task is our controlled test, not a reproduction of the paper's benchmarks. See the [positional guide](positional-comparison-guide.md) for the diagram and math.
