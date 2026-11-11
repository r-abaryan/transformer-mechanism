# Recurrent state versus Transformer: bounded architecture comparison

Frozen2026-10-01 before training. Question: does a learned recurrent-state baseline improve state-tracking performance and length transfer under controlled training?

## Models and controls

Transformer: reuse the frozen state-tracking model/forward, width16, two pre-norm MQA blocks, two Q/one KV head, RoPE offset4, biased ReLU FFN32, biased LayerNorm,7-token embedding, bias-free two-class output;4,080 parameters.

GRU: Embedding(7,16), one unidirectional biased PyTorch GRU(input16,hidden28), zero initial hidden state for every sequence, bias-free Linear(28,2) on every hidden state. No positional encoding, extra FFN, normalization, dropout or output feedback. Parameters:112 embedding +3*28*16 input weights +3*28*28 recurrent weights +6*28 biases +56 readout =4,032,48 fewer (1.18%). This is comparable parameter count, not equal capacity/compute.

Fresh seed pairs2081/2099/2111. For each pair instantiate the original Transformer under the seed. Reset the RNG to the same seed for GRU construction, then copy only the Transformer's initial embedding values into the GRU. Keep native remaining initializations. Full initial states/predictions differ; only embedding values and sampling streams are matched. Recreate both models when replaying initialization. Alternate architecture order by seed.

Use PyTorch2.11 GRU's reset-after-projection convention:

`r=sigmoid(W_ir*x+b_ir+W_hr*h+b_hr)`; `z=sigmoid(W_iz*x+b_iz+W_hz*h+b_hz)`;
`c=tanh(W_in*x+b_in+r*(W_hn*h+b_hn))`; `h_next=(1-z)*c+z*h`.

These are elementwise products where appropriate. The original Cho reset-before-projection formula is different; this is not an exact implementation of that paper's unit or a replication of its translation results.

## Data, loss and budget

Reuse the hashed state_tracking_data.json and exact KEEP/FLIP/SET rule: train8/12 operations, test8/12/16/24, reset-free/long-tail/mixed groups, complement pairs kept together. Validation/test have already been opened. This is a reused-distribution comparison, not fresh confirmation. No new test-driven selection.

Both arms use exactly `CE(QUERY,final_state) + mean CE(operation_position,state_after_operation)`. No loss on the initial-state token. Targets are never input tokens. GRU processes the initial token, every operation and QUERY sequentially; hidden state resets to zeros between examples/batches. Both models predict the final answer at QUERY.

Six attempts,3,000 Adam.001 updates each, batch64, alternating8/12 operations, sample generator seed+1000. CPU float64, one thread, deterministic algorithms.192,000 final labels,1,920,000 operation-state labels and2,304,000 input tokens per completed run. No clipping, warmup, weight decay, tuning, seed replacement or best-checkpoint selection. Check finite loss/gradients/parameters each step. Preserve failures. Score complete training and validation pools every500 steps. Save final weights, Adam/RNG/stream states, curves and hashes.

Training exposure is matched; wall time and operation counts are not. Forward dense projection/attention MAC estimates per length L (including initial and QUERY): Transformer3616L+64L^2; GRU3752L. Exclude embedding lookup, biases, normalization, rotations, nonlinearities, elementwise gates and backward computation. Report both estimates and elapsed loops; do not infer a controlled speed advantage.

## Decision and stopping

First freeze all final training/validation scores and checkpoint hashes before scoring test. Final metrics include loss, final-answer accuracy by label, and operation-state accuracy for every seed/group/length. Constant final prediction is50%; exact symbolic solution is100%; reuse the measured ignore-FLIP baselines.

Report three separate checks:

1. GRU reliability: all six attempts finite/complete, and every GRU seed/group has at least99% final training and validation accuracy at8/12,99% short test accuracy at8/12, and90% test accuracy at16/24.
2. Relative long-input advantage: every paired seed's equal-cell mean GRU-minus-Transformer accuracy over test16/24 no-reset/long-tail cells is at least10 percentage points.
3. Fit comparability: both architectures, every seed/group, reach99% final training and validation accuracy at8/12. If this fails, do not attribute the comparison purely to extrapolation; fitting and transfer are entangled.

The scoped performance claim passes only if checks1 and2 pass. Check3 is an interpretation gate, reported separately. Even success concerns these model packages and reused tests, not recurrence alone, general superiority, arbitrary lengths or novelty. Thresholds are practical tolerances, not significance tests. Stop after six attempts, one audit, educational guide/results regardless of outcome. No automatic deeper/wider/longer-budget follow-up.

## Verification and sources

Preserve prior source/protocol/data/report/checkpoint bytes. Verify existing supervision audit hashes. Before training compare logits, objective and all parameter gradients for each architecture/seed against independent math (absolute1e-10). Transformer: existing manual normalization/RoPE plus SDPA. GRU: explicit gate-by-gate recurrent unroll, no nn.GRU/GRUCell. Check causality, standalone versus joint-batch evaluation, chunked hidden-state carry and parameter counts. Deliberately resetting hidden state at each token must disagree on a fixture. Prefix labels are independently checked against the existing last-reset/parity solver.

Audit all saved training/validation/test full logits, final/operation predictions and final losses, absolute1e-10 and exact argmax agreement. Replay initialization and batch streams; check shared embeddings, parameter counts, optimizer steps, hashes, the pretest score snapshot and all decision criteria.

[Cho et al.2014, sections2.1/2.3](https://aclanthology.org/D14-1179.pdf) supplies recurrent/gated-state context. [PyTorch2.11 GRU equations and implementation note](https://docs.pytorch.org/docs/2.11/generated/torch.nn.GRU.html), checked against the installed2.11.0+cu128 module, defines the actual reset convention. Recurrent structure is a hypothesis about suitability, not evidence of local success.
