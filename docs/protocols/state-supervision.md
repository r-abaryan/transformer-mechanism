# Intermediate-state supervision: matched comparison

Fixed2026-10-01 before training. Question: does additional intermediate-state supervision improve learning with the same architecture and input streams?

## Controls and objective

Reuse the frozen state-tracking model, forward and data. Three fresh paired seeds2039/2053/2063. Each pair starts with exactly equal parameter values. Same training rows and sample draws in both arms. Alternate arm order by seed. This reuses previously opened validation/test examples; it is not fresh held-out confirmation. Earlier trained models are reference evidence, not controls.

Final-only arm: `L = CE(logits_QUERY, final_state)`.

Intermediate arm: `L = CE(logits_QUERY, final_state) + mean_t CE(logits_after_operation_t, state_t)`.

Each intermediate label is the state AFTER the operation at that input position. No loss on the initial-state token. QUERY keeps its original weight1; the auxiliary term has weight1 after averaging across all operations and examples. The input contains only the original tokens; intermediate answers are never fed back at training or inference. This changes feedback quantity and objective, not architecture. It also increases the total objective scale; this comparison cannot separate feedback placement from objective scaling or added label count.

Two pre-norm MQA blocks, width16, ReLU FFN32, biased LayerNorm, RoPE offset4;4,080 parameters. Six attempts,3,000 Adam.001 updates each, batch64, alternating8/12 operations, CPU float64, one thread, deterministic algorithms. Sample generator seed+1000. No tuning, warmup, clipping, dropout, weight decay or checkpoint selection. Check finite loss/gradients/parameters each step and retain failures. Save final checkpoints, streams and optimizer state; report training/validation final-answer scores every500 steps.

Each completed arm sees192,000 final labels and2,304,000 input tokens. Intermediate arms additionally use1,920,000 operation-state labels. Both perform the same forward/backward architecture computations, but auxiliary loss/gradient work differs; equal steps/tokens do not imply equal runtime or learning cost.

## Decision and evaluation

Primary cells:12-operation no-reset and long-tail validation. Evidence of useful learning improvement requires all six attempts finite/complete; each intermediate seed's equal-group mean validation accuracy improves at least10 percentage points over its paired control; each intermediate seed's equal-group mean training accuracy reaches90%; and no intermediate seed loses more than5pp on either8/12 mixed-reset validation cell. Practical thresholds, not significance tests. A failure may still contain partial gains; preserve those results.

Separately report every seed/group's original99% short and90% long test thresholds as descriptive reliability. Reused test scores cannot revise the validation-only decision. Freeze and hash the decision before test scoring. Evaluate complete training, validation and test pools for each final checkpoint and report per-label final-answer metrics. Also report accuracy of operation-state readouts; these are auxiliary measurements, not proof of a learned recurrent algorithm.

Stop after six attempts, one independent audit, and an educational result document regardless of outcome. No extra seeds, longer budget or automatically selected architecture change.

## Verification

Reuse and hash-check original label/split/forward verification. Independently obtain every prefix-state label using the last-reset/parity oracle. Verify sequential labels against these for all saved rows. Preflight each seed and objective using the separate manual-normalization/RoPE plus SDPA forward: logits/gradients absolute tolerance1e-10. An intentionally shifted state target must disagree with correct targets and change loss/gradients on a fixture. Verify paired initial states and streams, parameter counts and optimizer steps.

Audit all final training/validation/test predictions and losses with the independent forward and labels: logits/loss absolute tolerance1e-10 and exact argmax agreement. Preserve raw scores, checkpoints and source/protocol/input hashes. Prior experiments and protocols stay immutable. The original study's paper references supply context; this local intervention is not a paper replication or architectural impossibility test.
