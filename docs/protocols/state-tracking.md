# State tracking: one bounded capability test

Fixed 2026-10-01 before training. This is a new task, not an architecture comparison.

## Question and exact rule

Can the existing MQA baseline predict a binary state after ordered updates, including longer reset-free stretches?

Operations are KEEP (0), FLIP (1), SET0 (2), SET1 (3). Input is an initial-state token (4 or5), followed by operations, followed by QUERY (6). The target is the final state, 0 or1. No intermediate answers enter the input or loss.

Write each update as `s' = a*s XOR b`, over binary arithmetic. KEEP=(1,0), FLIP=(1,1), SET0=(0,0), SET1=(0,1). Composition is `(a2*a1, (a2*b1) XOR b2)`. Independently, the answer is the last SET value (or initial state) XOR parity of subsequent FLIPs. Verify both rules exhaustively through seven operations and detect deliberately omitted FLIPs.

## Data and controls

Train at8/12 operations; test at8/12/16/24. Three equally weighted strata: reset-free (KEEP/FLIP only); long-tail (SET0 first, then KEEP/FLIP); mixed (uniform operations conditioned on at least two SETs, including one in the second half). Base initial state is0. Every base sequence is paired with initial1 and SET0/SET1 exchanged; its answer must complement. Keep these pairs in the same split. Thus each cell is exactly label balanced.

Families per stratum: at8, train64/validation32/test32; at12, train256/validation64/test128; at16/24, test256 each. Generator seed20261001. Sample distinct base sequences without replacement across splits. All splits are created before training, but test scores are computed only after all attempts finish. Short splits are disjoint sequences, not disjoint abstract transition functions. Report every stratum and label, not only means. Constant prediction is50%; compare against last-SET/initial-state prediction that ignores FLIPs. Exact symbolic solver is an oracle, not a learned competitor.

## Model, budget and stopping rule

Reuse `mqa_training.model_new('mqa')` and its CPU-double forward unchanged. Replace only task vocabulary/readout: Embedding(7,16), bias-free Linear(16,2). Two pre-norm blocks, two Q/one KV head, head width8, RoPE offset4, biased ReLU FFN32, biased LayerNorm epsilon1e-5. Train from scratch, seeds2003/2011/2027. This does not transfer earlier trained weights.

Each attempt:3,000 Adam.001 updates, batch64, alternating lengths8/12; uniform samples from each length's equal-stratum training pool. Separate sampling generator seed+1000. Final QUERY cross-entropy only. No warmup, clipping, dropout, weight decay, hyperparameter search or best-checkpoint selection. CPU double, one thread, deterministic algorithms. Check finite loss, gradients and parameters every step; retain failed attempts. Save final weights, optimizer/RNG/stream hashes and validation curves every500 steps.

Success requires all three attempts complete and every seed/stratum reaches99% at8/12 and90% at16/24. These are operational reliability thresholds, not significance tests. Even a perfect result cannot prove arbitrary-length state tracking. Failure cannot establish architectural impossibility or identify its cause. Stop after these attempts, an independent audit, and one guide/results report. No automatic tuning, extra seeds or interventions.

## Verification and provenance

Before training compare logits and gradients for each seed against the existing independent manual-normalization/RoPE plus SDPA forward, absolute tolerance1e-10. Check causality and nonzero gradients. Archive source/protocol hashes. After training independently recompute all final validation/test logits, predictions and losses; absolute logit/loss tolerance1e-10 and exact argmax agreement. Replay initialization and sample streams, verify optimizer steps, paired labels, disjointness and the decision. Preserve all prior experiment sources.

## Research context

[Liu et al., NeurIPS2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/510ad3018bbdc5b6e3b10646e2e35771-Abstract-Conference.html) study long-range binary memory with flip-flop language modeling. Our explicit FLIP operation additionally requires parity after the last reset; this is not their benchmark replication. [Li, Guo and Andreas, ICML2025](https://proceedings.mlr.press/v267/li25r.html) investigate how models track state. Their reported mechanisms do not diagnose our model. Abstract-level context was checked for this protocol; no theorem or numerical paper comparison is claimed.
