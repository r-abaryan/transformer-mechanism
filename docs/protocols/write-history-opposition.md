# Where pooled write-gradient opposition arises

Frozen before formal reduction, 2026-10-04. Question: does the negative temporal cross term in [write credit assignment](../topics/attention/write-credit-assignment-results.md) arise inside individual histories or across different histories sharing value embeddings?

Inputs: audited `results/write_credit_assignment/gradients.pt`, its lock/report/audit, and the same data file to recover event value IDs. Seeds 3461/3463, step 12000, 768 train-32 histories each; correct/error/total masks remain fixed. Preserve original gradient weighting, masked CE / (4 H m). No new formal model evaluation, backward pass, optimizer call, training, held-out selection or parameter change. CPU float64, one thread.

An event gradient g_(h,t) has eight coordinates. Embed it into a 6-by-8 zero matrix a_(h,t), using only the row for the event's value ID. Different value rows are orthogonal; raw eight-coordinate dot products cannot be used without this mapping.

Define E0=sum_(h,t) ||a_(h,t)||^2 and J=||sum_(h,t) a_(h,t)||^2. Partition the off-diagonal cross terms into:

- W: same history, different positions, twice the unordered pair dot sum.
- B: different histories, different positions; the residual of the earlier temporal cross term C minus W.
- P: different histories, same position; squared pooled position norms minus E0.

Then C=W+B and J=E0+W+B+P. Report every raw term and each term/E0 (null if E0=0), preserving positive offsets and negative opposition. Do not turn signed terms into causal percentages. Also report per-history W and its negative/positive/unresolved counts, split by histories with/without supervised queries. Sign guard: absolute 1e-14. Counts are correlated histories, including cyclic renamings, not independent seeds.

Six cohort summaries and 4608 per-history rows. Primary reduction uses pooled-sum identities and per-history value scatter. Independent audit uses a chronological prefix calculation: each current event dots with earlier events in the same history, earlier events in other histories, and earlier histories at its current position. Check the saved primary position/embedding sums, old C/J, CSV, summaries, masks, shapes, counts and file hashes. Continuous tolerance: absolute+relative 1e-12. No changed thresholds after evaluation.

Before formal reduction, test aligned, orthogonal, opposing, zero and different-value-row controls; exercise the actual reduction/audit entry on a unique initialized saved primary checkpoint fixture. Reject intentional summary/CSV/input defects. These synthetic preflight forwards/backwards are disclosed separately; no optimizer runs. Stop after one formal reduction and audit. A repair or training-trajectory study needs a separate question and protocol.

Source inspected: [Yang et al., section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2), delta recurrence underlying the saved gradients. This decomposition is local linear algebra; it is not a paper claim about training failure or a full DeltaNet reproduction.
