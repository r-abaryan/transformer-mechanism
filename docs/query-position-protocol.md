# Query-position pathway protocol

Fixed before execution: five frozen checkpoints, CPU float64, 1,024 unique length-ten examples generated with seed 20261001, excluding all 512 previously inspected length-ten sequences. No tuning or training.

Compare original output with donor positional vectors at positions 0, 5, 6, 7 and 10. Position 6 is the training query position; 10 is a no-change control; other positions test specificity. For each donor patch (a) query Q only, (b) query-token K/V only, (c) all query-token Q/K/V, (d) query residual only, (e) both attention and residual paths. Symbol rows stay fixed. Use library hooks to patch these paths, then remove them after each call.

Hypothesis from prior evidence: the whole position-six patch achieves at least 95% accuracy for every seed on fresh examples. Treat this as confirmation of a rescue effect, not model generalization without intervention. Path decomposition and other donor positions are exploratory. Define a path as sufficient for the same operational rescue criterion only if all five seeds reach 95%; this is not a universal causal-sufficiency claim.

Record accuracy, loss, corrected/broken examples relative to each seed's original predictions, and baseline reconstruction/no-change errors. Validate combined position-six patch against the prior explicit implementation and require all no-change patches to match original logits within 1e-10. Preserve checkpoint/source/protocol/data hashes. Shared examples across seeds do not provide five independent datasets.
