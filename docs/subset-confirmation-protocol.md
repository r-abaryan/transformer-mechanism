# Retained-subset confirmation

Fixed before execution. Use five frozen deep confirmation models, no training. Generate 512 four-pair and 512 six-pair episodes with seeds 20261121/20261122, excluding canonical mapping/query identities from both original and confirmation long splits. Save exact examples. Fresh long inputs do not imply fresh cropped three-pair mappings.

For every episode retain the queried pair plus every combination of two other pairs, preserving order: C(n-1,2), hence 3 or 10 subsets. Evaluate original long inputs, crops with original PE, crops with marker/query PE reset to 6/7, and compact crops. Reuse verified position-hook machinery. Check labels, unique subset indices, exclusion, zero-correction sham (<1e-10), finite logits and unchanged checkpoint hashes.

Primary question: does removal plus tail reset reach >=90% accuracy for every seed at each length when averaged equally over all eligible subsets? Stronger robustness criterion: for each seed, >=90% of episodes are correct under **every** eligible subset. Report both separately, plus all control conditions, per-seed values, loss and paired rescued/harmed counts. These thresholds are diagnostic, not statistical significance tests.

For correctness c(i,s), report mean over episodes/subsets and mean_i min_s c(i,s). Subsets share an episode and are not independent samples. Also record original deterministic first-two crop accuracy for direct comparison. No winner selection or tuning. A failed robustness criterion directs the next investigation to subset sensitivity before claiming a layer mechanism. Query-aware cropping is not a practical repair; original PE gaps and position collisions remain intervention limitations.
