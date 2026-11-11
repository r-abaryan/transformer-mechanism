# Global versus pair-only position shifts

Fixed before execution. Use the same five frozen deep models and all compact crops from subset confirmation. Reuse inspected data; this is diagnostic, not fresh confirmation. No training or token changes. Enumerate the same query-preserving subsets in the same order.

Test shifts s=0,1,2,4,6,8. Global: p_i=i+s for all eight tokens. Pair-only: p_i=i+s for i<6, and p_6=6,p_7=7. Global shifts preserve every relative positional distance because (i+s)-(j+s)=i-j. Pair-only shifts preserve distances within the pairs but change pair-to-tail distances. Positive pair-only shifts may collide with or pass tail position numbers; causal masking still follows token order. This artificial control cannot uniquely isolate a relative-distance mechanism.

Report each seed/length/shift: accuracy, cross-entropy, maximum logit change, changed-prediction count, rescued/harmed count versus unshifted, and fraction of parent episodes correct under every subset. Primary descriptive tests: do global logits remain invariant within 1e-10; does every seed retain >=90% accuracy at every positive global shift? Report each shift separately too. No significance claim or automatic tuning.

Check global relative-distance identity, pair-tail coordinate formulas, unchanged labels, s=0 global/pair equivalence, and reproduction of previous compact metrics within 1e-12. Preserve source/protocol/input/checkpoint hashes; ensure finite outputs and unchanged checkpoints. Global-shift failure establishes sensitivity despite unchanged relative distances under this intervention. It does not establish which layer causes it or explain all failures on full long sequences.
