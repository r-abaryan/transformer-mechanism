# Fixed-content positional layout protocol

Predeclared diagnostic: reuse all crops of the 512 four-pair and 512 six-pair subset-confirmation episodes, all five frozen deep models. No training; data already inspected. Hold crop tokens, order, labels and causal mask fixed. Evaluate a complete 2x2x2 positional intervention.

For retained pair ranks j=0,1,2 with original key indices r_j, define offset a=r_0 and gaps g_j=r_j-r_0-2j. Key PE index is 2j+A*a+G*g_j; value index is key index+1. Tail indices are (6,7) when T=0 and the original long marker/query indices when T=1. A,G,T each take 0 or 1. Thus 000 is compact, 110 is the prior tail-reset crop and 111 preserves original positions. A/G affect pair tokens only; this is not a global sequence translation. All indices remain nonnegative. Tail resets may create positional collisions; token-order causality remains unchanged.

Report every cell's accuracy, cross-entropy and fraction of episodes correct under every subset. For each factor, compare on versus off holding the other two factors fixed: paired accuracy difference and rescued/harmed counts. Do not assume additive effects or select a favorable background. Record per-seed results and all-seed >=90% status per cell; no significance claims. Repeated crops are dependent.

Verify labels and crop ordering; reproduce prior cells 000/110/111 (accuracy/loss tolerance 1e-12), verify 000 against unhooked logits (<1e-10), finite outputs and unchanged checkpoint hashes. Preserve source, input, protocol and report hashes. The next question follows the full pattern of conditional effects, not a single winner; architecture changes remain deferred.
