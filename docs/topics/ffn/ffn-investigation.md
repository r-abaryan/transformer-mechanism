# Next subject: the feed-forward network

Question: what does each FFN branch contribute to successful predictions in our trained model? Start with the ordinary three-pair task at canonical positions. Do not assume the FFN explains the longer-sequence failures.

## The operation, in short

Attention mixes information across positions. The FFN transforms each position's features using the same learned function. Its input already contains context from attention. Position-wise computation therefore does not imply context-free information.

Our inspected implementation has two pre-norm blocks, each with width 16 and a 32-unit ReLU FFN. In column-vector notation, for post-attention residual u:

z = LayerNorm(u)

h = ReLU(W1 z + b1)

f = W2 h + b2; next residual = u + f.

W1 is 32 by 16; W2 is 16 by 32. ReLU keeps positive coordinates and sets negative ones to zero. The residual preserves a direct route alongside the learned update. These dimensions and activation were checked on the instantiated model, not inferred from a generic transformer diagram.

Equivalently, f = sum_j h_j v_j + b2, where v_j is column j of W2. This identity motivates asking which activated directions matter. It does not prove that each neuron stores one fact or one interpretable concept.

## What earlier local work already tells us

[Residual/FFN path tests](../normalization/residual-effects-through-ffn-normalization-and-logits.md) previously showed a positional rescue while keeping the original FFN output. That does not show the FFN is unnecessary; its output was retained. Those tests used earlier models and a different task. Reuse their branch-replacement helpers, but do not transfer their conclusions automatically.

## Primary papers and limits

| Source | Author-reported interpretation | What we may test locally |
|---|---|---|
| [Geva et al., EMNLP 2021: FFNs as key-value memories](https://aclanthology.org/2021.emnlp-main.446/) | Learned FFN input patterns and output directions can be interpreted as a memory-like computation in studied language models | Verify the weighted-direction identity; test effects of FFN outputs without assigning semantic facts to toy neurons |
| [Geva et al., EMNLP 2022: predictions in vocabulary space](https://aclanthology.org/2022.emnlp-main.3/) | FFN updates can promote vocabulary concepts in studied models | Measure actual logit changes after the residual update and final normalization |

Reading status: primary publication pages and abstracts checked. Full equations, appendices and implementation audits remain pending before any reproduction claim. This is a focused starting set, not an exhaustive or latest-literature review.

Our final logits are W_out LayerNorm(u+f). Thus W_out f alone generally does not equal the actual change in logits: normalization depends on the combined vector. Compare full execution with controlled branch replacements instead of calling a raw projection the exact causal contribution.

## First bounded experiment to implement next

Use the five frozen varied models and an explicitly exploratory, already-opened three-pair set at canonical +4. No new training, neuron search or architecture change. Record the protocol before interventions.

| Condition | Purpose | Limit |
|---|---|---|
| Ordinary model and unchanged-output sham | Establish baseline and verify replacement code | Passing a sham does not validate an interpretation |
| Zero block-1 FFN, zero block-2 FFN, zero both | Measure dependency on each branch | Zero outputs can be out of distribution; effects are not additive |
| Replace one branch with a saved random permutation across examples, same positions | Preserve the empirical branch-output distribution while disrupting example alignment | Donor content can transfer; report donor-answer agreement and label overlap |

All downstream computation remains live. Record accuracy, cross-entropy, margin and rescue/harm for every seed. Inspect FFN norms and ReLU activation rates descriptively; magnitude alone is not importance. Reconstruct the matrix operation and compare explicit/library outputs. Save logits, hashes and an audit. No claim that an FFN is universally necessary, stores a particular fact, or causes length failure follows from a local ablation.

Status: [branch interventions and math checks completed](ffn-branch-results.md), followed by an [exact final-normalization decomposition](ffn-normalization-results.md). Block 1 has the larger accuracy effect; block 2 still improves loss and can harm when mismatched. The raw FFN-to-logit shortcut fails. Close this bounded cycle without neuron searches. These are exploratory results; reserve fresh mappings if a finding later merits confirmation.
