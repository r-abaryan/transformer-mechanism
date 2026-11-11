# Shared KV heads: bounded mathematical protocol

Frozen before execution. Question: what cache savings and attention restrictions follow from sharing K/V across query heads? No trained checkpoint changes, training, GPU timing or quality benchmark.

## Four probes

| Probe | Fixed fixture and control | Required check |
|---|---|---|
| Storage and projections | B=2, L=2, T=128, H=4 query heads, G in {4,2,1}, d_k=2, d_v=3, float64; bias-free projection matrices with input/output width D=8 | Measure compact K/V storage bytes and projection element counts; match formulas exactly |
| Already-shared heads | H=4, five tokens, deterministic Q/K/V; G in {4,2,1}; causal and unmasked attention | Compact grouped computation matches MHA with copied within-group K/V, <=1e-10 max error. MQA heads 0/1 must differ in attention by >1e-3 on the unmasked fixture |
| Joint routing restriction | Two heads, one query each, three common visible keys, width 1; scores [1,0,0] and [0,1,0] | Joint reference-key log odds equal I_2 and have rank 2. A shared width-1 K cannot represent this exact joint pattern. Construct shared width-2 K/Q and reproduce probabilities within 1e-10 |
| Mean pooling | Q_1=Q_2=[1], K_1=[1,0,0], K_2=[0,1,0], V_1=[0,1,2], V_2=[3,4,5]; replace both K/V by their head means | Head-output discrepancy >1e-3. Positive control with both heads already using K_1/V_1 has <=1e-10 error |

The one-query fixtures can be final-token attention over three visible keys. All scores are pure dot products with sqrt(d_k) scaling; no additional score-bias matrix. The restriction concerns exact attention patterns, not task errors or complete-network capacity. These are fixed numerical witnesses, not sampled training runs.

For G KV groups, cache payload is `B L T G (d_k+d_v) s`. Bias-free Q/K/V/output projection parameters per layer are `D H d_k + D G(d_k+d_v) + H d_v D`. Report actual tensor storage, excluding allocator overhead, attention temporaries and other model parameters.

The grouped reference assigns adjacent query heads to equal-sized groups. Query projections remain separate. Copying group K/V into H dense heads gives a control with an equivalent function. Sharing unrelated existing heads by arithmetic mean is a separate operation and is not assumed exact.

Reuse the audited head-width rank lemma; add only its within-group stacked-log-odds implication. A standard-library auditor checks softmax and weighted sums using 50-digit Decimal arithmetic, the rank witness using exact rational determinant, and every recorded size/error. Freeze source/protocol hashes before the run. Save raw inputs/outputs and one audit; close after four probes. A failed probe is retained, not replaced by new fixtures or thresholds.

Primary method context: [Shazeer 2019, sections 2-3](https://arxiv.org/html/1911.02150v1) and [Ainslie et al. 2023 v3, sections 2-3](https://arxiv.org/html/2305.13245v3). Their training and performance claims are outside this probe.
