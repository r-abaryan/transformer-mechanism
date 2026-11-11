# What does “black box” mean here?

With code and weights available, the forward arithmetic is specified. That alone does not give a concise causal explanation of a learned behavior. This distinction is our working definition, not a claim that all mechanisms are permanently unknowable.

| Question | Available evidence | Remaining boundary |
|---|---|---|
| Which arithmetic produced an activation? | Code, weights, intermediate tensors | Access and instrumentation are required |
| Which tokens receive attention? | Attention matrix | Weights omit value content and downstream transformations |
| What information is represented? | Probes and feature inspection | Decodability does not show that the model uses the information |
| Which internal activity matters to an answer? | Controlled ablations and activation patching | Interventions can create unnatural states; redundant routes can conceal effects |
| Does a circuit explain all relevant cases? | Held-out intervention predictions | A few prompts cannot establish universal coverage |

[Jain and Wallace](https://arxiv.org/abs/1902.10186) report difficulties treating attention distributions as explanations. [Wiegreffe and Pinter](https://arxiv.org/abs/1908.04626) challenge the definitions and experimental design, proposing additional controls. Their examined settings do not settle interpretability for every modern transformer. We retain both positions rather than adopting a paper title as a theorem.

[Circuit Tracing](https://www.transformer-circuits.pub/2025/attribution-graphs/methods.html) builds attribution graphs using a more interpretable replacement model. Approximation quality and validation on the original model matter. A readable graph alone is not proof that a faithful mechanism was recovered.

Local S05 demonstrates non-uniqueness of attention distributions for a fixed value matrix. It does not reproduce these papers. Next tests should compare attention rankings, gradients, value-weighted contributions, and actual changes to output under interventions on a trained controlled task. Include random interventions, matched-size controls, multiple seeds, and held-out examples.

Unknown here: which mechanisms our future trained model learns, whether they transfer across seeds, and how much behavior a proposed explanation covers. No trained model has yet been examined.
