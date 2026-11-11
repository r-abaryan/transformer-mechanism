# Which context does a saved language model use?

We keep the next word fixed and change its preceding text. This asks how a trained model responds to context, without changing its weights. See the [results](language-context-use-results.md) and [frozen protocol](../../protocols/language-context-use.md).

| Intervention | What it tests | What stays fixed |
|---|---|---|
| Keep recent 32 of 64 words | Dependence on the earlier half | Target, recent words and their RoPE positions |
| Shuffle earlier 32 | Sensitivity to earlier order | Recent 32 words and the word multiset |
| Shuffle recent 31 | Sensitivity to recent order and distance | Earlier half, final input word and target |

The recent shuffle leaves the last input word untouched. Otherwise, a worse score could simply reflect changing the word immediately before the target. Moving other words still changes their distances and contextual representations, so this does not isolate grammar or one attention head.

## Read the loss correctly

For target word $y_i$ and condition $c$,

$$\ell_i^{(c)}=-\log p_\theta(y_i\mid x_i^{(c)}),\qquad
\Delta L_c=\frac1N\sum_i(\ell_i^{(c)}-\ell_i^{(original)}).$$

A positive delta means the intervention makes the correct word less likely on average. Perplexity is $\exp(L)$, so the paired ratio is $\exp(\Delta L_c)$. A ratio of 1.20 means 20% higher perplexity on these targets. It does not mean 20% lower accuracy. Accuracy counts only the most probable word.

Cropping helps some targets and hurts others. Rescues count originally wrong top predictions that become correct; harms count the reverse. A small mean can hide both. Ordinary words, unknown words and row-end tokens are reported separately.

## What we learned

Across all three saved models, removing the earlier half changes mean perplexity by less than 0.3%. Shuffling recent words raises it by 18.21%, 25.13% and 28.28%. Recent context order/distance has a larger measured effect here. Earlier context is not proved irrelevant: cropping harms individual predictions and changes subgroup losses.

These are 3400 fixed validation window endpoints per model, not every validation word. Shuffled inputs differ from the training distribution. A saved model's sensitivity does not establish how much information another model could learn to use.

Source inspected: [O'Connor and Andreas, 2021](https://aclanthology.org/2021.acl-long.70.pdf), section 3.3 and discussion, separates inference sensitivity from usable information measured through transformed-data training. Our diagnostic uses the former. Their longer-context experiments and published scores are not reproduced here. The equations above define our local paired measurements.

This closes the current attention phase. Continue with the [RL roadmap](../rl/rl-roadmap.md).
