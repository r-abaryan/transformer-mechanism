# Why one query gives a noisy gradient

The [learning replay](retention-learning-replay-results.md) showed different beta paths and many Adam steps against their current gradients. This [diagnostic](query-gradient-variability-results.md) asks how much the raw beta-control gradient varies when the query changes. It uses saved weights and makes no optimizer update.

## Two sources of variation

Let g(h,q) be the derivative of one answer's cross-entropy with respect to alpha, where beta = sigmoid(alpha). A positive derivative favors lower beta under plain gradient descent; a negative one favors higher beta.

At fixed weights and one load, the resampled objective averages histories h and their present-key queries q. Its exact mean derivative is

$$\mu=\frac1H\sum_h\frac1m\sum_q g(h,q).$$

A minibatch samples only some histories and one query per history. Its gradient changes with both choices. Write each history's query-mean derivative as $\bar g_h$. Then

$$\operatorname{Var}(g)=\underbrace{\operatorname{Var}_h(\bar g_h)}_{\text{between histories}}+
\underbrace{\mathbb E_h\operatorname{Var}_q(g\mid h)}_{\text{within-history query choice}}.$$

We enumerate the complete finite training pool. These are population variances with equal weights, not estimates from independent new experiments. This decomposition is a local use of the law of total variance.

## What exhaustive query feedback changes

Reading every present key for a sampled history replaces g(h,q) with its query mean. It keeps the expected all-query objective and removes the second variance term. It still varies between histories.

For 64 independently sampled histories with replacement at fixed weights and one fixed load:

$$\mathrm{SE}_{one}=\sqrt{\frac{\operatorname{Var}(g)}{64}},\qquad
\mathrm{SE}_{all}=\sqrt{\frac{\operatorname{Var}_h(\bar g_h)}{64}}.$$

Here SE means the standard deviation of the minibatch gradient around its exact mean. [McCandlish et al. 2018, section 2.2](https://arxiv.org/html/1812.06162v1#S2.SS2) gives the iid covariance/B relation. Our scalar measurement is not its full-model gradient-noise scale or an optimal batch estimate.

Exhaustive feedback uses m times as many supervised answers at the same history count. Its smaller variance is not evidence of better learning at equal compute or label exposure. Fixed assigned-query loss is a different objective; its mean is reported separately.

## How to read the results

| Quantity | Meaning | Does not establish |
|---|---|---|
| Query variance fraction | Share of raw alpha-gradient variance attributable to query choice within a history | Share of prediction errors caused by sampling |
| Signal/noise = absolute mean divided by SE | Size of the expected scalar derivative relative to sampling variation | Probability that the next gradient has the wrong sign |
| Signal/noise below one | Batch-gradient standard deviation exceeds the mean magnitude | Training cannot work or has converged |
| Smaller exhaustive-feedback SE | A less variable estimator of the same expected objective at fixed weights | Better eventual accuracy or a fair resource comparison |

The analysis holds weights fixed and treats each load separately. The original optimizer cycles loads, accumulates moments and changes all parameters. This does not measure noise in its actual transformed updates or explain its complete trajectory. Only alpha's raw derivative is studied. Zero denominators are reported as null; tiny derivatives remain visible.

A large relative signal/noise ratio can coexist with an extremely small absolute gradient. It does not imply a useful-sized parameter update. The results retain the inherited 1e-7 cutoff for a resolved mean direction.

The per-answer derivative reuses our audited state tangent and sigmoid chain rule. [DeltaNet, section 2.2](https://arxiv.org/html/2406.06484v2) supplies the correction rule; our globally shared beta and standalone reader differ from the full model. Numerical audits check implementation. The measured gradient distributions at learned weights are the research evidence. Read the [results](query-gradient-variability-results.md) for both seeds, every load and remaining limits.
