# Same answer count, different query feedback

The [gradient diagnostic](query-gradient-variability-results.md) found substantial query variation in a raw beta-control derivative. The [training comparison](query-feedback-training-results.md) tests whether changing that feedback policy changes actual learning.

## What changes

A history contains m distinct keys. Both arms process the same 64 histories and read m answers from each.

| Policy | Queries for one four-key history | Answer count | Possible omission |
|---|---|---|---|
| Exhaustive | A, B, C, D | 4 | None |
| Sampled with replacement | B, B, D, A, for example | 4 | C in this example |

The memory input order stays fixed. All queries read the same final state. This changes supervision sampling, not the memory architecture or write order. Both arms start with identical weights and receive identical history draws, updates and optimizer settings.

## Why the expected objective stays the same

For one history, exhaustive loss is

$$L_h=\frac1m\sum_{q=1}^m\ell(h,q).$$

The random arm draws $q_1,\ldots,q_m$ uniformly with replacement and uses

$$\hat L_h=\frac1m\sum_{j=1}^m\ell(h,q_j),\qquad\mathbb E[\hat L_h]=L_h.$$

The derivative has the same expectation at fixed weights. Duplicates and omissions change its realized value. Conditional on a history, averaging m independent draws divides query-gradient variance by m. Reading every key removes that conditional variance.

Using the earlier within/between notation, a batch of H independent histories has

$$\operatorname{Var}(\hat g_{sampled})=\frac{V_{history}+V_{query}/m}{H},\qquad
\operatorname{Var}(\hat g_{exhaustive})=\frac{V_{history}}H.$$

This local variance derivation concerns a raw coordinate at fixed weights. [McCandlish et al. 2018, section 2.2](https://arxiv.org/html/1812.06162v1#S2.SS2) supplies the iid averaging/covariance rule. The replacement control already averages m answers; comparing it with exhaustive feedback is stricter than comparing exhaustive feedback with one query.

## What learning can tell us

The whole model learns jointly. Different realized gradients can produce different Adam moments and later weights, even when their expectation agrees. Exact per-update coverage also differs. A policy effect does not identify beta-only causality or show that variance alone caused every earlier failure.

| Observed outcome | Interpretation |
|---|---|
| Exhaustive improves fitting in both matched pairs | Feedback policy helps under this fixed recipe and budget |
| Both improve relative to the old one-query scores | More feedback may help; shared gains do not isolate removal of query randomness |
| Both remain inadequately fitted | This intervention is insufficient under the tested conditions; capacity, optimization and representation remain open |
| Training fits, held-out maps fail | A generalization gap remains |
| One seed or group loses | Keep that loss visible; a good mean is not reliable benefit |

Both arms have 14.5 times the old answer count, averaged across the four loads. Within this comparison, histories, answer counts and forward/read shapes match; exact elapsed runtime need not. The two reused seeds make this a development intervention. Reserved test data measure held-out behavior, not fresh-seed architecture confirmation.

[DeltaNet, section 2.2](https://arxiv.org/html/2406.06484v2) defines the correction/read rule. Our 615-parameter standalone reader uses a globally shared beta and differs from full DeltaNet. Neither cited paper predicts the local outcome. See the [results](query-feedback-training-results.md) for fitting, held-out groups, complete-map scores and failed criteria.
