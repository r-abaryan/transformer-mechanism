# Decoder scale versus non-scalar change

Frozen before transformed-logit evaluation, 2026-10-05. Question: does positive decoder scale dominate the held-out load-32 CE penalty observed at either fixed routing endpoint?

## Locked decomposition

For each seed, let $A=WV^T$ and center the rows of $B=[A\;b]$ by subtracting each column's class mean. This removes only a common logit shift. Define from weights alone

$$s=\frac{\langle B_L,B_N\rangle_F}{\|B_N\|_F^2},\qquad Q=B_L/s.$$

L/N denote the saved L-BFGS/negative-gradient decoder donors. Require finite nonzero norms and $s>0$; otherwise stop that decomposition without an alternate fit. No labels, temperature search, validation selection or intermediate checkpoint choice sets s.

| Variant | Effective decoder | Exact score from audited saved logits |
|---|---|---|
| base | $B_N$ | centered $z_N$ |
| scale | $sB_N$ | $s$ times centered $z_N$ |
| shape | $Q$ | centered $z_L/s$ |
| full | $sQ=B_L$ | centered $z_L$ |

Evaluate all three existing seeds (3473/3479/3481), both fixed routing donors, four variants, three splits and four loads: **3 x 2 x 4 x 3 x 4 = 288 cells**. Use all cached exhaustive-query logits from the audited representation study. Pools and labels are unchanged. Report CE, accuracy and complete maps for all/nonempty unique/repeated groups, retaining absent groups explicitly. CPU float64, one thread. No new producer model forwards or training. Audit independently recomputes the 144 source cells through the saved memory models.

## Prediction and accounting

For each cell, let f00/base, f10/scale, f01/shape and f11/full denote CE. Report both conditional scale effects (f10-f00, f11-f01), both conditional shape effects (f01-f00, f11-f10), and interaction f11-f10-f01+f00. Report scale-first and scale-last ratios to total penalty f11-f00 only when its magnitude exceeds 1e-8; do not clamp ratios or assign a unique causal percentage.

The predeclared scale-dominance rule requires a positive total penalty >1e-8 and both positive scale effects larger than the absolute magnitude of either shape effect in **every seed, route and validation/test load-32 cell** (12 contrasts). Report failures separately. Training and other loads are descriptive, not settings-selection criteria. No new claim of calibration, reliable maps or a historical learning cause follows from this rule.

Positive scaling must preserve base/scale and shape/full class decisions and complete maps. This is an implementation control, not the research hypothesis. Retain native base/full metric controls. Preflight must exercise supplied logits, an initialized saved model, the real scorer/audit, empty groups and intentional projection/logit/label/count/decision/missing-cell faults. Check the score transformations against an actual scaled classifier forward. Independently check the projection using scalar arithmetic, labels, native matrix logits and scalar metrics.

Stop after 288 score cells and audit. Zero optimizer, gradient or solver calls. Preserve all original sources and artifacts; no donor variants or extra scales. Repeated predictions and the six related value shifts are not independent experiments. This decomposition is coordinate/metric dependent and includes bias; it is not the previous warm-start scalar statistic.

Sources inspected: [Guo et al. (2017), section 4.2/equation 9](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) for positive logit scaling and invariant predictions; [Yang et al. (2024), section 2.2](https://arxiv.org/html/2406.06484v2#S2.SS2) for the delta read/write context. Guo et al. fit temperature on validation data; we use a fixed weight projection and do not measure calibration. The decomposition and dominance rule are local choices, not source predictions.
