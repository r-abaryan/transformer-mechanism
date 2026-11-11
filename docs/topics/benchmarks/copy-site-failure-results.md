# The weak cross-only seed mostly fails at stopping

2026-10-07. **Preventing premature EOS restores 127/128 complete length-10 copies in cross-only seed 3767, from 0/128.** This is an inference-time diagnostic with known content length, not a learned repair. The declared 128/128 recovery criterion fails.

[Localization protocol](../../protocols/copy-site-failure.md) · [Intervention protocol](../../protocols/copy-site-eos.md) · [Saved-output reduction](../../../results/copy_site_failure/report.json) · [Intervention and audit](../../../results/copy_site_eos/report.json) · [Every group](../../../results/copy_site_eos/metrics.csv)

## What failed first?

| Weak seed's random pool | First failure | Native full copying |
|---|---|---:|
| Length 8 | Nine examples miss EOS after all content is correct | 119/128 |
| Length 10 | Seven emit EOS at output position 4; 121 at position 5 | 0/128 |

Positions are 1-based. No wrong content precedes these first errors. Teacher-forced and generated first errors agree, as expected while their prefixes remain identical. Under correct prefixes, length 10 has 136 erroneous content decisions in total, despite every terminal EOS decision being correct. Later teacher-forced positions are diagnostic; ordinary generation never reaches them after stopping.

The other two cross-only seeds and same-seed combined model copy every random length-8/10 example correctly. All four models pass the short controls.

## One intervention, no retraining

For source content length n, replace only the EOS logit by negative infinity during content decisions 1 through n. Then allow all classes normally. The model generates its own prefix; no content token or terminal EOS is supplied. Cap generation at n+3 as before.

Formally, `z'_EOS(t) = −∞` for `t ≤ n`, with every other logit unchanged. This changes the decision rule, not the weights. The source length supplies oracle information, so the result is conditional recovery.

| Model | Length 8 full, native → blocked | Length 10 full, native → blocked |
|---|---:|---:|
| Cross-only 3767 | 119 → 119 / 128 | 0 → 127 / 128 |
| Cross-only 3779 | 128 → 128 / 128 | 128 → 128 / 128 |
| Cross-only 3793 | 128 → 128 / 128 | 128 → 128 / 128 |
| Combined 3767 | 128 → 128 / 128 | 128 → 128 / 128 |

For seed 3767, blocked content exactness at length 10 is also 127/128 (99.22%). The remaining example substitutes 4 for 6 at content position 8 and then stops correctly. Thus early EOS is not the only error removable from this model. Length-8 content remains 128/128; its nine terminal-EOS errors remain.

No full-copy harms occur in any evaluated group. Weak-seed uniform length 10 improves from 1/8 to 8/8. Cross-only seed 3793's uniform length-10 failure remains at 7/8. All short train6/validation6/test6 controls remain perfect. The CSV includes every seed, pool and distinct/repeated group, including absent groups.

## Meaning and limits

The 0% native length-10 score did not mean the model lacked all useful longer-copy behavior. Premature stopping was sufficient to hide successful content and eventual stopping on 127 examples. Yet this does not explain why training learned that EOS preference, identify a particular head or prove a learned counting mechanism.

This differs from [the original EOS-blocking experiment](copy-stopping-results.md), where substantial content failures survived. Different checkpoints can fail the same task for different immediate reasons; the historical negative result remains valid.

[Newman et al. (2020), sections 1 and 3](https://arxiv.org/html/2010.07174v1#S3) were inspected for separating content and stopping with oracle-length evaluation. That distinction motivates the diagnostic. Their hidden-state explanations are not established here.

## Verification and stopping boundary

Four models × seven opened pools = 28 cells, 2,880 sequence evaluations per path, zero training updates. Known-logit, EOS-boundary, intentional off-by-one and initialized saved-reload checks pass. Native logits reproduce archived evidence within 1.120e-13; producer and explicit-equation reference decoding agree on all native and intervened answers. Scalar and padded-tensor counts agree; 43 protected files remain unchanged. Forward structure/projection helpers are shared, so this is not fully independent model code.

The initial preflight attempt could not write to the Windows temporary directory. It stopped before evaluation; fixtures were moved into unique workspace test folders, then checks and execution passed. No historical artifact was replaced.

The promised localization and single intervention are complete. Preserve the residual content failure and terminal-EOS failures. No additional intervention, retraining or parameter sweep is queued.
