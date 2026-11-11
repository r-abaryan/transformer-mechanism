# Watching retention learn

The [endpoint diagnostic](objective-beta-pressure-results.md) measured loss gradients after training. This [reproduction](retention-learning-replay-results.md) observes the original learning path. The recipe, data streams and update count stay fixed. Measurements are passive, and final model/optimizer equality is checked against the original run.

## What beta does

For a normalized key k and value v, the compact reader writes

$$S_{t+1}=S_t+\beta k_t(v_t-S_t^T k_t)^T,\qquad\beta=\operatorname{sigmoid}(\alpha).$$

Here S has shape 16 x 8; k and v are column vectors of sizes 16 and 8. The outer product adds a 16 x 8 correction.

A stronger correction can improve that key's read. When keys overlap, it can also disturb other reads. [DeltaNet, section 2.2](https://arxiv.org/html/2406.06484v2) supplies the correction rule. Our globally shared beta differs from its input-dependent strength and complete architecture.

## What training rewards

Fixed-query training rewards one assigned read per history. Resampled training chooses uniformly among its present keys. At any saved weights, exhaustive all-query loss is the expected resampled objective. A sampled minibatch is noisy and need not have the same derivative.

We inspect both full objectives at steps 0, 10, 100, 300, 1000, 2000 and 3000. Equal weights across the four loads match the original sampler. Seven observations locate measured differences; they cannot date every change between them.

## Why Adam can move against today's gradient

Let g be the sampled derivative with respect to alpha. Plain gradient descent moves opposite g. Adam uses accumulated moments:

$$m_t=.9m_{t-1}+.1g_t,\quad u_t=.999u_{t-1}+.001g_t^2,$$
$$\Delta\alpha_t=-.003\frac{m_t/(1-.9^t)}{\sqrt{u_t/(1-.999^t)}+10^{-8}}.$$

The first moment determines the update's sign. If today's gradient changes sign, accumulated momentum can still move alpha in the old direction. Beta moves in the same direction as alpha because the sigmoid is increasing. These equations follow the [installed-version Adam definition](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html); the recorded updates are checked independently.

## What the observations can establish

| Measurement | What it tells us | Limit |
|---|---|---|
| Identical final model, optimizer and streams | Instrumentation reproduced the original execution | These are reused seeds, not fresh confirmation |
| Conflict at a saved step | The two objectives locally reward opposite beta changes | Does not prove the conflict caused the eventual solution |
| Beta separation before measured conflict | Endpoint conflict need not initiate the split | Coarse observations can miss intervening conflict |
| Adam step against its current gradient | Stored moments determine a different immediate direction | Does not show momentum caused the retention failure |
| All-query errors despite assigned fit | The assigned objective did not ensure every association was fitted | Geometry, decoding and exposure still interact |

All parameters learn together. Passive traces establish chronology and update arithmetic. A causal explanation needs a separately controlled training intervention. Read the [results](retention-learning-replay-results.md) for every seed, exceptions and audit scope.
