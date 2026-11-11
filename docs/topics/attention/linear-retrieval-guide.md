# Can accumulated attention memory learn retrieval?

For `A:2, B:5, A:4`, asking A requires4. Asking B requires5. The first query tests updating; the second tests selective retrieval. All keys are queried during training.

## The one change

Both models have the same4,544 parameters, initial weights, examples and update count. Their attention weights differ:

\[
a^{soft}_{ij}=\frac{\exp(q_i^Tk_j/\sqrt8)}{\sum_{t\le i}\exp(q_i^Tk_t/\sqrt8)},
\qquad
a^{lin}_{ij}=\frac{\phi(q_i)^T\phi(k_j)}{\sum_{t\le i}\phi(q_i)^T\phi(k_t)}.
\]

Here phi(x)=x+1 for nonnegative x, exp(x) otherwise. Both outputs are weighted sums of values. Only the second rule admits the fixed S/z accumulation verified in our [math audit](linear-attention-guide.md).

This experiment trains the dense pairwise form. A separate recurrent calculation checks its predictions. We measure learning, not kernel speed.

## How to read the results

| Observation | Meaning |
|---|---|
| Low training accuracy | The recipe has not learned the task within its budget |
| High training, weak short held-out accuracy | A generalization gap exists |
| Strong short fit, weak longer accuracy | Length transfer fails under the tested conditions |
| Good mean, weak seed or group | Reliability still fails |

Three paired seeds train for6,000 updates each. Training and validation use3/4 pairs; tests use3/4/6. Report repeated and unique queries separately, so success on one cannot hide failure on the other. The data are reused from an earlier study; this is not confirmation on a newly reserved dataset.

## What this cannot explain alone

The collision example in the math audit proves that a particular state can merge different histories. A trained model can change its key/value representations. An accuracy gap therefore cannot establish that collisions caused its errors.

Both arms apply RoPE to queries and keys. The linear arm then applies a nonlinear positive feature map. That changes RoPE's usual relative-position dot-product identity. A result belongs to this complete recipe, not every linear-attention architecture.

No automatic budget extension or tuning follows. See the [frozen protocol](../../protocols/linear-retrieval.md). The original [linear-attention paper](https://proceedings.mlr.press/v119/katharopoulos20a.html) supplies the kernel construction; this small retrieval comparison is our own experiment.

[Results and stopping decision](linear-retrieval-results.md).
