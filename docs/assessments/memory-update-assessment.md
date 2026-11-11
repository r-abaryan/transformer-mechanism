# Memory updates: what the evidence establishes

This mathematical branch is closed. Four audited rules answer different questions about storage and editing. The rows are not an accuracy ranking.

| Rule | What changes | Verified behavior | Remaining limitation |
|---|---|---|---|
| [Additive positive-kernel memory](../topics/attention/linear-attention-results.md) | Add each key/value write to S and key features to z | Pairwise and recurrent reads agree | Distinct histories can share the same state |
| [Scalar gated positive-kernel memory](../topics/attention/gated-memory-results.md) | Decay S and z before adding | Changes old-value influence; can reset history | Scalar reset can erase unrelated associations |
| [Plain delta memory](../topics/attention/delta-memory-results.md) | Write the error between incoming value and current key read | Unit-key full correction replaces exactly and preserves orthogonal reads | Overlapping keys can interfere; key length matters |
| [Gated delta memory](../topics/attention/gated-delta-memory-results.md) | Decay, then correct the decayed read | Supports replacement, partial retention and full reset | Decay still weakens unrelated reads; overlap still matters |

**Different conventions matter.** The first two audits use positive features and normalized S/z reads. Delta audits use supplied L2-unit keys, signed vectors and an unnormalized S read. Comparisons of their toy outputs do not isolate one architectural change. All tested recurrence gradients and chunk-state handoffs agree with independent equations within their declared tolerances.

**Mathematical ability is not learned behavior.** We supplied the keys, gates and corrections. We have not shown that training discovers useful geometry or chooses appropriate resets. None of these audits explains the cause of the earlier retrieval failures.

The one [trained softmax/ELU+1 comparison](../topics/attention/linear-retrieval-results.md) favored softmax95.29% versus33.16% on six-pair retrieval, but both all-seed fit/reliability gates failed. That study does not rank gated or delta models.

**Engineering claims remain separate.** A fixed-size forward state is not proof of constant training memory, GPU speed or stronger reasoning. Efficient execution and learned quality were not evaluated by these math studies.

Stop here. A future trained comparison would need a distinct task question, learning-fit gates, matched controls and a frozen budget. No such comparison is selected automatically.

Separately authorized learning follow-up: [memory control](../topics/attention/learned-memory-control-results.md) is now closed. Both plain/gated delta readers achieved100% on every declared group/pool across three seeds. Zero accuracy gain does not support the added gate here; the internal reset mechanism remains unknown. The mathematical findings above remain unchanged.

The subsequent [checkpoint-only reset diagnostic](../topics/attention/memory-reset-diagnostic-results.md) found different history states after reset despite identical correct classes. Disabling scalar forgetting changed no tested class predictions. This narrows the reset interpretation without identifying the complete mechanism; no training or automatic extension follows.
