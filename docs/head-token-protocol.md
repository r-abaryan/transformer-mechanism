# Head and token localization protocol

Fixed before execution. Same five frozen models, 5,120 inspected six-derived crops each, compact matched donors into +4 shifted recipients. V stays shifted. No training. Reuse explicit QKV implementation and stored source/input provenance.

Conditions: head 0 only, head 1 only or both heads, each replacing final-position Q and all-position K. Both-head token controls replace final Q alone; all Q with all K; non-final Q with all K; all K alone. Source-local conditions replace final Q plus K at: symbolic key slots 0/2/4, value slots 1/3/5, tail slots 6/7, queried pair's key slot, queried pair's value slot, or all six pair slots.

Q at position i affects attention output row i. K at position j affects all queries' scores for source j. In this final block, remaining operations are position-wise and loss reads only the final position. Therefore replacing other Q rows cannot change final logits when final Q, K and V stay fixed. Verify this structural prediction numerically: final-Q/all-K equals all-Q/all-K; non-final-Q/all-K equals K-only (<1e-10). This is a control, not a discovered circuit.

Primary sufficiency criteria per head and source subset: every seed >=90% recipient accuracy. Report all seeds, losses, paired rescue/harm and mean accuracy, without selecting a universal head afterward. Head numbers are not aligned semantic roles across independently trained models. Target-source selection is label/query-informed. A source K may encode earlier context rather than its literal token alone. Do not infer necessity or claim attention weights prove a mechanism. Verify prior matched-Q/K/QK metrics, finite logits, unchanged checkpoints and paired arithmetic.
