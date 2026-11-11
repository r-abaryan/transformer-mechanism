# Mechanism assessment before stricter evaluation

This closes the current diagnostic chain, not transformer research as a whole. No novel architecture or universal repair is established.

| Evidence | Bounded conclusion | Still unknown |
|---|---|---|
| [Depth replication](../topics/training/depth-confirmation-results.md) | Two blocks learned our same-length task; wider single block did not under fixed recipes | Other recipes, compute-matched comparison, broader tasks |
| [Global shifts](../topics/positions/shift-results.md) | Predictions depend on absolute position under tested shifts | Full cause of positional sensitivity |
| [Matched QK/V replacements](../topics/attention/qkv-results.md) | Compact routing reads useful information from shifted values | Upstream origin and general necessity |
| [Block-1/source interventions](../topics/attention/block1-source-results.md) | Paired-key reading helps construct useful value-position representations | Complete binding circuit |
| [Direct forcing and confirmation](../topics/retrieval/pair-confirmation-results.md) | A fixed task-specific forcing rule helps, but fails all-seed robustness at larger shifts | Reliable full-sequence or distribution transfer |
| [Value counterfactuals](../topics/retrieval/value-counterfactual-results.md) | Answer repetition improves accuracy without increased average target-slot attention | Exact mediation; accuracy alone does not prove retrieval |

Working explanation: block-1 reading can contribute key-associated information at value positions; block 2 can use routing over those representations to answer. Interventions support parts of this explanation. They do not establish every link, uniqueness or generality. Donor patches and forced attention are diagnostic manipulations, not learned improvements.

Stop adding patches for now. The next test fixes unique values so each supplied wrong value corresponds to a wrong answer. Hold out whole mappings from model training, balance target labels/query ranks, and compare the unchanged model with the already fixed intervention. Report limited mapping coverage, prior research overlap and every losing seed. Source equations remain those in the [Q/K/V guide](../topics/attention/qkv-results.md#learn-the-operation).

Completed: [strict retrieval evaluation](../topics/retrieval/strict-retrieval-results.md) preserves the intervention's benefit on distinct-value, training-held-out mappings, but fails every-seed 90% at +4/+6/+8. Only 21 eligible mappings exist and most have appeared in prior diagnostics. Close this cycle with a bounded benefit and explicit lack of reliable general repair. A new distribution with a reserved holdout is needed for genuinely fresh confirmation.

New-cycle follow-up: [twelve keys/six values with reservation before training](../topics/training/reserved-cycle-results.md) confirms useful mean shift gains, but again fails all-seed robustness at +4/+6/+8. It also fails compact-preservation tolerance for seed 433. Retain the forcing rule as a diagnostic, not a reliable architectural repair.
