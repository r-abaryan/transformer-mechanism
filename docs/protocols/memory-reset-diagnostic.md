# Saved-checkpoint reset diagnostic

Frozen2026-10-03 before diagnostic evaluation. No training. Use all six archived learned-memory-control checkpoints and all test6/8/12 rows. Preserve original sources, reports and checkpoints.

- Native full-pool predictions must match archived scores within1e-10 and have zero argmax changes.
- For every reset-present/missing test row, create one paired history by rotating every pre-reset value +1 modulo4. Keep keys, reset, suffix and final query identical. Verify labels independently; these derived inputs are diagnostic counterfactuals, not a fresh confirmation test.
- Measure Frobenius norm of state differences immediately before reset, immediately after reset and after the suffix. Record maximum/mean, fraction<=1e-10 and mean post/pre ratio where pre>1e-10. Also measure final logit differences, paired prediction agreement, accuracy and loss for both histories. Nonzero state difference disproves complete history-state equalization for that tested pair; matching predictions alone does not establish forgotten information or a nullspace mechanism.
- At fixed gated checkpoints set alpha=1 (a) at every event and (b) only at reset events. Leave beta/projections/readout unchanged. Full-pool accuracy/loss per seed/length/group and native prediction changes. Interventions test dependence in this evaluation setting, not benefit of training a different model or a complete explanation of reset.
- Record learned reset alpha/beta. With identical reset event, DeltaS_after=alpha*(I-beta*k*k^T)*DeltaS_before; verify<=1e-10. Alpha/beta values are descriptive, not causal attribution by themselves.
- Independently reconstruct all outputs and states using explicit matrix transitions; <=1e-10 and zero argmax differences. Error control: deliberately read a zero final state and verify output differences>1e-6. Compare interference transport against separate analytical equation. Full initial-state erasure and identical-history state equality are distinct questions; this study tests the latter.
- End after this diagnostic and concise educational results. No retraining, additional inputs, tuning or automatic follow-up selected.
