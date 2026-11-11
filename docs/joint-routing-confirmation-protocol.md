# Fresh-mapping joint-routing confirmation

Status: defined before generation/evaluation; not executed. The question is whether the fixed joint QK rescue transfers to untouched whole mappings. Ordinary model robustness is a separate endpoint. No Q-only/K-only or token-key breakdown, retraining, architecture edits or parameter search.

## Frozen models and fresh inputs

Use the five varied-position checkpoints, seeds 461/487/509/541/571, from position_training.json. Verify their stored hashes and source provenance before evaluation. Same three-pair task: keys 0..11, distinct values 12..17, marker 18, eight tokens.

Exclude every whole mapping in every split of reserved_cycle_splits.json and position_training_splits.json, including reordered/query variants. These files cover the inputs used in the current twelve-key/six-value cycle and its downstream diagnostics. Verify this inventory before generation; if another matching-task corpus exists, add its mappings to the exclusion manifest before making predictions. Earlier eight-key/four-value tasks are a different token vocabulary/distribution, not evidence of semantic independence.

Generator seed 20261223. For each of the 20 value triples, enumerate key triples and value permutations in sorted order, remove excluded mappings, shuffle with one local Python Random instance and take 20 mappings. Total 400 mappings, 7,200 sequences after enumerating all six orders and three queries per mapping. Labels and queried ranks must balance exactly. Save inputs, exclusion/source/protocol/checkpoint hashes before any new-set forward pass. Never regenerate after seeing predictions.

## Conditions and meaning

Evaluate ordinary offsets +4, +8 and +12 for every checkpoint. Then replace both block-2 Q and K at +8 and +12 with same-example +4 Q/K from the same model. Both heads, all rows. Keep recipient V, residual and downstream computation. Do not filter examples based on donor correctness. Report donor +4 correctness as context.

Joint patched attention is softmax(Q_4 K_4^T/sqrt(8) + mask)V_s for recipient offset s. It requires a donor forward pass. This is donor-assisted retrieval, not an unassisted model or an efficiency improvement. All inputs stay eight tokens long; no length-generalization claim.

## Criteria fixed before evaluation

| Endpoint | Criterion | Interpretation |
|---|---|---|
| Primary +8 diagnostic confirmation | Every seed patched accuracy >=95%, and mean paired gain over ordinary +8 >=10 percentage points | Transfer of the observed donor-assisted rescue |
| Secondary +12 stress test | Every seed patched accuracy >=90%, and mean paired gain >=5 points | Broader offset transfer of the same fixed patch |
| Ordinary robustness, separately at +8/+12 | Every seed ordinary accuracy >=90% | Unassisted performance only; patch scores cannot satisfy this |

The primary 95% threshold is a new confirmation target, selected after exploratory results and before fresh predictions. It is not a retroactive success criterion for earlier work. Report all components and any failure, even if baseline headroom limits gain. Do not conflate primary and secondary endpoints or tune to make either pass.

## Evidence and stop rule

Report each seed and means: accuracy, cross-entropy, correct-minus-best-other-logit margin, rescue/harm and fraction of mappings correct across all 18 variants. Save full logits. Verify mapping exclusion, complete variants, labels/ranks, finite outputs, causal normalized attention, library/explicit baseline agreement, sham reconstruction and unchanged checkpoint hashes. Recompute summaries from saved logits in a separate audit. Do not treat 7,200 variants as independent mapping samples.

After all five checkpoints and conditions are evaluated, document results and stop. Preserve failures; no added offsets, selected subsets, changed donors or retraining against this set. A pass confirms this bounded intervention on new inputs in the same task family, not its mechanism, a usable repair, independent implementation or a novel solution.
