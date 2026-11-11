# Distinct values and training-held-out mappings

**The fixed intervention helps on whole mappings absent from training. It still fails reliable rescue at larger shifts.** Repeated-answer redundancy cannot explain this test's gains because all three values differ.

## What makes this stricter?

If A->1, B->2, C->3 and the query is B, reading another pair's literal value gives the wrong answer. Values are unique. A mapping is the complete set of three key/value associations, independent of query and order. No selected mapping occurs anywhere in the models' training data.

Correct output still does not prove the exact internal retrieval path. We report attention separately and retain that limitation. The [mechanism assessment](../../assessments/mechanism-assessment.md) summarizes the evidence so far without proposing another patch.

## Protocol and coverage

The [protocol](../../strict-retrieval-protocol.md) was fixed before predictions. Enumerating all 1,344 distinct-value mappings left only 21 absent from the shared training set. Select 16 deterministically, four per omitted value label, without looking at predictions. Enumerate six pair orders and three queries each: 288 primary inputs. Target labels occur 72 times each; query ranks occur 96 times each. Supplementary evaluation includes all 21 mappings and 378 inputs.

Five frozen models; shifts 0/1/2/4/6/8; unchanged forward versus the fixed block-1 rows-3/5 paired-key forcing. No training, compact donors or additional sites. Every input has three pairs and eight tokens. The 288 observations represent only 16 mappings with dependent variants, not 288 independent mapping trials.

Across all 21 mappings, training overlap is zero, but 17 occur in earlier confirmation validation, 16 in its test, 20 in earlier confirmation crops, and six in value-counterfactual inputs. These sets overlap each other. This is training-held-out evaluation, **not globally untouched confirmation**. The available evidence cannot honestly support that stronger label.

## Primary results

Means across five seeds. Robust mapping accuracy requires every order/query variant of that mapping to be correct.

| Shift | Unchanged accuracy | Fixed intervention | Intervention loss, nats | Robust mapping accuracy |
|---|---:|---:|---:|---:|
| 0 | 100% | 99.93% | 0.0023 | 98.75% |
| +1 | 89.38% | 99.31% | 0.0201 | 93.75% |
| +2 | 70.00% | 97.64% | 0.1100 | 82.50% |
| +4 | 50.97% | 91.46% | 0.4865 | 58.75% |
| +6 | 81.46% | 88.33% | 0.6927 | 46.25% |
| +8 | 60.49% | 82.50% | 0.9204 | 32.50% |

| Seed, intervention | +1 | +2 | +4 | +6 | +8 |
|---|---:|---:|---:|---:|---:|
| 137 | 100% | 99.65% | 98.61% | 95.49% | 93.40% |
| 163 | 100% | 100% | 99.65% | 97.92% | 90.28% |
| 191 | 98.96% | 97.22% | 92.01% | 90.97% | 88.19% |
| 223 | 97.92% | 92.71% | 72.22% | 70.14% | 66.67% |
| 257 | 99.65% | 98.61% | 94.79% | 87.15% | 73.96% |

Every-seed >=90% passes at +1/+2 and fails at +4/+6/+8. Mean gain >=5 points passes at every positive shift. The combined across-shift criterion fails. Compact preservation within one point per seed passes. These descriptive criteria do not establish statistical significance.

The supplementary all-21-mapping intervention accuracies are 99.95%, 99.47%, 97.62%, 91.48%, 89.10%, 83.12% for shifts 0/1/2/4/6/8. This agrees qualitatively with the primary result; it is not an independent replication because the primary mappings are included.

## Bounded conclusion

The benefit survives removal of repeated answers and exclusion of training mappings. It is therefore not merely an artifact of those two factors. It remains task-specific and unreliable across seeds/shifts. Exact source-routing mechanism, full long-sequence transfer and broader distribution generalization remain unresolved.

We have enough evidence to stop claiming or seeking a universal repair through these patches. A fresh broader claim would need a newly reserved mapping/task distribution and frozen training/evaluation decisions. That would be a new research cycle, not retroactive confirmation of this one. Underlying [attention equations and source comparisons](../attention/qkv-results.md#learn-the-operation) remain unchanged.

## Verification

[Script](../../../experiments/strict_retrieval.py), [exact inputs and mapping flags](../../../results/strict_retrieval_inputs.json), and [raw report](../../../results/strict_retrieval.json) preserve predictions, per-head target attention, losses, mapping robustness and paired rescue/harm counts. Whole-mapping exclusions, label uniqueness and exact balance passed. Baseline/library and attention reconstruction maximum error was 4.67e-14. Causal masks, normalized rows, finite outputs and immutable checkpoints passed. Source hashes were verified and checkpoint/training-split provenance was traced to the original confirmation report.
