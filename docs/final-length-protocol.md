# Final bounded length check

Fixed before generation. Test five frozen varied-position models with canonical start +4, no donors, training or patches, at three/four/six pairs (8/10/14 tokens). This evaluates the current recipe; earlier length failure is already established for different models/data. No claim of a controlled improvement over that earlier task.

Seed 20261225. Reserve 420 whole mappings per length, distinct keys from 0..11 and distinct values from 12..17. Balance value combinations: 21 mappings per three-value combination, 28 per four-value combination, 420 for the single six-value combination. For each mapping sample six distinct pair orders and ask all n queries in each order. Thus 7,560/10,080/15,120 examples; each mapping has 6n dependent variants. Answer labels and query ranks balance exactly. Longer sets do not enumerate every possible order.

Exclude the 5,600 three-pair mappings in all prior expanded-task source corpora (reserved-cycle splits, position-training splits, joint confirmation inputs, canonical inputs). Verify matching-task inventory before evaluation. Four/six-pair whole mappings are new at these vocabularies, but may contain familiar three-pair submappings; do not claim novel components. Save data, protocol/source/checkpoint hashes before predictions. No selection by model output.

Criteria: every seed >=99% at three pairs (baseline integrity); every seed >=90% at four pairs and, separately, six pairs. Report all accuracies, losses, margins, queried-rank accuracy and all-sampled-variants mapping correctness. Verify labels, mapping identity/uniqueness, six unique orders, complete queries, balance, finite outputs, library/explicit parity and unchanged checkpoints. Audit summaries from saved logits.

Stop after this evaluation regardless of outcome. No cropping, new offsets, finer attention patches or retraining to chase scores. Move to the separately scoped FFN investigation. Length, distractor count, token roles at positions and available answer labels change together; this test does not isolate their effects.
