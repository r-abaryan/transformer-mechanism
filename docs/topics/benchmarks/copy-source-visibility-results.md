# Added decoder read positions explain a recoverable first-token failure

Executed 2026-10-07. **Keeping the full encoder unchanged and hiding added positions from cross-attention recovers all 351 native first-token errors on the random longer pools, with zero harms.** Removing their encoder influence alone rescues 26 answers and harms 50. This localizes a sufficient intervention point; it does not explain the training history.

[Short guide](copy-source-visibility-guide.md) | [Frozen protocol](../../protocols/copy-source-visibility.md) | [Producer](../../../experiments/copy_source_visibility.py) | [Independent audit code](../../../experiments/audit_copy_source_visibility.py)

## Method

Use the three saved marked RoPE models, seeds 3163/3167/3169, and eight unchanged pools: train6, validation6, test6/8/10, uniform6/8/10. Models have 1,888 parameters. Training/validation evaluation here covers first tokens at length 6 only; the earlier [full training-fit results](copy-marker-training-results.md) remain separate. Pools are opened evidence, not fresh confirmation.

All rows use BOS only and target their original first token. E0 blocks added content slots, indices 6 through n-1, as encoder keys/values. M0 blocks them from decoder cross-attention. E1/M1 expose them. Source EOS remains visible at position n. Encoder-masked slots retain their own query/token residuals. Two controls use ordinary six-token prefixes (`short`) or force both heads to source position zero (`aligned`).

Three seeds x eight pools x six conditions = 144 cells and 13,104 first-token outputs per implementation. CPU float64, one thread, batches up to 64, zero training/optimizer calls. Producer evaluation took 1.044 seconds excluding import, preflight and audit. These are tiny single-token forwards, not training or a hardware benchmark.

## Results

Each row contains 128 original sequences. Accuracy is first-token accuracy, not full-copy accuracy. CE uses natural logarithms.

| Seed | Length | Native accuracy / CE | Encoder-only exclusion accuracy | Decoder-only exclusion accuracy / CE | Both excluded accuracy |
|---|---:|---:|---:|---:|---:|
| 3163 | 8 | 55.47% / 3.306826 | 46.09% | 100% / 0.000169 | 100% |
| 3163 | 10 | 56.25% / 3.056674 | 48.44% | 100% / 0.000168 | 100% |
| 3167 | 8 | 38.28% / 4.120113 | 39.06% | 100% / 0.000163 | 100% |
| 3167 | 10 | 50.00% / 2.924901 | 48.44% | 100% / 0.000158 | 100% |
| 3169 | 8 | 67.97% / 2.082215 | 67.97% | 100% / 0.000171 | 100% |
| 3169 | 10 | 57.81% / 2.997325 | 57.03% | 100% / 0.000171 | 100% |

All nine present random-long seed/group cells recover to 100% under decoder-only or joint exclusion. Groups use the original sequence's distinct/repeated status, even for cropped controls; length 10 has no distinct group. Complete per-group accuracy, CE, margin and attention mass, including null absent groups, are in the [432-row table](../../../results/copy_source_visibility/metrics.csv).

| Frozen criterion/control | Outcome |
|---|---|
| Decoder exclusion: at least 10-point gain and positive mean-margin gain in every primary group | Pass; gains 29.69-65.63 points, margins +7.717 to +12.710 |
| Encoder exclusion: same rule | Fail; accuracy changes -10.94 to +1.56 points |
| Joint exclusion and complete first-token recovery | Both pass |
| Ordinary short versus E0M0 accuracy differs by more than 1 point | No flagged groups; both are 100%, but margins differ |
| Train6, validation6, test6 and every uniform pool | First-token accuracy 100% in all six conditions |
| Aligned positive control | First-token accuracy 100% throughout |

Interaction in mean margin ranges from -0.645 to +2.540 across the primary groups. Encoder context therefore changes the effect quantitatively; its influence is not zero. Decoder exclusion alone is nevertheless sufficient in every evaluated case. These are controlled within-model effects on reused examples, not independent replications per answer.

## Interpretation and stopping

The learned full-source encoder can support correct first-token answers. The native decoder read becomes unreliable when it includes the added positions. This narrows the earlier one-hot-alignment finding: we retain native softmax over the first six positions plus EOS and still recover all first-token errors.

We have **not** isolated normalization-only dilution from distractor score/value effects, proved a unique cause, explained why training learned this read, or repaired full-sequence copying. The task-informed mask is not a learned solution. Encoder-only harms and the interactions remain part of the result.

Stop this diagnostic. The outcome supports a local instance of competing-read failure, already discussed in [Chiang and Cholak (2022), sections 5.1-5.3](https://aclanthology.org/2022.acl-long.527.pdf), inspected for the design. Their FIRST model and scaling remedy differ from our source mask; no novelty or source-predicted score is claimed. No temperature sweep, head subdivision or training run is queued. Any later learning study must predict a representation change beyond re-demonstrating this recovery.

## Verification

Three preflight methods pass: 18 initialized output/all-parameter-gradient cases, native equivalence, short/gap positional control, both signs of masked input perturbation, supplied logits/ties/absent groups, saved initialized producer/audit entry, and six deliberately corrupted artifact cases. Fixtures use separate test directories.

The independent audit recomputes all 144 cells and scalar decisions with explicit layer/rotation equations. Maximum tensor difference is 8.171e-14; metric difference 6.661e-15; zero argmax differences. Native and aligned logits match the earlier archived evidence. All 18 protected files remain unchanged. Shared architecture construction/loading limits implementation independence; historical training audits were not rerun.

[Raw report](../../../results/copy_source_visibility/report.json) | [Audit](../../../results/copy_source_visibility/audit.json) | [Freeze manifest](../../../results/copy_source_visibility/lock.json) | [Preflight](../../../results/copy_source_visibility/preflight.json) | [Tests](../../../tests/test_copy_source_visibility.py)
