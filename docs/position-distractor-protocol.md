# Frozen depth models: positions and distractors

Fix this diagnostic before execution. Use all five deep confirmation checkpoints and the existing 1,024 four-pair and 1,024 six-pair examples. Reusing inspected data makes this exploratory, not fresh confirmation. No training or architecture changes.

| Condition | Intervention | Question |
|---|---|---|
| Original | Unchanged long input | Reproduce saved accuracy/loss |
| Query reset | Query PE becomes PE(7) | Is the query's absolute position sufficient for rescue? |
| Tail reset | Marker/query PE become PE(6)/PE(7) | Does restoring both tail positions suffice? |
| Three pairs, original positions | Keep queried pair plus first two other pairs in original order; preserve their original PE indices and tail indices | Effect of removing distractors while holding retained token positions fixed |
| Three pairs, tail reset | Same retained pairs and original pair PE; tail PE(6)/PE(7) | Interaction of removal and tail positions |
| Three pairs, compact | Same retained tokens, positions 0..7 | Effect of remapping positions after removal |

Zero-based sinusoidal PE is unchanged: PE(p,2j)=sin(p/10000^(2j/d)), PE(p,2j+1)=cos(p/10000^(2j/d)). A temporary embedding-output hook adds PE(desired)-PE(default), so the original forward adds the desired positions. All attention, FFN, residual and normalization paths then execute normally. The hook also runs with zero correction as a sham; require maximum logit error <1e-10. Verify original metrics match saved results within 1e-12, target lookup survives every crop, all logits are finite, and checkpoint hashes remain unchanged.

For each condition/length report accuracy, cross-entropy, paired rescued/harmed counts relative to original, and all-seed >=90% rescue status. Primary sufficiency questions concern query reset, tail reset and removal with original positions; evaluate each separately without selecting a winning seed. Report every outcome.

Interpretation limits: removing distractors also changes softmax denominators and upstream contextualization. Reset positions may duplicate positions occupied by other tokens. Compact crops jointly shorten the context and restore in-training positions; they are a positive control, not an isolated cause. The deterministic crop selects one of many possible subsets and uses knowledge of the queried key. This is a diagnostic intervention, not a deployable retrieval algorithm. No additive causal decomposition or binding-circuit proof follows from these comparisons.
