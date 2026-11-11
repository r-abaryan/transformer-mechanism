# First-token copy failure: source visibility

Frozen design, 2026-10-07. Question: in the saved marked RoPE models, does the longer source's first-token failure depend on additional cross-attention competitors, encoder-context changes, or their interaction? The [gap assessment](../assessments/learning-mechanism-agenda.md#review-outcome-choose-copying) selects this contrast. Zero training, optimizer calls, new seeds or parameter search.

## Inputs and conditions

Use saved RoPE seeds 3163/3167/3169 and unchanged pools train6, validation6, test6/8/10 and uniform6/8/10 from `results/copy_marker_training/report.json`. These are opened development evidence, not fresh confirmation. Score every row, preserving duplicates between cropped prefixes. Report all/distinct/repeated groups defined by the ORIGINAL content sequence; absent groups have count zero and null metrics. Train/validation coverage is length 6 only; historical full training-fit evidence remains separate.

At content length n, append EOS at position n. Prefix is BOS only; target is the original first content token. Added content slots are indices 6 through n-1. EOS always remains visible. Cross both binary factors:

| Mode | Encoder can attend to added slots E | Decoder can read added slots M |
|---|---|---|
| E0M0 | No | No |
| E0M1 | No | Yes |
| E1M0 | Yes | No |
| E1M1 | Yes | Yes, native long |

All four use identical input tokens, positions and EOS location. Encoder masking removes added slots as keys/values, not as queries: when M=1, these slots still have representations, including their own token residuals. Thus the M contrast changes visible competitors at fixed encoder outputs. The E contrast changes representations at fixed decoder visibility. It does not isolate K from V or mean that invisible encoder slots are zero.

Two controls: `short` uses the first six tokens and EOS at position 6; `aligned` uses the full native encoder and selects source position zero in both cross-attention heads. The latter supplies task knowledge. E0M0 versus short isolates the moved EOS/gap configuration for retained tokens, so E0M0 is not silently equated with an ordinary short source. At n=6 all four factorial conditions equal short. All model weights and other operations stay fixed.

## Metrics and decisions

Record first-token accuracy, natural-log CE, correct-class minus best-other logit margin and matching-source attention mass for each seed/pool/group. Save logits, first Q, K/V, raw cross scores and weights. No full generation or later token/EOS quality is measured.

For each random test8/10 seed and present non-all group, report accuracy and mean-margin effects:

- Encoder exclusion: E0M1 minus E1M1, and E0M0 minus E1M0.
- Competitor exclusion: E1M0 minus E1M1, and E0M0 minus E0M1.
- Joint exclusion: E0M0 minus E1M1.
- Marker/gap control: short minus E0M0.
- Interaction: E1M1 - E1M0 - E0M1 + E0M0, for mean margin and accuracy.

Report rescues and harms relative to E1M1 for every condition and group. A descriptive encoder/competitor/joint recovery criterion passes only if its native-context exclusion gains at least 10 accuracy points AND positive mean margin in every random long seed/group. This chosen practical threshold is not a significance test, unique-cause proof or novelty criterion. Full first-token recovery additionally requires 100% E0M0 accuracy in every such group. Flag any absolute short-versus-E0M0 accuracy difference exceeding 1 point; such cells block treating the masked baseline as ordinary short copying. Keep all effects even if a criterion fails. Uniform and short groups are reported controls, not extra independent seeds.

## Verification and budget

CPU float64, deterministic, one thread, evaluation batch at most 64. Formal budget: exactly 3 seeds x 8 pools x 6 modes = 144 cells. Pool row counts are 256/64/128/128/128/8/8/8, totaling 728; 13,104 first-token outputs per implementation. Producer and independent forward audit each evaluate this budget once. Repeated outputs are not independent experiments. Audit arithmetic tolerance 1e-9, zero argmax differences; no threshold relaxation.

Before formal scoring: exercise saved initialized checkpoint reload, actual producer/audit entry points, supplied known logits, ties, absent groups, independent labels/counts, output and all-parameter gradients, native forward equivalence, n=6 null masks and short/gap equivalence under explicit positions. Check both positive and negative perturbations of excluded encoder slots; independently check cross masking and deliberate unmasking. Corrupt logits, masks/weights, group metadata and cell coverage and require rejection. Fixtures live in unique test directories and never overwrite frozen runs.

Freeze source/protocol/preflight/checkpoint and relevant historical report/audit hashes before scoring. Verify them afterward. Audit uses separate explicit layer, rotation and scalar metric equations; shared checkpoint loader and architecture construction are disclosed. Native E1M1 and aligned first logits must reproduce the archived alignment evidence. Stop after the fixed cells, audit and concise report. No learning experiment follows automatically: endpoint localization must first support a predicted learning intervention.

Source inspected: [Chiang and Cholak (2022), sections 5.1-5.3](https://aclanthology.org/2022.acl-long.527.pdf), existing competing-value explanation and log-length scaling. Their FIRST task is not our copy model. The factorial design, thresholds and measured outcomes are local; no paper predicts these scores.
