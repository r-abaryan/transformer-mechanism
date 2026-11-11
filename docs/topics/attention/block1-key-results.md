# How block 1 contributes to the useful keys

**Block 1's compact attention contribution is sufficient for strong rescue in this controlled key-construction test.** Its residual or FFN contribution alone is not.

## Short explanation

A value token does not enter block 2 as its original embedding. Block 1 has already changed its representation. We separated three contributions: the incoming representation, the attention output and the FFN output. We mixed compact and shifted versions, then built block-2 K from the result.

Only the resulting K vectors at value-token positions were inserted into the recipient. We did not replace the recipient's entire block-1 output. Its block-2 V and residual remained shifted. Final Q stayed compact, as in the preceding experiment. The baseline is therefore 63.57%.

## Method and equations

Follow the [fixed protocol](../../block1-key-protocol.md): five frozen models, 5,120 inspected crops per model, no training. For block 1:

\[
u=x+a,\quad f=\mathrm{FFN}(\mathrm{LN}_2(u)),\quad y=u+f,
\qquad K_2=\mathrm{LN}_{1,2}(y)W_K^T+b_K.
\]

x is the incoming residual representation; a is block-1 attention output. LN1,2 means the normalization before block-2 attention. In the first test, x,a,f are saved from separate compact/shifted runs and mixed without recalculating f. In the second test, f is recomputed from the mixed u. The distinction matters: a live FFN responds to its new input, while a saved donor output does not.

Only K2 slots 1,3,5 are transferred, in both heads. Q at the final position is compact. Other K, all V and downstream recipient execution stay shifted. Layer normalization is recomputed; it prevents interpreting these as additive shares of K or accuracy.

## Frozen-component results

Bits mean x,a,f respectively: 1 selects compact, 0 selects shifted. Means across five seeds.

| x a f | Compact components | Accuracy | Cross-entropy | Every seed >=90% |
|---|---|---:|---:|---|
| 000 | None | 63.57% | 2.0033 | No |
| 001 | FFN only | 68.29% | 1.7668 | No |
| 010 | Attention only | 94.75% | 0.2408 | Yes |
| 011 | Attention + FFN | 95.32% | 0.2065 | Yes |
| 100 | Residual only | 62.08% | 2.1799 | No |
| 101 | Residual + FFN | 67.98% | 1.8253 | No |
| 110 | Residual + attention | 97.21% | 0.1131 | Yes |
| 111 | All three | 98.66% | 0.0502 | Yes |

## Recomputed-FFN results

Here the FFN receives the mixed residual-plus-attention input. It uses unchanged weights.

| Compact components before FFN | Accuracy | Cross-entropy | Every seed >=90% |
|---|---:|---:|---|
| None | 63.57% | 2.0033 | No |
| Attention only | 95.60% | 0.2041 | Yes |
| Residual only | 60.84% | 2.2887 | No |
| Residual + attention | 98.66% | 0.0502 | Yes |

| Seed | Compact attention, saved shifted FFN | Compact attention, recomputed FFN |
|---|---:|---:|
| 137 | 96.56% | 97.52% |
| 163 | 95.51% | 96.43% |
| 191 | 93.44% | 94.43% |
| 223 | 95.27% | 96.25% |
| 257 | 92.97% | 93.36% |

The attention-only rescue survives either FFN treatment. Compact residual and FFN contributions can still improve results when combined with compact attention. Failure of their isolated replacements does not make them irrelevant.

## What this establishes

Block 1's attention output supplies a useful contribution to the value-position representation from which block 2 builds K. This narrows the construction path behind the earlier routing rescue. It does not show which source tokens block 1 reads or what its heads encode. We have not proved that it copies each preceding key into its value position.

All results depend on compact final Q, donor activations and a shifted recipient. These are artificial mixtures on reused, dependent crops. They identify sufficient interventions, not necessity, additive mediation or a deployable repair. The [existing attention equations and source comparison](qkv-results.md#learn-the-operation) still apply; no new architecture or benchmark reproduction is claimed.

## Verification and next question

[Script](../../../experiments/block1_keys.py) and [raw results](../../../results/block1_keys.json) preserve all 12 conditions per seed and paired rescue/harm counts. Cached block-1 sum reconstruction, FFN endpoint reconstruction and mixed/live endpoint logit checks all had zero error. Shifted/compact K endpoints reproduced prior metrics within 1e-10. Labels, finite outputs, checkpoint integrity, source hashes and paired-count arithmetic passed.

Follow-up completed: [paired-key versus self/other-key source interventions](block1-source-results.md). Paired-key forcing rescues under the fixed downstream construction, and its removal harms compact-source performance. A full forward test without compact donors is next.
