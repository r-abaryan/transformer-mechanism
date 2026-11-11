# Source-end marker: distinction is not correct stopping

2026-10-03. Saved-checkpoint intervention, **zero training updates**. A visible source-end token breaks all 48 tested RoPE length collisions. The saved models get both required answers right in only 5 pairs. This provides a concrete way to remove the tested ambiguity, not a learned copy repair.

## What changed

Compare sources `A A A` and `A A A A` at the common decoder prefix `BOS A A A`. The next labels are EOS and A. Append EOS ID2 to each **encoder input**, while keeping those labels and the decoder prefix unchanged. The marker is a source boundary, not an extra symbol to copy.

Use all six saved sinusoidal/RoPE checkpoints, all eight symbols and length pairs (3,4)/(6,8). Each architecture/format has 48 paired cases. Controls append another A or mask the appended EOS in both encoder and memory attention. Source EOS never appeared in these checkpoints' source training inputs, so marker insertion changes the input format without teaching its meaning.

Encoder and decoder embeddings are separate. Sharing EOS ID2 does not automatically give the source embedding a learned stopping meaning.

## Results

“Distinguished” means maximum next-logit difference >1e-6. “Both correct” requires EOS for the shorter source and A for the longer one. These fixed-prefix probes are not free-running sequence accuracy.

| Source format | RoPE distinguished /48 | RoPE both correct /48 | Sinusoidal distinguished /48 | Sinusoidal both correct /48 |
|---|---|---|---|---|
| Original content | 0 | 0 | 48 | 24 |
| Visible EOS marker | 48 | 5 | 48 | 4 |
| EOS masked in both attentions | 0 | 0 | 48 | 24 |
| Same-symbol marker | 0 | 0 | 48 | 1 |

| RoPE seed, visible marker | Distinguished /16 | Different argmax /16 | Both correct /16 | Smallest logit difference |
|---|---|---|---|---|
| 3119 | 16 | 11 | 2 | 1.154934 |
| 3121 | 16 | 9 | 2 | 0.936572 |
| 3137 | 16 | 8 | 1 | 3.341564 |

The distinction criterion passes in every RoPE seed/symbol/length pair. All original, masked and same-symbol RoPE controls remain collisions within 1e-9. Masked-marker outputs match original outputs in both architectures, with maximum error 6.573e-14. The sinusoidal correctness decrease also shows that an unfamiliar marker can disrupt existing behavior.

## Why the marker matters

Without a marker, identical source values give \(\sum_t P_{pt}v=v\): changing weights or source length cannot change that average. For one encoder head with identical content values v and a distinct marker value u:

\[
o_p=(1-P_{p,\mathrm{END}})v+P_{p,\mathrm{END}}u
   =v+P_{p,\mathrm{END}}(u-v).
\]

If u differs from v, the marker's weight can affect the output. Its position and attention normalization can make that weight length-sensitive. The encoder can then form different memory vectors. Masking the marker removes this route; using another A retains the original identical-value premise.

This breaks the **forced** uniform-value invariance. It does not guarantee distinguishability for every parameter setting, a correctly learned stopping rule or longer-copy success. The equation describes the first attention mixture; later normalization, FFNs and decoder operations also affect the measured logits. The intervention does not isolate a single learned head or prove that RoPE alone supplies the distinction.

## Checks, limits and next question

All 384 paired cells, 768 source/prefix examples, agree with explicit rotation/manual-attention and separate layer equations: maximum logit error 5.951e-14 and zero reference argmax changes. All original witness predictions, masks and checkpoint/dependency hashes pass. Projection/shape utilities are shared with the reference. Counts are probe evaluations, not independent training runs.

The first execution completed the numerical/control probes but failed a mistaken final count of 192. The Cartesian product is 384. Its source/protocol/design remain unchanged; a separate recovery changes only that assertion literal and artifact directory. No scenarios or tolerances changed. [Original protocol](../../protocols/copy-source-end-marker.md), [count correction](../../protocols/copy-source-end-marker-count-correction.md), [recovery record](../../../results/copy_source_end_marker/count_recovery/count_correction.json).

[All logits and results](../../../results/copy_source_end_marker/count_recovery/report.json), [scalar/document verification](../../../results/copy_source_end_marker/count_recovery/verification.json), [producer](../../../experiments/check_copy_source_end_marker.py), [recovery entry](../../../experiments/recover_copy_source_end_marker_count.py). The [six-run positional comparison](copy-rope-results.md) and its quality criteria remain unchanged. This diagnostic addition does not change the six-study copy assessment's count.

**Next question:** when the marker is present during training, can the model learn correct stopping and retain content copying? A bounded matched comparison should give both positional arms the same source marker, keep copy targets unchanged, measure training fit first and reserve longer inputs. More inference-only marker substitutions will not answer that learning question. No new training is launched by this note.

[RoFormer, section 3](https://arxiv.org/html/2104.09864v5) supplies the Q/K rotation identity used by the original variant. The marker experiment and value-averaging derivation above are local evidence, not claims reproduced from that paper.
