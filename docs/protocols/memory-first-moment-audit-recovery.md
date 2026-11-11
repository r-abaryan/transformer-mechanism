# First-moment audit recovery, unchanged training

The executed independent auditor stopped when comparing the report's per-seed criteria list. Its inherited comparison helper handles dictionaries/scalars but not lists. Preserve the original auditor, reports, checkpoints and failed-audit record. Training and evaluation sources are unchanged.

Recovery: recursively compare exact dictionary keys and list lengths/order before delegating scalar comparisons to the original helper. Run the same independent audit entry under that narrowly extended comparator. This repeats independent checks only; it performs no training, changes no score/criterion and relaxes no tolerance. Audit all remaining counts, streams and exposure fields.

Preflight must compare criteria computed independently from supplied two-seed records, catch a reversed/missing list and changed decision, and exercise the saved train/evaluator/audit fixture under the recovered comparator. Formal recovery stops after one complete audit and binds all new recovery source/protocol/preflight hashes in a separate record. The original report and 16 executed optimizer calls remain evidence.

This is a validation implementation defect, not an empirical reset failure. Immediate and four-update criteria retain their original definitions. Mathematical sources remain [Kingma and Ba, Algorithm 1](https://arxiv.org/pdf/1412.6980) and [PyTorch Adam](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html); list comparison is a local software repair.
