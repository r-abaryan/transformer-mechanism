# Learning to retain, overwrite and reset

Frozen2026-10-03 after complete optimizer-step profiling, before data generation/evaluation. Six runs only. No tuning, replacement seeds or automatic extension.

## Task and data

Structured event records, not language tokens. Three keys, four values and an explicit reset record. A separate final query names a key; answer its latest value since the last reset, or class4 (unknown). No architectural state reset is supplied. Both models must process the marker through their learned projections.

Four equally sampled groups: retain (one queried-key write amid other-key updates), overwrite (two queried-key writes with different values), reset_present (queried key rewritten after reset), reset_missing (queried key appears only before reset). Query-relevant positions and reset positions vary. Other writes exclude the query key. These controlled templates test limited behaviors, not arbitrary event programs.

Fresh generator seed2026100301. Train lengths6/8 with96 context families per group/length; validation/test6/8 and test12 with24 each. Four cyclic value renamings per family. Canonical full event-context identity excludes query/group and is disjoint across all pools, so renamings stay together. Shared symbols and templates remain. Total4,992 sequences, including3,072 training. Reset-missing labels are always unknown; groupwise results prevent that shortcut hiding failures elsewhere. No held-out hyperparameter selection.

## Models and budget

| Item | Fixed choice |
|---|---|
| Input | Eight features:3 key indicators,4 value indicators,1 reset indicator; reset has only that final bit |
| Core | One8-by-8 memory, learned linear key/value projections8->8, L2-unit keys, learned sigmoid beta8->1, query embedding3->8, linear5-class readout |
| Delta | alpha=1;222 parameters |
| Gated delta | Adds learned sigmoid alpha8->1;231 parameters. Alpha weights0, bias2 initially. Nine extra parameters are part of the treatment, not parameter-matched capacity |
| Matching | Seeds2711/2719/2729; exact shared initial tensors and minibatch indices. Rotate arm order. Profile seed2709 discarded |
| Budget |3,000 Adam updates, lr0.003, batch64,16 examples per group; alternate lengths6/8. CPUfloat64, one thread, deterministic.192,000 examples/final-query targets and1,344,000 event records per arm |
| Memory reporting | Full forward/backward/Adam state allocation profiled at batch64,length8. GPU unused: allocated/reserved0 bytes. CPU peak memory unmeasured. Cold single-step times are not speed comparisons |
| Measurements | Minibatch loss every500 updates; full final training/validation/test group accuracy and CE loss. Test only after all six attempts and saved pretest snapshot. No intermediate-state supervision |

The gate's parameters and its initial behavior both change; results concern this controlled recipe. Equal examples/updates do not mean equal compute. These are small memory readers, not complete Transformer blocks or reproductions of DeltaNet/Gated DeltaNet.

## Verification and decisions

Independent reverse-scan labels and canonical families; oracle, always-unknown, constant0, first-match, last-value and ignore-reset baselines. Deliberately shifted labels and omitted old-read subtraction must be detected. Compare all model outputs/parameter gradients against explicit matrix transitions at initialization, tolerance1e-10. Audit every saved final prediction and independent CE/group metrics, tolerance1e-10, zero argmax changes. Verify shared tensors, replayed streams, optimizer steps and pretest checkpoint hashes.

Short fit requires every seed/group to reach99% on training and validation at6/8 events. Reliability additionally needs99% test6/8 and90% test12 in every group/seed. Gated adoption requires both short-fit gates, gated reliability, at least10-point test12 gains in BOTH reset groups for EVERY pair, and no more than2-point loss in either retain/overwrite group. Report all group gains even if ineligible. Accuracy alone does not establish a learned gate mechanism. No pure transfer explanation when short fitting fails.

Nonfinite attempts fail without restart; preserve checkpoints and failure details. Stop after six attempts, final audit and concise educational documentation. Main arithmetic source: [Gated Delta Networks v3, section3.1 equation10](https://arxiv.org/html/2412.06464v3); [closed local audit](../topics/attention/gated-delta-memory-results.md).
