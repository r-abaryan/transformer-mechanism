# Global retrieval: paused handoff

**Completed2026-10-02 after resuming.** Both audits passed and [results](../topics/attention/global-retrieval-results.md) are published. Study closed; the checklist below is historical and must not trigger a rerun.

Paused at the user request to conserve usage. All nine fixed6000-update runs, native evaluations and global-connection ablations completed. No active training process remains.

Raw report: [global_retrieval.json](../../results/global_retrieval.json). It includes saved checkpoint/score paths and hashes. Protocol: [global-retrieval.md](../protocols/global-retrieval.md). Educational draft: [guide](../topics/attention/global-retrieval-guide.md).

Preliminary driver decisions: all arms pass eligible training, validation and short-test gates. Full/local longer-transfer gates pass; the global longer-transfer gate fails. These final evaluations have NOT yet passed the independent audit. Do not declare the study closed or infer a cause.

## Resume exactly here

1. Run `.venv-gpu/Scripts/python.exe experiments/audit_global_retrieval.py`.
2. If it passes, run `.venv-gpu/Scripts/python.exe experiments/audit_global_retrieval_dependence.py`.
3. Inspect every seed/distance and ablation, especially global test21 failures. Preserve the frozen criterion and budget; no retraining or replacement seeds.
4. Write `docs/topics/attention/global-retrieval-results.md`, complete the guide, update indices/checkpoint/failure register/AGENTS and validate tables, links and original hashes.

Training sources and protocol are already hash-frozen in the report. Do not edit them to hide failed checks. Preserve any audit failure separately and investigate before deciding whether a correction is needed.
