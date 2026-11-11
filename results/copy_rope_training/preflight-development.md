# Pretraining issues

No training run had started.

1. Windows sandbox denied access to `tempfile.TemporaryDirectory` checkpoint folders. The approved unsandboxed rerun exercised saving, reload, evaluation and audit successfully before the next failure. Temporary sandbox folders were not deleted.
2. The preflight report assembly supplied `logit_error` twice. All 18 equation/evaluator cases reached this reporting line, but the test run failed. Rename its initialized forward error to `initial_logit_error`; preserve the failure here, rerun without changing equations or tolerances.

These are environment/reporting failures, not failed architecture results. The reusable tests write a frozen `preflight.json` only with explicit `--record`; ordinary reruns do not overwrite it.
