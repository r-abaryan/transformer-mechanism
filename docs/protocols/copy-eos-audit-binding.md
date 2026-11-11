# EOS evaluation audit: restore the missing model binding

2026-10-03. Evaluation-only recovery completed and saved all oracle logits/generations and ordinary metrics. Its independent audit stopped before checking its first model because `audit_copy_eos_training.py` uses `Model()` without importing Model. Preserve that executed source, the recovery runner and all reports unchanged.

Run the same audit against `evaluation_recovery/`, explicitly binding `Model` to the existing audited `encoder_decoder_contract.Model` and its directory to the recovered results. These are execution bindings; model equations, decoding, tolerances, metrics, criteria and training budgets stay unchanged. Record the wrapper/protocol hashes and verify the model-class identity before execution. Stop after the unchanged audit completes. No training or evaluation rerun.
