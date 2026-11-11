# Query-switch diagnostic: fixed protocol

Evaluate the five saved FFN32 models from the width comparison. No training or parameter changes. Use every four-pair test context (7,200 rows,200 dependent families). Keep its eight context tokens and marker fixed. Evaluate the original repeated-key query and both singly occurring keys, ordered numerically. Answers follow the latest-value rule. Distinct values ensure both changed queries require different answers from the repeated-key answer.

This separates query transfer from length transfer by using the trained four-pair length. Contexts were previously evaluated; switched queries are outside the training query distribution. This is a diagnostic, not fresh benchmark confirmation or an architecture comparison.

Report original accuracy, switched-query accuracy, unchanged-prediction rate, repeated-key-answer rate on switched queries, all-three-query context correctness and maximum logit change. A counterexample with changed logits refutes exact query invariance on tested inputs; unchanged argmax alone does not prove the query is ignored. Reliable transfer criterion: every seed >=90% switched-query accuracy. This fixed threshold is descriptive engineering policy, not statistical significance.

Verify original logits against archived scores; independently reconstruct all original/switched logits using matrix rotations and SDPA within1e-10. Validate that only the last token changes; derive labels with a separate reverse scan; save logits and provenance hashes. Stop after five checkpoints and verification. Do not tune or retrain from these results. Failure can arise from unseen query types and does not prove a unique internal shortcut or transformer incapacity.
