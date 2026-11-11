# First-token recall results

The baseline learned this narrow retrieval task at the training length, but did not consistently transfer to the longer length. The [protocol](../../first-token-protocol.md) was written before the run. No tuning or retry followed the results.

Input: six symbols followed by a query token. Target: the first symbol. Longer evaluation uses ten symbols. This is first-token retrieval, not full copying, associative recall, or language understanding.

| Seed | Unseen length-six accuracy | Unseen length-ten accuracy |
|---|---:|---:|
| 11 | 100% | 100% |
| 29 | 100% | 97.27% |
| 47 | 100% | 82.03% |
| 73 | 100% | 99.80% |
| 101 | 100% | 97.27% |

The primary criterion (at least 95% test accuracy for every seed) **passed**. The separate length-transfer criterion **failed**. Mean longer-length accuracy was 95.27%, with sample standard deviation 7.52 percentage points across seeds. That mean must not hide seed 47's failure.

Mean cross-entropy was 0.001690 nats on the same-length test and 0.149264 on longer sequences. The uniform four-symbol baseline has expected accuracy 25% and loss log(4)=1.38629. Exact first-symbol lookup achieves 100% by construction. Actual majority and last-symbol baseline accuracies and target counts are retained in the [raw report](../../../results/first_token_training.json).

## What the evidence supports

All runs generalized to the 512 held-out length-six sequences, with exact sequence overlap excluded from training and validation. The runs used one fixed recipe, 600 updates and 19,200 sampled training examples each (134,400 input tokens, including queries). Training sampling was with replacement. The five models saw the same split but different initializations and minibatch draws.

Success on one length did not guarantee the same accuracy at another length under this recipe. The observed gap does not identify its cause: positional representation, attention behavior and optimization differences remain hypotheses. It also does not establish that transformers generally cannot extrapolate.

## Reproduction and artifacts

Run `.venv/Scripts/python.exe experiments/train_first_token.py` from the repository root after installing the pinned CPU environment. [The script](../../../experiments/train_first_token.py) reuses the verified decoder. JSON records source/protocol/split hashes, all validation curves, final metrics, CPU float64 environment, parameter counts and descriptive training times. Final weights for every seed are saved under `results/first_token_checkpoints/` with hashes in the report. Rerunning replaces those generated artifacts.

The task uses no padding. Only the final query logit receives loss; the answer is not supplied after the query. There is no checkpoint selection or test-driven hyperparameter search. This experiment did not measure FLOPs, peak memory, GPU behavior or confidence intervals over independently resampled datasets. Five seeds share the same evaluation examples.

## Next question

Why did seed 47 fail more often at length ten despite perfect same-length accuracy? Inspect the saved models before modifying architecture: compare attention and value contributions, then use controlled interventions to test causal explanations. Further task families and an untouched confirmation set are required before broader conclusions.
