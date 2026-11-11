# 0003: WikiText-2 and a matched MHA baseline

Status: selection frozen on2026-10-01. Dataset acquisition and model preflight are separate from the later trained comparison. No benchmark-quality result is implied by this decision.

## Question and choice

Does the accepted MQA tradeoff survive natural-language next-word prediction under a fixed small-model budget? Compare MQA against MHA. This tests one architectural difference; it does not rank the complete recipe against modern language models.

| Choice | Reason and boundary |
|---|---|
| WikiText-2, word-level `wikitext-2-v1` | Published external language-model corpus with official train/validation/test splits. Larger vocabulary and natural text replace our synthetic labels. |
| Publisher `Salesforce/wikitext` | Pin revision `b08601e04326c79dfdd32d625aee71d232d685c3`, file hashes and dataset card. Use all three official splits, without resplitting or sampling a favorable test. |
| MHA baseline, two K/V heads | Same width/depth and training stream as MQA, which has one K/V head. The resource difference is intentional, so parameter counts are not equal. |
| Tiny width16 recipe | Keeps the earlier architectural setting. This is a budget-limited external comparison, not a competitive leaderboard run. Failure could reflect the fixed capacity/budget. |

Use the publisher's word-level files, including existing `<unk>` substitutions. Do not substitute raw text, change vocabulary cutoff or add a tokenizer search. Licenses and attribution are retained in the acquisition manifest/card.

## What we measure

At position t the input word is w_t; its target is w_(t+1). Teacher forcing supplies the preceding true words. A causal mask must prevent access to future inputs. Correct masking alone cannot repair wrong target shifting.

For N evaluated targets:

\[
L=-\frac1N\sum_t\log p(w_{t+1}\mid\text{visible prefix}),
\qquad \mathrm{PPL}=\exp(L).
\]

Lower perplexity is better. It is the geometric mean inverse probability assigned to the true next word. It is not percentage accuracy or a direct measure of reasoning. Average token losses first, then exponentiate; do not average batch perplexities.

Build the vocabulary from training text only, with explicit `<unk>`/`<eos>` handling. Preserve case, punctuation and official split order. Replace each stored line terminator with `<eos>`, preserving blank-line structure; verify decoded rows/newlines before encoding. Map validation/test words absent from training to `<unk>` and report those rates. Freeze encoded streams and their hashes before training. Published token/vocabulary counts are reference values, not assumed local measurements.

## Frozen planned comparison

| Part | Fixed setting |
|---|---|
| Shared structure | Width16, two pre-norm blocks, two query heads of width8, RoPE offset4, ReLU FFN32, LayerNorm gains/biases, epsilon1e-5, untied embedding/readout |
| Only arm difference | MHA two K/V heads; MQA one, initialized from the first template K/V head. All common initial tensors match within seed. |
| Seeds and budget |1723,1741,1753; six attempts;6,000 updates each; Adam lr.001/default betas/epsilon, batch8,64 input positions;3,072,000 next-word targets per attempt |
| Execution | Native Windows RTX4060; FP32, TF32 off, no AMP, dropout, clipping, weight decay, warmup or hyperparameter sweep. CPU float64 independent reference. |
| Batches | Uniform random starts in the training stream, with replacement; identical starts within each paired seed; no cross-split window |
| Checkpoints | Validation at0/1000/.../6000; final checkpoint only; no early selection. Test after all attempts finish. |
| Evaluation | Full official validation/test streams;64-position blocks with reset context/RoPE offset4; retain the preceding boundary token and score each target once, including the final partial block. No artificial tokens between dataset rows beyond specified `<eos>`. |

This reset-context perplexity is protocol-specific. It must not be compared directly with paper scores using different contexts, token handling, models or budgets. Training-token matching does not establish equal FLOPs or wall time.

With vocabulary size V, parameters are `32V+4480` for MHA and `32V+3936` for MQA, a difference of544. Vocabulary matrices dominate here, so earlier10.69% total parameter savings do not transfer unchanged. Compact FP32 K/V payload per64-token sequence is16KiB/8KiB for MHA/MQA, excluding temporaries and training activations. This does not predict speed.

## Pass rule and stopping boundary

Freeze a training-count unigram baseline with add-one smoothing, `p(w)=(count(w)+1)/(training_tokens+V)`. It ignores context and is a learning sanity control.

All six attempts must complete with finite losses/gradients. Every final arm must beat that baseline by>=.05 nats/token on test. MQA's geometric mean paired perplexity ratio must be<=1.05, and every seed's ratio<=1.10. These are research tolerances, not guarantees from a paper. Record per-seed losses, ratios and unknown-word rates; failed attempts fail the rule and are not replaced.

Before training: verify parsing/newlines, train-only vocabulary, target shifting, window coverage, matching initialization, functions/gradients and GPU/reference agreement. A deliberately unshifted-target control must be detected. After training: independently audit all test token losses, metrics, checkpoints, streams and counts. Archive per-token losses/predictions rather than full vocabulary logits; retain a fixed logit fixture for numerical checks.

Close after this paired comparison and its audit. No budget extension, new seed, width/tokenizer change or tuning on opened tests. Any broader benchmark phase needs a separate decision.

## Primary sources

[Merity et al.2016, sections4-5/Table1](https://arxiv.org/html/1609.07843v1) defines WikiText and its original evaluation setting. [Publisher dataset card](https://huggingface.co/datasets/Salesforce/wikitext) distinguishes word/raw variants and official splits. [Shazeer2019, sections2-3](https://arxiv.org/html/1911.02150v1) defines shared K/V attention. Original methodology takes priority over recency; our experiment will not reproduce their model-quality or speed benchmarks.
