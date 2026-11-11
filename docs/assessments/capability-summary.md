# Capability summary: what the experiments establish

Additional within-Transformer retrieval study: [causal global relay](../topics/attention/global-retrieval-results.md) learns short distant retrieval, but one seed scores83.33% at both longer distant groups. Removing global edges reduces all distant test groups to1/6. Audited and closed with failed longer reliability; this is a retrieval variant, not a new task family or GRU comparison.

Additional within-Transformer comparison: [full versus local marked-value retrieval](../topics/attention/window-retrieval-results.md), three matched pairs. Both score100% at reachable distances1/3/4 in every pool; only full attention reaches100% at5/8, while local stays at1/6. This simple retrieval variant adds evidence of learned relay within the same retrieval task family, not another GRU/Transformer comparison. All24 pools audited; study closed.

Updated2026-10-02. Six task families, four architecture comparisons, a larger GPU fitting diagnostic and an encoder-decoder copy baseline are closed. Task changes use fresh training, not zero-shot transfer of earlier weights. Different datasets, objectives, variants and budgets prevent combining scores into an overall capability rating.

## Findings

| Task | What it asks | Observed result | Crucial limit |
|---|---|---|---|
| State tracking | Apply ordered updates to a bit | [GRU](../topics/benchmarks/recurrent-state-results.md): every seed100% through24 operations | Transformer short fitting is incomplete; the gap does not isolate recurrence or pure length transfer |
| Retrieval | Select the latest association for a key | [Transformer versus GRU](../topics/benchmarks/recurrent-retrieval-results.md): six-pair balanced mean97.82% versus23.36% | GRU training fit is weak; this is not solely a longer-input failure |
| Aggregation | Identify the majority of input bits | [Five Transformer seeds](../topics/benchmarks/majority-transfer-results.md):100% at9/11/15 bits | At21 bits,97.18% mean hides an86.25% seed. Exact counting and arbitrary-length reliability are unproven; no GRU comparison |
| Word prediction | Predict the next word in WikiText-2 | [Transformer versus GRU](../topics/benchmarks/recurrent-language-results.md): validation PPL396.64 versus597.34,33.60% lower | Training fit is a fixed subset; losses still improve. Convergence and eventual ranking remain unknown |
| Graph composition | Follow one or two relationships supplied in context | [Transformer versus GRU](../topics/benchmarks/recurrent-composition-results.md): two-step validation means22.43% versus26.95% | Both one-step control and two-step training fit fail. Positive GRU paired gains miss the10-point rule; no composition-specific cause or consistent advantage is established |
| Sequence copying | Generate the source symbols in order, then EOS | [Encoder-decoder baseline](../topics/benchmarks/encoder-decoder-copy-results.md):100% training exactness; test3/4 means96.88%/90.63%; test6/8 both0% | Validation groups fail reliability despite perfect training fit. [Blocking early EOS](../topics/benchmarks/copy-stopping-results.md) gives six-symbol content exact1.56% and eight-symbol0%; content failure remains. One architecture and one split |

The separate [larger GPU diagnostic](../topics/benchmarks/composition-learning-adequacy-results.md) raises6-edge one-step validation means to91.15% Transformer/35.11% GRU, but both99% fit gates still fail. All42 pools pass independent audit, with three recorded precision changes and stable decisions. This adds evidence about a fixed recipe on reused graph pools, not a new task family or an isolated scaling effect.

Accuracy counts correct answers. Perplexity is `exp(mean negative log probability of the correct word)`; lower is better. A percentage reduction in perplexity is not an accuracy-point gain.

## Why the ranking cannot isolate architecture

Each comparison uses three fresh paired seeds, shared initial values where applicable, matched data streams and a fixed stopping rule. These control some variation; they do not match every computational detail.

| Comparison | Supervision | Updates | Transformer / GRU parameters | GRU variant |
|---|---|---:|---:|---|
| State | Final plus intermediate operation targets |3,000 |4,080 /4,032 | One hidden28 layer |
| Retrieval | Final answer only |6,000 |4,544 /4,462 | One hidden27 layer |
| Word prediction | Every next-word position |6,000 |1,068,832 /1,068,160 | Two hidden16 layers |
| Graph composition | Final answer; equal one/two-step sampling |6,000 |4,416 /4,524 | One hidden28 layer |
| GPU graph fitting diagnostic |Same final-answer objective; batch1,024 |3,000 |60,672 /60,690 |One hidden110 layer |

The Transformer also differs across tasks in vocabulary, embeddings and output size. The original four comparisons use two width16 MQA blocks; the GPU diagnostic uses width64. Normalization, depth, hidden width, positions, initialization and computational structure differ between arms. Language totals are dominated by vocabulary matrices; its GRU core has17.07% fewer parameters. Equal updates or total parameters do not imply equal compute or fitting capacity. The diagnostic changes several recipe factors together and uses eight times the old example exposure.

Task and supervision change together across studies. Therefore the state/retrieval reversal cannot be attributed to the task alone. The language result is budget-specific. The composition study gives a bounded failure to learn this graph encoding; it does not contradict the earlier successful retrieval setup.

## What we learned

Single lookup, counting, ordered state updates, word prediction and combining relationships ask for different operations. Success on one does not establish success on another.

Training fit must precede a transfer explanation. Low training and held-out accuracy means the recipe did not master the task. High short-length fit with longer-input failure establishes a transfer boundary. Neither pattern, by itself, identifies its cause.

Independent arithmetic audits support implementation correctness within their scope. They do not establish a learned algorithm, convergence or architectural impossibility. Hundreds of thousands of correlated predictions are not hundreds of thousands of independent training runs.

## Stopping decision

The component studies, four architecture comparisons, GPU fitting diagnostic and sequence-copy baseline are closed. Preserve their successes, weak seeds and failed thresholds. No automatic tuning, replacement seeds, budget extension, new experiment or novel architecture proposal is selected.

## Remaining questions

These are evidence gaps, not mandatory extensions of the closed studies.

| Area | What remains unknown | A bounded way to investigate later |
|---|---|---|
| Learning adequacy | Why short fitting remains incomplete, including the larger GPU diagnostic; convergence and eventual ranking | Inspect saved curves/checkpoints first. Another study needs a distinct question and predefined gate |
| Practical scale and transfer | Whether findings survive larger models, longer contexts or another real dataset | One predefined workload and comparison, with declared data/token budgets and independent seeds |
| Mechanisms | Which differences in attention, recurrence, encoding or training account for a gap | Change one declared factor or inspect saved checkpoints with controlled interventions; accuracy alone cannot decide |
| Whole-model GPU efficiency | Training throughput and peak memory at useful batch/context sizes | Measure complete forward/backward/optimizer steps alongside quality; kernel timings alone are insufficient |
| Architecture coverage | [Full encoder-decoder arithmetic](../topics/foundations/encoder-decoder-results.md) and [learned copying](../topics/benchmarks/encoder-decoder-copy-results.md) are now tested; broader sequence generation, GPU/cache behavior and generalization causes remain open | First inspect saved checkpoints for a distinct question; any new training task needs a separate frozen protocol |

The graph learning-adequacy study remains closed with both fit gates failed. The later copy study does fit training data and shows a held-out gap. These are different tasks and neither result invalidates the other. No additional tuning cycle or experiment is selected.

### Future GPU budget

The RTX4060 laptop's8GB capacity is verified. The [hardware profile](../../results/composition_learning_adequacy_profile.json) establishes batch1,024 fits these approximately60K-parameter models at11/15/19 tokens. It does not establish a fitting configuration for other models or contexts. [PyTorch memory counters](https://docs.pytorch.org/docs/2.11/notes/cuda.html#memory-management) distinguish tensor allocation from allocator reservations.

For a single GPU, effective batch = microbatch × accumulation steps. [Gradient accumulation](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html#gradient-accumulation) allows an effective batch to span several passes. At6,000 updates, batch64 gives384,000 examples; batch128 gives768,000. Any comparison must declare this exposure change, precision and compute. Larger batches alone do not make a synthetic task a real-world benchmark.

## Evidence

The [copy report](../../results/encoder_decoder_copy.json) and its [independent audit](../../results/encoder_decoder_copy_audit.json) support the sequence-generation row. The [stopping diagnostic](../../results/copy_stopping.json) independently checks24 pools and15,798 intervention tokens with zero generated differences. It reuses those checkpoints and adds no training seeds. The three fresh runs add a baseline, not another GRU/Transformer comparison.

Headline results were recomputed from [query exposure](../../results/query_exposure.json), [majority](../../results/majority_transfer.json), [WikiText-2](../../results/wikitext2_training.json), [supervision](../../results/state_supervision.json), [recurrent state](../../results/recurrent_state.json), [retrieval](../../results/recurrent_retrieval.json), [word prediction](../../results/recurrent_language.json), [graph composition](../../results/recurrent_composition.json) and [GPU learning adequacy](../../results/composition_learning_adequacy.json). Each report hash matches its passing saved audit. Consolidation adds no independent training replications. Methods, educational explanations and sources remain in the linked reports.
