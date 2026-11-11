# Transformer architecture research

Copy branch closed: [supplying source alignment to marked RoPE models](docs/topics/benchmarks/copy-marker-alignment-results.md) raises ordinary length 8/10 copying from 10.42%/2.34% to 100% in every tested seed/group, without retraining or suppressing EOS. A shifted route gives 0%. All 72 cells pass independent equations with zero prediction differences. Learned content processing works under supplied alignment; why native routing transfers poorly remains open. [Eight-study synthesis](docs/assessments/copy-generalization-assessment.md). No further copy tuning or subdivisions selected.

Latest learning result: [training with a source-end marker](docs/topics/benchmarks/copy-marker-training-results.md), six completed 6,000-update runs. Both arms fit every training example. RoPE scores 100% on random tests and uniform probes at training lengths 3/4/5/6; longer random copying averages 10.42% at 8 and 2.34% at 10. Sinusoidal averages 0.78%/0% there. Both extrapolation rules and the RoPE long-gain rule fail. All 120 cells pass independent equations with zero prediction differences. The document distinguishes this learning test from the earlier inference-only insertion. Closed without tuning.

Earlier unmarked result: [sinusoidal versus RoPE copying](docs/topics/benchmarks/copy-rope-results.md). Both training-fit gates pass, but the RoPE benefit and reliability rules fail. Ordinary exact means at length 8 are 0%/1.04%; both arms score 0% at 10. A uniform-source boundary proves identical logits for different source lengths at the same prefix. [Inserting a marker without training](docs/topics/benchmarks/copy-source-end-marker.md) breaks that ambiguity but gets only 5/48 RoPE pairs both correct. These earlier results remain valid. [Educational guide](docs/topics/benchmarks/copy-rope-guide.md), [copy assessment](docs/assessments/copy-generalization-assessment.md).

The current phase studies trained mechanisms at toy scale. Mathematical checks support implementation correctness; task interventions test behavior. Broader improvements remain hypotheses requiring suitable training and confirmation evidence.

Started: 2026-09-29. This is an initial research foundation, not an exhaustive review or a reproduced language-model benchmark. No novel architecture or general superiority is established.

## Read in this order

For the latest task-level overview, read the [capability summary](docs/assessments/capability-summary.md): retrieval, aggregation, state tracking, language prediction, graph composition and sequence copying.

Start with the [architecture assessment](docs/assessments/architecture-assessment.md), then the [evidence summary](docs/assessments/evidence-summary.md).

Browse [FFN](docs/topics/ffn/README.md), [attention](docs/topics/attention/README.md), or the [full subject index](docs/topics/README.md). The [documentation index](docs/README.md) also links protocols and primary sources. The [full research reading order](docs/reading-order.md) preserves the earlier sequence.

Consult the [failure register](docs/assessments/failure-register.md) before repeating a study.

Persistent experiment and failure-diagnosis guidance lives in [AGENTS.md](AGENTS.md).

Latest learning study: [encoder-decoder copy](docs/topics/benchmarks/encoder-decoder-copy-results.md). All three seeds fit every training sequence exactly. Mean exact test accuracy96.88% at length3 and90.63% at4;0% at6/8, mostly with early EOS. The95% every-group training/validation criterion fails on validation, not training. All24 pools and12,686 generated tokens pass independent audit with zero differences. [Learning guide](docs/topics/benchmarks/encoder-decoder-copy-guide.md). Closed without tuning.

Latest correctness study: [complete encoder-decoder contract](docs/topics/foundations/encoder-decoder-results.md). All18 CPU float64 cases pass layer/loss equations, gradients, padding controls and135 same-prefix decoding comparisons. [Short guide and diagram](docs/topics/foundations/encoder-decoder-guide.md). That initialized-model audit is closed; learned copying is assessed separately above.

Previous component audit: [cross-attention](docs/topics/attention/cross-attention-results.md). All36 CPU float64 cases pass independent equations, gradients, directional finite differences and masking controls. A weak initial negative-control fixture is preserved and corrected. [Educational guide and diagram](docs/topics/attention/cross-attention-guide.md). Closed within its arithmetic and masking scope.

Latest diagnostic: [GPU learning adequacy](docs/topics/benchmarks/composition-learning-adequacy-results.md). Three60K-parameter pairs finish3,000 updates at batch1,024, eight times the old example exposure. Six-edge one-step validation means91.15% Transformer/35.11% GRU; both fail the99% fit gate. All42 pools and626,688 predictions pass independent audit, with three documented cross-precision changes and stable decisions. [Short explanation and fit-gate diagram](docs/topics/benchmarks/composition-learning-adequacy-guide.md). Closed without tuning; no composition-specific or causal conclusion follows.

Latest architecture comparison: [one-step/two-step graph lookup](docs/topics/benchmarks/recurrent-composition-results.md). Neither recipe masters the one-step control or two-step task. Two-step validation means22.43% Transformer/26.95% GRU; every GRU paired gain is positive but below the frozen10-point rule. All42 pools and626,688 final predictions pass independent audit with zero argmax changes. [Educational guide and diagram](docs/topics/benchmarks/recurrent-composition-guide.md). Study and [cross-task consolidation](docs/assessments/capability-summary.md) are closed; no tuning or further experiment selected.

Previous comparison: [GRU versus Transformer word prediction](docs/topics/benchmarks/recurrent-language-results.md). Transformer validation perplexity396.64 versus GRU597.34 is33.60% lower, with every pair improving. All24 pools and4,645,902 losses pass independent audit; four cross-precision top1 differences are recorded. [Educational guide](docs/topics/benchmarks/recurrent-language-guide.md). Study closed; losses still fell at the budget limit, so eventual ranking remains open. [Retrieval](docs/topics/benchmarks/recurrent-retrieval-results.md) also favored the tested Transformer, while [state tracking](docs/topics/benchmarks/recurrent-state-results.md) favored the earlier GRU recipe. These do not establish an overall winner.

Original state-tracking study: [results](docs/topics/benchmarks/state-tracking-results.md) and [short educational guide](docs/topics/benchmarks/state-tracking-guide.md). That fixed Transformer study remains closed and valid; reliable reset-free tracking fails already at training lengths.

## Project tree

```text
Transformers_RL/
|-- README.md
|-- docs/
|   |-- README.md            # documentation index
|   |-- topics/              # FFN, attention, normalization and other subjects
|   |-- guides/              # index of learning documents
|   |-- results/             # index of study reports
|   |-- reference/           # papers, claims and foundational audit
|   |-- protocols/           # frozen protocol index; future protocols
|   |-- assessments/         # cross-study conclusions and failures
|   |-- decisions/           # frozen research choices
|   |-- figures/             # diagrams and measured plots
|   |-- reading-order.md     # complete earlier reading sequence
|   |-- research-plan.md     # current status and stopping rules
|   |-- *protocol.md         # original paths retained for provenance
|   `-- *.md                 # a few targets linked by frozen protocols
|-- experiments/             # producers and independent audits
|-- results/                 # raw measurements and checkpoints
|-- data/                    # pinned corpus and encoding
|-- requirements-cpu.txt
|-- requirements-gpu.txt
`-- requirements-data.txt
```

## Reproduce the first experiments

Requires Python 3.11 or later, standard library only:

```powershell
python experiments/attention_spikes.py --output results/attention_spikes.json
```

If `python` resolves to the Windows Store alias, use an installed interpreter's full path. This session used CPython 3.14.7. No packages or datasets are required. The script exits nonzero on a failed check and writes the results, configuration, timestamp, interpreter, and source hash. These CPU experiments test mathematics, not GPU speed or trained-model quality.

Research rule: every conclusion must carry its conditions and evidence status. A paper claim, an algebraic derivation, a toy observation, and a benchmark reproduction are different kinds of evidence.
