# Initial results, 2026-09-29

Executed `experiments/attention_spikes.py --output results/attention_spikes.json` using CPython 3.14.7 on Windows, CPU binary64 arithmetic. Exit code 0; **8 of 8 experiments passed**. Raw measurements, configuration, timestamp and source SHA-256 are in [the JSON report](../../../results/attention_spikes.json).

| Experiment | Observation | Bounded conclusion |
|---|---|---|
| S01 scaling | Scaled sample variance 0.9659-1.0335 over 3 seeds, widths 4/32/128, 4,000 samples per case | Consistent with variance-one derivation for independent standard normal inputs |
| S02 causal mask | Earlier-output change 0; unmasked control change about 7.9998 | Future edits were excluded by this head's causal mask |
| S03 permutation | Equivariance error 0; fixed positional signal control differed by 1.7144 | Position-free, unmasked attention satisfied the tested equivariance |
| S04 derivative | Maximum finite-difference error 3.924e-11; wrong Jacobian error 0.3836 | Analytic derivative agrees at the selected point; control detects a wrong derivative |
| S05 non-uniqueness | Different attention rows both produced 1; altered values gave difference 0.2 | Attention distributions do not uniquely identify the value aggregate |
| S06 online softmax | Maximum dense/streaming difference 5.552e-17 across 6 cases | The scalar online identity matched dense arithmetic within tolerance |
| S07 reassociation | Raw-product error 0; softmax replacement difference 4.3576 | Moving multiplication is valid for raw products but the proposed softmax deletion is not equivalent |
| S08 storage | 256, 1,024, 4,096 elements at lengths 16, 32, 64 | Explicit dense score count quadruples when length doubles |

C01-C08 in the ledger now have local numerical evidence within these fixtures. C09-C12 remain untested here. Positive/negative controls ran inside the probes; no production bug fix or regression-repair claim is being made.

## What follows and what does not

We can explain and inspect the attention arithmetic precisely. These experiments also expose important limits on common shorthand: an attention heatmap need not uniquely explain an output; avoiding quadratic score storage need not change the attention function; and removing softmax to reorder products changes the function.

This initial probe stage did not test training. Subsequent [decoder verification](decoder-verification.md) and [controlled training](../retrieval/first-token-results.md) are complete within their stated scope. No paper benchmark, GPU performance claim, learned causal mechanism or novel solution has been established. See the [current summary](../../assessments/evidence-summary.md).

## Environment observations

`nvidia-smi` reported an NVIDIA GeForce RTX 4060 Laptop GPU with 8188 MiB total memory. The default `python` command resolves to a Windows Store alias; the run used the existing interpreter at `C:/Users/Russ/AppData/Roaming/uv/python/cpython-3.14.7-windows-x86_64-none/python.exe`. An optional `uv python list` inventory command could not access its default cache under the sandbox. Direct interpreter execution succeeded, so no environment installation or cache change was necessary. GPU runtime compatibility remains unverified.
