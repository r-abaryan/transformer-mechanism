# ADR 0001: establish evidence before proposing replacements

Date: 2026-09-29. Status: accepted for the initial research cycle.

Context: the workspace contains no existing model or experiment infrastructure. We need to distinguish transformer definitions from explanations of learned behavior and empirical architecture claims.

Decision: retain a primary-source register, a claim ledger, explicit mathematical assumptions, and executable small probes with source hashes and raw measurements. Use standard-library Python for initial arithmetic probes. Plan a separately validated tensor implementation for training. Use a decoder-only baseline for initial controlled tasks while documenting encoder-decoder differences. No model family is declared superior in advance.

Alternatives considered: immediately training a large model would confound implementation errors and architecture effects; only collecting papers would leave basic claims unchecked. GPU kernels will be necessary for credible performance comparisons, but add no value to the first scalar identities.

Consequences: initial runs need no installation or downloaded data, but cannot establish practical training quality, GPU efficiency or novelty. The baseline choice is a project decision, not a claim that pre-norm or decoder-only models are universally best.

Revisit when: the mathematical probes fail, an empirical question requires a different model family, or a trainable implementation and a precisely specified benchmark are ready.
