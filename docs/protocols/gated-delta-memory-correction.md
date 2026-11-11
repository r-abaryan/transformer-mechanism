# Fixture expectation correction

2026-10-03. The [first attempt](../../results/gated_delta_memory.json) passed all18 numerical cases, then failed the final correlated-read fixture. Original source and protocol are preserved.

The declared initial state is(2,4.75). With alpha=.5, beta=1, keyA=(1,0), target8, the final state is(8,2.375). Query(.6,.8) therefore reads .6*8+.8*2.375=6.7. The original6.8 expectation was an arithmetic mistake. No implementation equation or tolerance changes.

The [corrected execution source](../../experiments/gated_delta_memory_verified.py) changes that single expected value and uses a separate report/provenance path. Re-execute the identical18 cases and fixtures. Preserve the failed report in the new history hashes. This is correction of a demonstrated test defect, not tuning or a replacement experiment.
