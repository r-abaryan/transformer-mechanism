# Source-marker case-count correction

The original [protocol](copy-source-end-marker.md) lists three seeds, two architectures, eight symbols, two source-length pairs and four formats. That Cartesian product is **3 x 2 x 8 x 2 x 4 = 384 paired cells**, or 768 source/prefix examples. Its stated 192 cells and the producer's final assertion were arithmetic mistakes.

The first execution completed the prescribed cases and numerical/control checks, then failed its count assertion before saving a report. Its source, protocol and design remain unchanged in their original locations. No training occurred.

The recovery changes only that final literal from 192 to 384 in an in-memory copy of the main function. It redirects artifacts to a separate `count_recovery` directory and uses this correction protocol. The original scenario list, thresholds, labels, checkpoints and equations are unchanged. Record both function hashes and relevant dependencies before/after. Stop after the same full Cartesian product, not an expanded probe set. This is an implementation/reporting correction, separate from any empirical criterion.
