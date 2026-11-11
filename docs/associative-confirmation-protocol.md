# Untouched associative episode confirmation

Fixed after the five training runs and exploratory diagnostics, before executing this confirmation. The three-pair canonical universe contains C(8,3)*4^3*3=10,752 mapping/query identities. The original splits used 10,240, leaving 512 untouched identities. Enumerate the exact complement, randomize pair order with seed 20261050, and test all five unchanged final checkpoints.

This is confirmation on unseen mapping/query episodes, not unseen individual mappings or key/value pairs. Many component associations were necessarily seen during training. No training, tuning or checkpoint choice follows this check.

Prediction to check: no seed meets the original 90% learning threshold, and no seed exceeds the deterministic most-frequent-stored-value baseline by five percentage points. The latter is a new post-hoc confirmation criterion, not the original protocol. Report each seed's accuracy/loss and its difference from the frequency baseline, plus exact lookup and chance. Preserve the full complement hash, source hashes and checkpoint hashes. This finite-universe check does not cover all pair orders or establish a general impossibility result.
