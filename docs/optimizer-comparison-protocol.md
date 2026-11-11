# Fixed-recipe optimizer comparison

Fixed before execution. Question: does one predetermined Adam recipe improve learning on the full associative task relative to the existing SGD recipe? This is a comparison of optimizer-plus-learning-rate recipes, not a universal optimizer ranking.

Keep TinyDecoder(13,16,2,32), original train/validation splits, CPU float64, batch 64, 3,000 updates, zero dropout and weight decay. Seeds 11/29/47/73/101. For each seed, both recipes start from identical parameter values and consume identical minibatch indices generated with seed+1000.

Recipes: SGD learning rate 0.1, no momentum; Adam learning rate 0.001, betas (0.9,0.999), epsilon 1e-8. No scheduler, search, early stopping or gradient clipping. Final checkpoints only; train/validation evaluation at step 0 and every 500 steps. Record finite-loss/gradient checks and full curves.

Primary learning criterion: Adam final training accuracy at least 90% for every seed. Separate validation criterion: at least 90% for every seed. Relative criterion: Adam validation accuracy exceeds paired SGD by at least five percentage points on average. Report each criterion separately and all paired results, losses and sample SD. Thresholds are project choices, not significance tests.

Do not evaluate original test sets or select an optimizer using them. All original three-pair canonical episodes have already been inspected across prior work, so existing validation is a reused diagnostic set, not fresh confirmation. No untouched-generalization claim is available from this comparison. SGD rerun must match the recorded prior final train/validation metrics within 1e-10, or investigate reproducibility before interpretation.

Report initialization and minibatch hashes, source/protocol/split/checkpoint hashes, versions, parameter count, tokens/examples processed, and descriptive runtime. Equal updates/data are not equal wall time or optimizer-state memory. Preserve original checkpoints. Failure of Adam does not prove an architecture limit; success does not prove general associative reasoning.
