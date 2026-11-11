# One fixed training-budget extension

Freeze before execution. Continue all twenty normalization-placement runs from 3,000 to 6,000 updates. Same data, seeds, offsets, Adam settings and model. After update 3,000 the learning rate remains .001 in every condition. Stop at 6,000; no tuning or further extensions against these results.

Original checkpoints contain weights only. Replay updates 1..3,000 with the original recipe. Require exact checkpoint tensors and original minibatch/offset hashes before continuing with the same Adam instance. Save optimizer and RNG states every 500 updates. A resumed process must preserve them. A mismatch stops execution; a nonfinite training loss/gradient is recorded as failure.

Reserve 400 fresh whole mappings using generator 20261230, twenty per value triple. Exclude all 8,420 earlier three-pair mappings, including the entire normalization-placement split. Expand all eighteen order/query variants (7,200 examples). Save before training. Evaluate both original and extended checkpoints on this same new test, only after all twenty continuation attempts finish. This compares budgets on paired examples, without comparing different test sets.

Record full existing-validation accuracy/loss at 3,000 and every 500 continuation updates. Report every seed, test loss, accuracy, fully correct mappings, and paired changes. Each condition passes only if all five seeds finish finitely and reach 99% test accuracy. Catch-up of earlier weak seeds supports budget dependence in this recipe; persistent weakness establishes only failure within 6,000 updates. Neither outcome explains the internal mechanism or establishes a universal normalization ranking.

Verify fresh mapping exclusion/variants/balance, replay equality, matched continuation streams, checkpoint reload predictions and metrics. Preserve the original study. Document the adaptive choice of this follow-up after inspecting its results; fresh examples do not make the choice independent of those findings.
