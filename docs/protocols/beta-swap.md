# Does smaller beta contribute to the query-exposure gain?

2026-10-04. Saved-model diagnostic authorized by the user. Fixed-query checkpoints learned beta .95/.96; resampled checkpoints .32/.32. Their geometry and decoders also differ. Question: what changes when only write strength is exchanged within each frozen checkpoint?

## Frozen scope

Use all four completed delta-query-coverage checkpoints, seeds 3461/3463, fixed/resampled arms. Evaluate existing development train/validation pools at loads 2/8/16/32: 768/192 histories per load, 32 events each. Already opened data, no fresh confirmation, new test pools or optimizer updates. Same labels and input order. Original 615 parameters and 16x8 state, CPU float64, one thread.

Two variants per checkpoint: native and donor beta from the opposite arm at the same seed. Copy only donor `write_logit` into a disposable clone. Every other weight stays bitwise unchanged. No beta grid, output rescaling, classifier refit, key replacement or checkpoint selection. Preserve executed evidence; verify native final logits/metrics against the parent.

Read every present key. **Immediate** means just after its first occurrence; never grade a key before it appears. Values are consistent, so later repeats do not change labels. These immediate answers come from different prefix states and do not measure one complete map. **Final** means after all 32 events, with every query reading one shared state. Report accuracy/CE for both stages; complete-map accuracy only for final. Count correct-to-wrong and wrong-to-correct changes versus native, and immediate-to-final transitions. Report every seed, arm, load and split.

Expected Cartesian count: 2 seeds x 2 arms x 8 pools x 2 variants = 64 cells, 128 stage-metric cells. Per variant/checkpoint: `(768+192)*(2+8+16+32)=55,680` answers per stage. Total 445,440 immediate and 445,440 final predictions, 890,880 combined. These are correlated measurements, not independent seeds.

## Interpretation and stopping

Predeclare directional checks at validation load 32: lower donor beta improves fixed final accuracy by >=1 percentage point in both seeds; higher donor beta harms resampled final accuracy by >=1 point in both seeds. Report each and their conjunction. Also report whether either swapped arm attains >=95% complete-map validation accuracy in both seeds; this would remain supplied inference intervention, not a trained repair. Low-load damage and all negative outcomes stay visible.

Only inference effects within a checkpoint are isolated. Opposite-arm beta need not suit its value/classifier calibration. Failure of a swap does not prove beta was irrelevant during training. Neither directional success nor geometric overlap alone explains the historical gain. No adoption claim, extra training or follow-up grid. After audit, consolidate the retention evidence and close this diagnostic.

## Checks and sources

Five reusable preflight methods cover beta-only/self-copy contracts, independent first-write labels/positions, actual initialized saved evaluator/audit at every load with absent groups and supplied logits, wrong scalar/map/logit controls, two-write analytical and numerical beta derivatives, and inverse directional/coverage checks. Audit applies the donor directly from saved state, reconstructs prefixes independently with matrix transitions, reduces CE/map counts and paired changes in Python, and checks native parent agreement and preserved hashes. Shared data/weights/PyTorch; decision replay shares tested policy functions.

[Yang et al., DeltaNet 2024, section 2.2](https://arxiv.org/html/2406.06484v2) was inspected for the correction and read equations. With our transposed state convention, `S'=S+beta*k*(v-S^T*k)^T`; changing beta scales the local correction when S/k/v are held fixed. Across a whole history, previous states also change. The paper uses input-dependent beta, while ours is one learned scalar; it predicts no local swap score. [Earlier query-exposure results](../topics/attention/delta-query-coverage-results.md) motivate the diagnostic.
