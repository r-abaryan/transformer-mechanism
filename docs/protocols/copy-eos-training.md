# Does EOS training affect copy-content extrapolation?

Frozen 2026-10-03 after discarded complete-step profiles, before training/evaluation. One diagnostic objective comparison; no architecture change or novel solution. Seeds 3083/3089/3109, paired EOS and content-only arms from identical initial parameters. Stop after six 6,000-update runs and one audit/report; preserve nonfinite failures without restarting. No schedules, candidate search, best checkpoint, replacement seeds or automatic extension.

## Task, objective and exposure

Reuse all frozen `results/copy_length_exposure/data.json` pools. They are already inspected and shared across seeds; this is not fresh confirmation. Train both arms on cycle3/4/5/6, using identical balanced distinct/repeated batches, indices and generator seed+1000. Labels are source+[EOS]; decoder input is BOS+source. Eight content symbols3..10, all16 output classes retained. EOS remains a competing class even without a positive EOS target. Length10 has no distinct group.

For content length n, batch B and alpha1 (EOS) or0 (content-only):

\[
L_\alpha=\frac{\sum_{b,j<n}\mathrm{CE}_{bj}+\alpha\sum_b\mathrm{CE}_{b,n}}{B(n+1)}.
\]

Both arms compute all n+1 target rows. The content denominator remains B(n+1); removing EOS does not increase content gradient weighting. Content-only terminal loss is diagnostic and contributes zero gradient. This differs from renormalizing loss by Bn and from training an output vocabulary without EOS. It isolates removal of the explicit terminal target term within this recipe; later gradients and representations may differ.

Batch sizes3/4/5/6:210/168/140/120, 840 computed rows per update. Per run:957,000 sequences, 4,083,000 content targets, 5,040,000 computed rows. EOS arm adds957,000 positive terminal targets; content-only adds zero. Same updates/data/content feedback, unequal total active targets. Record streams, optimizer counters, visible attention-pair counts and loop wall time including scheduled evaluations/saves. Equal forward shapes do not prove identical wall time.

Unchanged 1,888-parameter encoder-decoder, width8, two heads, one post-norm encoder/decoder layer, ReLU FFN16, sinusoidal positions, dropout0. Adam .001, betas .9/.999, eps1e-8, no clipping/decay/schedule. CPU float64, deterministic native Windows, one thread. GPU unused, peak GPU allocation/reservation unmeasured. Current profile includes backward/allocated Adam states; earlier identical shape profiles are retained.

## Metrics and evaluation

At1,000/3,000/6,000 updates evaluate every full training/validation pool. No selection. Save all six final checkpoints and a pretest snapshot before model test scoring, then score every full pool. Save teacher logits and generated content.

Shared oracle-length decoding: start from BOS, exclude only EOS at every step, generate exactly n symbols using learned attention and each generated prefix. Do not mask PAD/BOS/unused symbols, force source alignment or insert answers. Correct length is privileged information. This compares content, not learning to stop or ordinary generation.

Report oracle content exact/token accuracy, teacher content accuracy with all16 classes and with EOS excluded, conditional teacher exactness, raw content CE, terminal EOS CE/accuracy, weighted objective and teacher content EOS rate per seed/length/group. Content CE is the common loss; the two weighted objectives include different terms and are not interchangeable quality metrics. Separately report ordinary greedy stopping/exact metrics at train6/test6/test8/test10 as descriptive controls, not the primary comparison or a content-only adoption gate.

Training fit requires >=99% oracle content exact and >=99% unrestricted teacher content accuracy in every training group/seed for each arm. Seen-length reliability adds >=95% oracle exact in every test3/4/5/6 group. Removal-benefit rule: both training-fit gates, >=10-point paired oracle exact gains at both test8 and test10 in every seed, and no >5-point regression in any seen-length test group. Content-only length8/10 oracle transfer: seen-length reliability plus >=90% every eligible group/seed, separately per length. Preserve descriptive gains when criteria fail. No ordinary-generation or general causal impossibility claim follows.

## Verification and sources

Reuse independent split/label/baseline checks. Check wrong EOS selection, prefix feedback and off-by-one decoding, unwanted terminal gradient and content-denominator rescaling. Compare first native logits/parameter gradients with independent equations; verify EOS gradient decomposition. Independently reconstruct every final teacher logit and oracle sequence and descriptive ordinary control, recompute scalar losses/metrics and all sampling streams/counters/exposure. Tolerance1e-9, zero teacher argmax/oracle sequence differences. Independently recompute criteria and inject a criterion error. Hash only relevant dependencies before/after.

[Newman et al. 2020, sections1/3](https://arxiv.org/html/2010.07174v1) motivate contrasting EOS/no-EOS training under oracle length. Our fixed content coefficient, task/model/data/budget and retained output class differ; no paper reproduction or hidden-state mechanism claim. The six-run diagnostic closes after audit and short educational documentation.
