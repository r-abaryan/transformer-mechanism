# Removing EOS supervision does not repair content extrapolation

Executed 2026-10-03, six fresh paired training runs. All models fit their training content perfectly. Under shared oracle-length evaluation, eight-symbol exact copying averages **1.56% with EOS training versus 1.30% without terminal EOS supervision**. At ten symbols it is **0% versus 0.26%**. The required benefit and reliable transfer criteria fail.

[Educational guide](copy-eos-training-guide.md) | [Frozen training protocol](../../protocols/copy-eos-training.md)

## Method and the one changed term

Seeds 3083/3089/3109, identical initial parameters and training indices within each pair. Reuse the previously inspected length-exposure pools, training at lengths 3/4/5/6. These are opened data, not fresh confirmation. Unchanged 1,888-parameter encoder-decoder, width 8, two heads, one post-norm encoder/decoder layer, ReLU FFN16, sinusoidal positions, dropout 0. Adam .001, 6,000 updates each; CPU float64, one thread, native Windows.

\[
L_\alpha=\frac{\sum_{b,j<n}\mathrm{CE}(z_{bj},x_{bj})
+\alpha\sum_b\mathrm{CE}(z_{b,n},\mathrm{EOS})}{B(n+1)},
\qquad\alpha\in\{1,0\}.
\]

Content coefficients remain identical. EOS stays in the output vocabulary in both arms. Only its positive terminal target is removed. Later parameter trajectories can differ; the identity of initial content coefficients does not make later gradients identical.

Per run: 957,000 sequences and 4,083,000 content targets. EOS training adds 957,000 terminal targets; content-only adds zero. Both compute 5,040,000 target rows with identical batch shapes and 57,657,000 visible attention pairs before multiplying by heads. Total active supervision differs. Complete-step profiling includes Adam-state allocation; GPU unused and memory unmeasured. Training loops total 499.65 seconds including evaluations/saves, not a speed benchmark.

Full train/validation evaluation at 1,000/3,000/6,000 updates. All six checkpoints and pretest snapshot precede model test scoring. In the primary evaluation, both generate exactly the supplied n symbols, exclude only EOS, and use learned attention and their own prefixes. This measures content under privileged length, not ordinary generation or knowing when to stop.

## Oracle content exact accuracy

Mean percentages across three seeds. One wrong content symbol makes the whole sequence incorrect.

| Test symbols | EOS trained | Content-only |
|---|---:|---:|
| 3 | 100 | 100 |
| 4 | 100 | 100 |
| 5 | 99.22 | 100 |
| 6 | 98.44 | 95.57 |
| 8 | **1.56** | **1.30** |
| 10 | **0** | **0.26** |

Per-seed/group percentages; worst seen test/validation spans all distinct/repeated groups at lengths 3/4/5/6. Eight-symbol groups have 64 examples each; ten-symbol repeated groups have 128. Ten distinct symbols are impossible in this vocabulary.

| Seed | Arm | Test8 distinct | Test8 repeated | Test10 repeated | Worst seen test | Worst validation |
|---|---|---:|---:|---:|---:|---:|
| 3083 | EOS | 0 | 4.69 | 0 | 100 | 100 |
| 3083 | Content-only | 0 | 3.13 | 0.78 | 92.19 | 90.63 |
| 3089 | EOS | 0 | 3.13 | 0 | 98.44 | 100 |
| 3089 | Content-only | 0 | 0 | 0 | 93.75 | 100 |
| 3109 | EOS | 1.56 | 0 | 0 | 95.31 | 93.75 |
| 3109 | Content-only | 1.56 | 3.13 | 0 | 92.19 | 87.50 |

Every training group has 100% unrestricted teacher content accuracy and 100% oracle exactness. Both training-fit gates pass. EOS passes the declared >=95% seen-length test reliability gate; content-only fails. This test gate does not include validation, where EOS also has a 93.75% group. Content-only paired gains at length 8 are -0.78/-1.56/+1.56 points; at length 10, +0.78/0/0. Seed 3083 also fails seen-length preservation. The >=10-point every-seed gain rule and both oracle-transfer rules fail.

## Less EOS competition is not complete copying

Teacher forcing supplies correct earlier symbols. Content CE uses all 16 classes and is the common loss across arms; total objectives include different terms.

| Symbols | EOS content CE | Content-only CE | EOS teacher accuracy, EOS excluded | Content-only teacher accuracy, EOS excluded |
|---|---:|---:|---:|---:|
| 8 | 3.8753 | 2.6844 | 73.44% | 73.01% |
| 10 | 5.6040 | 3.6267 | 54.22% | 56.25% |

Removing terminal supervision lowers mean raw content CE and removes observed teacher EOS selections at content positions. Yet many content choices remain wrong after EOS is excluded, and oracle exactness stays very low. This demonstrates why lower loss or higher token accuracy need not imply accurate whole sequences.

Ordinary-generation controls remain separate: EOS-trained models average 96.61% full exact copying at test6 and 0% at lengths 8/10. All content-only controls fail to emit EOS within the limit, including training-length inputs. They have not learned stopping under this objective; their oracle scores are not ordinary-generation scores.

## Evaluation defects, recovery and verification

Training completed normally. Final evaluation failed because the older ordinary evaluator requires a nonempty distinct group at length10. Evaluation-only recovery reused the existing group-aware evaluator. The independent audit then encountered a missing Model import; a separate runner binds the existing audited model class and recovered directory. Both failed sources/reports remain unchanged. No training updates, data, tolerances or thresholds changed. [Evaluation recovery protocol](../../protocols/copy-eos-evaluation-recovery.md), [audit binding protocol](../../protocols/copy-eos-audit-binding.md).

The completed independent audit verifies 84 oracle cells, 66,048 teacher rows, 10,176 sequences/55,872 generated content tokens and 24 ordinary-control cells. Zero teacher argmax or oracle-sequence changes; maximum logit discrepancy 2.76e-13 and scalar-metric discrepancy 5.33e-15. Initial parameter gradients and the terminal-gradient decomposition pass. Labels, streams, counters, exposures and criteria pass; deliberate loss/controller/criterion errors are detected. The absent-group failure is reproduced and corrected on supplied logits. Archived hashes remain unchanged.

## Interpretation and stopping

Under this fixed-weighting recipe, removing positive terminal EOS targets does not produce the required content-extrapolation gain. This does not prove EOS never affects learned representations, rule out another recipe, or explain the remaining errors. [Newman et al. (2020), sections1/3](https://arxiv.org/html/2010.07174v1) studied different tasks/recipes using EOS/no-EOS training and oracle length. Our null criterion neither reproduces nor refutes their reported gains or mechanisms.

Close the six-run diagnostic without objective-weight tuning, architecture changes or another patching cycle. Original copy studies remain valid.

Evidence: [completed recovered report](../../../results/copy_eos_training/evaluation_recovery/report.json), [audit](../../../results/copy_eos_training/evaluation_recovery/audit.json), [per-seed/group CSV](../../../results/copy_eos_training/evaluation_recovery/metrics.csv), [failed original producer](../../../experiments/copy_eos_training.py), [evaluation recovery](../../../experiments/recover_copy_eos_evaluation.py), [audit runner](../../../experiments/audit_copy_eos_recovery.py), [preserved pretest snapshot](../../../results/copy_eos_training/pretest.json), [failure fixture](../../../results/copy_eos_training/evaluation_recovery/evaluation_failure.json), [audit bindings](../../../results/copy_eos_training/evaluation_recovery/audit_binding.json).
