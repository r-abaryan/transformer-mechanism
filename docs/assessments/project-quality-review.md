# Project quality and direction review

2026-10-04. **The mechanism research has improved; maintenance has not kept pace.** The recent copy result is supported within its declared conditions. There is no basis here to invalidate the verified runs, or to certify every historical implementation as correct.

## What we have learned

The project now has three useful kinds of evidence: numerical implementation checks, exact structural constraints, and trained-model interventions. Keep them distinct.

State work moved beyond reporting weak accuracy. A declared training-only schedule grid improved fit without changing architecture, but failed its eligibility gate. Breaking the initial/reset correlation improved disagreeing inputs. Later momentum clearing added little beyond continued training. These narrow the explanations without proving a complete state algorithm.

Copy work moved from failed scores to an exact unmarked uniform-source ambiguity, marker-present learning, and conditional recovery. Marked RoPE fits training and all short tests. Supplied source alignment recovers every tested longer answer, including EOS, while native transfer remains weak. This establishes a useful learned content path under task-specific alignment. It does not establish learned routing, a deployable repair or an overall architecture winner. [Copy synthesis](copy-generalization-assessment.md).

## Did we act on the criticism?

| Earlier concern | Current evidence | Assessment |
|---|---|---|
| Coverage replaced explanatory questions | State cue, optimization and copy-boundary interventions follow observed failures | Addressed in recent science; do not resume gap-based topic selection |
| No-tuning prevented fitting development | Eight declared state recipe attempts; selection uses training only; none qualifies | Implemented for that branch, not retroactively applied to every comparison |
| Toy models were asked to settle broad superiority | README now frames toy-scale mechanisms; latest reports disclose supplied alignment and opened data | Scope corrected; broad improvement remains untested |
| Arithmetic audits consumed research overhead | Three reusable test modules exist, but most older checks remain standalone study scripts | Partly addressed |
| Duplicate summaries and stale priorities obscured direction | Five latest-style README entries, 23 latest/current plan entries, repeated prose | Not resolved |
| Versioning and artifact overhead | No Git repository; 19 scripts contain broad JSON-glob patterns; newer studies use targeted dependencies | Partly addressed, no version-control setup |
| MoE design and startup weaknesses | TV/overflow redundancy and near-zero CE fragility are acknowledged; data-before-prerequisite ordering remains | Recorded, not repaired in the frozen historical runner |

The original [direction review](research-direction-review.md) remains the dated assessment of the criticism. This document checks implementation of its recommendations. No new architecture comparison or training sweep was run for this review.

## Findings that need action

| Priority | Finding and evidence | Consequence |
|---|---|---|
| High | [Alignment test entry](../../tests/test_copy_marker_alignment.py), lines 50 and 81–85, writes fixed checkpoint names and rewrites preflight.json without an existence guard. The original main block overwrote an isolated sentinel when supplied a successful test result | A normal rerun can replace provenance artifacts. Use isolated fixtures and explicit, non-overwriting recording in future test entry points; preserve executed sources |
| Medium | [Research plan](../research-plan.md), line 13, still recommends MoE in a current-status paragraph, although the studies are complete and the archival heading begins at line 51 | Readers can mistake historical proposals for today's queue |
| Medium | README, checkpoint, plan and assessments repeat study narratives. Eleven exact prose blocks of at least 160 characters occur in multiple files | Keep one current status, one historical log and links from summaries. Copies can drift even when individual metrics are accurate |
| Medium | Only three test modules cover 222 experiment scripts; this is a file census, not a numerical coverage percentage | Move future implementation checks into reusable tests. Preserve old scripts as executed evidence rather than refactoring them in place |
| Low | AGENTS.md is 20,617 bytes and contains substantial study history; 195 prose/code-example lines match the declared word-number formatting patterns | Short rules and clearer typography would reduce reading overhead. Pattern hits are candidates, not 195 independently confirmed errors |

Archived protocol/report repetition is sometimes necessary for reproducibility. Independent equation implementations intentionally duplicate some mathematics. Repeated current-status prose is the avoidable duplication; neither deleting all archives nor merging every reference implementation is warranted.

## Accuracy checks actually performed

| Check | Outcome | Limit |
|---|---|---|
| Parse 225 experiment/test Python files | No syntax errors | Not import execution or algorithm verification |
| Inspect 320 Markdown files and 2,256 local file links | No missing file targets | Anchors and external URLs not validated |
| Parse 367 accessible result JSON files | All parsed | No full schema/metric audit; 18 result directories were inaccessible |
| Re-run all three test modules in isolated review fixtures | 12 methods pass; 63 initialized equation/gradient cases | Old TemporaryDirectory cleanup replaced by retained review fixtures; standalone recording behavior assessed separately |
| Check five recent study manifests | All audit/report hashes and 132 declared dependency entries match | Provenance, not a re-execution of every trained checkpoint |
| Inspect alignment forward path and generation; check existing final audit | 72 cells audited with zero prediction differences; native archive reproduced | Encoder/self-attention reference shares utilities. No wholly independent software stack or untested-length guarantee |

The reviewed recent code uses correct-prefix, free generation and oracle-content conditions separately. The forced route selects learned encoder values; it does not insert output labels into logits. Both heads and mixture sharpness change together. The report states this confound, so it does not prove that incorrect position selection alone explains the original failure.

Historical missing measurements, failed recipes and recovered evaluator defects remain distinct. A fit failure is valid evidence about its fixed recipe; it is not an architectural impossibility. The old MoE missing-prerequisite vulnerability is not evidence that its completed runs crashed.

Evidence: [inventory](../../results/project_review_20261004/inventory.json), [runtime checks and overwrite witness](../../results/project_review_20261004/runtime.json), [review harness](../../results/project_review_20261004/run_checks.py), [recent provenance](../../results/project_review_20261004/provenance.json), [maintenance scan](../../results/project_review_20261004/maintenance.json).

## Direction from here

Pause new experiments for a small maintenance pass: make test reruns safe, reduce current-status duplication, separate archived priorities, and keep concise rules in AGENTS.md. Preserve all raw evidence and frozen experiment sources. Version code/docs when version-control setup is explicitly authorized.

The copy branch stays closed. Future research should name one unresolved explanatory question, its distinguishing intervention, and its stopping rule. Future architecture adoption needs adequate training fit and fresh confirmation. More scripts, batches or audited predictions alone do not establish novelty or real-world superiority.
