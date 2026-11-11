# Review of the research direction

2026-10-03. Review of supplied criticism against current code, reports, saved provenance and a primary automata reference. No training, checkpoint interventions or historical edits performed. The supplied dense-only experiment was not independently reproduced here.

**The central criticism is justified:** coverage and closure have become selection criteria in their own right. Correct implementation evidence is useful, but accumulating it does not identify what trained models compute. The next question should follow an unresolved observation, not an empty topic folder.

## Claim-by-claim assessment

| Criticism | Assessment | Practical consequence |
|---|---|---|
| Coverage is replacing a central question | Supported by the coverage review's explicit gap-based MoE selection and repeated latest-status entries | Pause tokenization; choose one explanatory question |
| State tracking is at chance on training | Correct for the reported twelve-operation no-reset mean, 50.65%; not all groups. Mixed-reset training means are much higher | Diagnose group differences; do not describe the entire model as uniformly at chance |
| Long copying still fails after blocking EOS | Substantively correct. Length-six content/full exact is 1.56%/.52%, not zero; length eight is zero | A stopping-only explanation is inadequate, but the cause remains open |
| No-tuning conflates development with confirmation | Fair criticism of our workflow, though rules allow separately authorized new studies | Add a bounded recipe-development phase before future architecture comparisons |
| Tuning on held-out scores is always p-hacking | Too broad: validation-based selection is legitimate when declared and reported; repeated test-based selection corrupts confirmation | Separate development validation from untouched confirmation, record all attempts and budgets |
| Arithmetic audits are not architecture research | Implementation equivalence mainly belongs in reusable tests. Algebraic counterexamples and reachability results carry additional conceptual content | Keep short educational derivations; reduce per-audit report machinery |
| Toy scale cannot settle broad superiority | Correct. Toy scale can still expose exact constraints and causal behavior in the tested model | Frame the current phase as toy-scale mechanism research; broader improvement remains a future hypothesis |
| Git removes the need for historical hash sweeps | Mostly helpful, not automatic: Git only covers tracked content, and checkpoint/data provenance still needs explicit records | Version code/docs/protocols; keep per-run manifests for data, checkpoints and environment |
| Expected balancing improvements are uninformative | Too strong. An objective can fail to produce its intended effect, but this test has limited novelty and mechanistic value | Do not treat a small objective ablation as a major architecture finding |

Sources for task corrections: [state fitting diagnostic](../topics/benchmarks/state-tracking-results.md), [copy stopping diagnostic](../topics/benchmarks/copy-stopping-results.md). Earlier [retrieval interventions](mechanism-assessment.md) did pursue mechanisms; saying the project has built no understanding would also overstate the criticism.

## MoE criticism checked against the executed study

- **Residual bypass is real.** The model can represent a dense classifier through its input/output path. That path has 146 parameters: 6*16+16+16*2+2. A successful separately trained dense model would show that experts are unnecessary to solve this task under that recipe. It would not establish that the saved MoE models do not use experts. The supplied scratch result uses fresh draws and remains externally reported, not a paired local replication.
- **Quality preservation is not trivial.** Dense solvability does not guarantee unchanged CE when adding another objective. Our fixed comparison actually fails its CE criterion. This result is narrow and the criterion's practical meaning is weak near zero.
- **The two routing criteria are redundant.** For counts n_i, T=128 and C=T/E=32, overflow/T = sum(max(n_i-C,0))/T = .5*sum(abs(n_i/T-1/E)) = TV. All saved cells have exactly zero difference. Thus TV reduction >=.05 already implies overflow reduction >=.02. This was documented in the guide, but still should have been removed from the decision design before execution.
- **Relative CE tolerance is fragile.** The frozen 5% rule must remain in the historical result, but failure is not a universal recommendation against balancing. Future rules need justified absolute as well as relative tolerances, chosen before evaluation.
- **Startup ordering is defective when prerequisites are missing.** The code writes data before hashing its protocol/profile. A missing prerequisite leaves a data artifact that blocks the next attempt. The executed protocol does exist and matches its saved hash; all six runs completed 3,000 updates. The alleged missing-file crash is not the history of this completed run. Future runners should validate prerequisites before writes and use explicit separate run directories. Preserve the executed source when making that improvement.

Evidence: [training source](../../experiments/moe_balance_training.py), [raw report](../../results/moe_balance_training.json), [results](../topics/ffn/moe-balance-training-results.md). No post-hoc threshold change or claim that the original measurements are invalid follows.

## Maintenance findings

Current top-level counts: 191 experiment Python files, 288 result JSON files, no local .git directory, and AGENTS.md is 19,057 bytes. README has four latest-style announcements. These differ from the supplied snapshot but support the overhead concern. A narrow search finds 33 scripts containing lr=.001 syntax and 115 occurrences of All immediately followed by a digit or CPUfloat followed by a digit in docs. These are explicitly defined counts, not a reproduction of the supplied broader 217-formatting count. No standard scheduler-class name was found; custom schedules could exist, so this is not a schedule-use census. Repeated learning rates alone do not identify a failure's cause.

Recommended cleanup: concise rules in AGENTS.md, study history in assessments, one current README entry, version control plus run manifests. Keep historical evidence; do not mass-rewrite reports or migrate every audit before resuming science. The folder name is a low-priority naming mismatch, not a scientific problem.

## Recommended scientific focus

**Why does the tested state-tracking model succeed on recent-reset groups while failing long parity-dependent groups, even during training?**

First review existing checkpoint behavior against explicit hypotheses: initial-state neglect, sensitivity to recent rather than distant FLIPs, or failures that depend on FLIP count. A useful intervention must distinguish these hypotheses: flipping the initial bit, toggling one KEEP/FLIP at controlled positions, and a two-toggle control with unchanged final parity. Measure probability/margin changes as well as accuracy, across all seeds and matched groups. Mere attention plots cannot identify the cause. These are proposed diagnostics, not executed results or a claim that the hypotheses exhaust the causes.

[Liu et al., Transformers Learn Shortcuts to Automata (ICLR 2023), author explanation](https://clarabing.github.io/shortcut_automata/) is relevant: it distinguishes sequential state updates from parallel automaton computation and studies learned shortcuts. Its state-sequence supervision and constructions must be compared with our final-answer setup before treating it as a recipe. An existence construction does not guarantee our optimizer learns it. Here, shortcut means fewer sequential computation steps, not necessarily a spurious correlation.

If saved models cannot distinguish candidate explanations, use a bounded training-only recipe-development phase. Declare the small candidate grid, equal per-arm search allowance and fitting criterion before running. Record all attempts, lock the recipe, then use fresh confirmation pools and seeds for architectural comparisons. Training fit is an eligibility check, not evidence of algorithm learning. Keep original failed recipes as historical evidence.

**Decision:** prioritize this state-tracking question over new topic coverage. No new diagnostic, sweep, Git initialization or document migration has been executed by this review. A closed budget stays closed; a distinct explanatory follow-up is legitimate when explicitly selected.
