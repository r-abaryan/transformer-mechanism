# Claim and evidence ledger

Status vocabulary: **derived** (argument under stated assumptions), **locally tested** (bounded execution), **reported** (authors' results), **open** (not established here). Derived and locally tested are independent labels. See [raw measurements](../../results/attention_spikes.json) and [protocol](../research-plan.md).

| ID | Bounded claim | Evidence / experiment | What would challenge it? | Status |
|---|---|---|---|---|
| C01 | Independent standardized q,k have scaled dot-product variance 1 | Architecture derivation; S01, 3 seeds, 3 widths | Incorrect derivation or systematic deviation under those assumptions | Derived; execution status in results |
| C02 | Causal head outputs at i ignore later input rows | Visibility definition; S02 perturbation with unmasked positive control | Earlier outputs change under future-only edits | Derived; execution status in results |
| C03 | Position-free unmasked attention is permutation equivariant | Algebra; S03 with positional negative control | Nonzero error beyond numerical tolerance under the premise | Derived; execution status in results |
| C04 | Softmax Jacobian equals diag(a)-aa^T | Differentiation; S04 finite differences and wrong-Jacobian control | Gradient disagreement above tolerance | Derived; execution status in results |
| C05 | Different attention distributions can give the same value aggregate | Explicit V=(0,1,2) construction; S05 | Arithmetic failure in the witness | Derived; execution status in results |
| C06 | Online stable softmax accumulation matches dense attention | Running numerator/denominator identity; S06 | Output mismatch, including extreme scores | Derived; execution status in results |
| C07 | Associating raw dot products is valid, but does not preserve softmax attention | Matrix associativity and counterexample; S07 | Claimed counterexample has zero difference | Derived; execution status in results |
| C08 | Direct score storage grows quadratically in n | n x n element count; S08 | Allocation count not n² | Derived; count checked, not a runtime benchmark |
| C09 | Pre-norm improves optimization in the conditions studied by P04 | [P04](https://arxiv.org/abs/2002.04745); training replication pending | Controlled matched-training failures | Reported, not locally tested |
| C10 | Gated memory can improve particular retrieval tasks | P10 and P13 in source register | Matched tasks, tuning, state size and budgets remove benefit | Reported, not locally tested |
| C11 | A hybrid is universally better than a transformer | No supporting universal evidence; P15 motivates task breakdown | Any well-controlled losing case refutes universality | Open; do not assert |
| C12 | Our proposed modification is novel and superior | No modification validated yet | Prior art or lack of held-out advantage | Open; do not assert |
| C13 | Explicit and library block outputs/gradients agree at tested settings | [Three-seed verification](../topics/foundations/decoder-verification.md), numerical derivatives | Error exceeding recorded tolerances | Locally tested |
| C14 | All five runs meet 95% accuracy on held-out length-six first-token recall | [Fixed protocol and results](../topics/retrieval/first-token-results.md) | Any recorded seed below threshold | Locally observed: passed |
| C15 | All five runs meet 95% accuracy on length-ten first-token recall | Same experiment, separate length criterion | Any recorded seed below threshold | Locally observed: failed; weakest seed 82.03% |
| C16 | Skip correction reliably rescues all tested models across lengths | [Fixed confirmation](../topics/retrieval/skip-confirmation-results.md) | Any seed/length misses 95% | Locally observed: failed |
| C17 | The unchanged one-block recipe learns reliable associative recall | [Training and untouched confirmation](../topics/retrieval/associative-recall-results.md) | Any seed misses predefined 90% criterion | Locally observed: failed; not an impossibility claim |
| C18 | Tested Adam recipe resolves associative learning plateau | [Matched recipe comparison](../topics/training/optimizer-comparison-results.md) | Fails training/validation thresholds or relative gain | Locally observed: all three criteria failed; other recipes untested |
| C19 | Two blocks reach 90% train/validation and beat the wider one-block control by 10 points on average under the fixed recipe | [Depth comparison](../topics/training/depth-comparison-results.md) | Any deep seed misses 90%, or mean advantage is below 10 points | Locally observed: passed; reused validation, unmatched compute |
| C20 | Fixed depth advantage replicates with five new seeds and regenerated splits | [Replication](../topics/training/depth-confirmation-results.md) | Any deep seed misses 90% train/test, or mean test gain below 10 points | Locally observed: passed; internal holdout, prior-data overlap disclosed |
| C21 | Three-pair-trained deep models all reach 90% at four and six pairs | [Separate transfer criteria](../topics/training/depth-confirmation-results.md) | Any seed below 90% at each length | Locally observed: failed at both lengths |

Never promote C09-C12 because C01-C08 pass. Component mathematics cannot establish end-to-end model quality.
