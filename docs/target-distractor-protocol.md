# Target versus distractor value-position K

Fixed before execution. Five frozen models and the same 5,120 inspected six-derived crops per model. No training. Final Q is compact in every condition, both heads. Attention V and residual/FFN computation stay at the recipient state. This isolates changes to value-position K conditional on that Q.

Forward recipient is shifted +4. Replace K from the matched compact run at none, target-value slot, other two value slots, or all three value slots. Reverse recipient is compact. Replace the same subsets with shifted K. Target means the queried pair's value position, not every occurrence of the answer token. Other pairs can have the same value symbol. Do not assume they are always harmful competitors.

Report accuracy, loss, paired rescue/harm relative to each direction's no-K-replacement baseline. Primary forward sufficiency criterion per subset: every seed >=90%. Reverse disruption criterion per nonempty subset: every seed loses >=10 percentage points. No significance or necessity claim.

Measure final-query attention per head: mean target-slot probability, sum over other value slots, and pre-softmax target score minus maximum other-value score. Compare on identical crops. These describe routing, not proof of complete causal mediation. Verify softmax rows, finite outputs, checkpoint hashes and forward none/target/all reproduction from head-token results (<1e-10). Reverse none must reproduce compact reference. Save protocol/source/input hashes and all seeds. Reused crops and mixed activations remain limitations.
