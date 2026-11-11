# Compact-to-shifted activation interventions

Fixed before execution. Use all five frozen deep models and every six-pair-derived crop from the 512 subset-confirmation episodes (5,120 crops/model). Choose global shift +4 because the prior diagnostic showed a large error there. This choice and the data are exploratory, not untouched confirmation. No training.

For block l, compute a=Attention(LN1(x)), u=x+a, f=FFN(LN2(u)), y=u+f. Record x,a,f from the compact run. In a shifted run independently replace one of: residual operand x in u (while computing a from the original live x), attention output a, or FFN output f. Test each branch in each of two blocks, plus both attention branches, both FFN branches and both residual operands. Replace all token positions, not just the query. These are donor-activation tests, not ordinary forward passes on a new natural input.

Controls: unmodified compact/shifted explicit reconstruction must match the original model within 1e-10; replacing every branch from compact must reconstruct compact logits; replacing every branch with shifted donors must reconstruct shifted logits. Check finite outputs, unchanged checkpoint hashes, original accuracy/loss reproduction within 1e-12 and paired rescue/harm arithmetic.

Report all conditions per seed: accuracy, loss, rescued/harmed versus shift+4. Sufficiency criterion per condition: every seed >=90% accuracy. Do not infer necessity, uniqueness, mediation fractions or additive causes. Replacing a residual operand also changes subsequent FFN input; replacing FFN output bypasses its live computation. Interventions mix activation distributions. Results can locate rescuing paths but need reverse and donor controls before strong mechanism claims.
