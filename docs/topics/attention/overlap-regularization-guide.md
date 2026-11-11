# Can learning reduce interference without a bigger matrix?

Our supplied orthogonal keys preserved every queried association, but doubled the state. This comparison keeps the original 16x8 state and changes the training objective.

## The penalty

Normalize all 32 learned keys u_i. Measure:

\[
R=\frac{1}{32\cdot31}\sum_{i\ne j}(u_i^{\mathsf T}u_j)^2,
\qquad L=\mathrm{CE}+\lambda R.
\]

Squaring penalizes large positive and negative overlaps. Excluding the diagonal avoids charging each unit key for its unavoidable self-overlap. Lambda sets the penalty's weight; it does not add model parameters.

This is a normalized off-diagonal form of the **frame potential**, described by [Casazza, Fickus and Mixon, 2010, introduction](https://arxiv.org/pdf/1009.5562). Our joint Adam/classification objective differs from their frame-only method and does not inherit its convergence guarantees.

## Why the minimum is not zero

Let U be the 32x16 matrix of unit keys. Its squared Gram norm obeys:

\[
\|UU^{\mathsf T}\|_F^2=\|U^{\mathsf T}U\|_F^2
\ge\frac{32^2}{16}.
\]

Subtract the 32 diagonal terms and divide by 992: **R >= 1/31**. This follows from Cauchy-Schwarz on the eigenvalues, consistent with the cited frame-potential bound. It means zero total cross-overlap is impossible at this width. It does not prove a classification ceiling.

Even minimum R does not separate every pair. Two copies of the 16 basis vectors attain 1/31 while containing identical key pairs. Our preflight checks this fixture. Low average overlap is therefore not sufficient evidence of successful storage.

## Why lower R may still fail

The [delta correction rule](https://arxiv.org/html/2406.06484v2), section 2.2, gives a query-read change:

\[
\Delta r=\beta(q^{\mathsf T}k)(v-S^{\mathsf T}k).
\]

Overlap is only one factor. Correction residuals, write strength and the classifier also affect the answer. Average squared overlap can improve while individual reads remain wrong.

The experiment therefore reports task accuracy/CE and complete maps, alongside R. All-query evaluation asks every present key from one final state. Training still supervises one selected query per example; those additional evaluation queries were not extra training targets.

[Measured comparison](overlap-regularization-results.md): selected penalty improves mean all-query test accuracy from 62.19% to 65.79%, but no complete maps pass. Near-minimum R is not successful retention. [Frozen comparison](../../protocols/overlap-regularization.md) | [Earlier supplied-geometry evidence](key-geometry-retention-results.md).
