# What the eight attention tests mean

These are small mathematical checks, not training experiments. Proofs depend on assumptions; passing examples check our code at selected inputs. See [measurements](results.md) and [implementation](../../../experiments/attention_spikes.py).

## Notation

One head has n tokens. Q and K contain query and key vectors of width k; V contains value vectors of width v. A query compares against keys to decide how to combine values:

\[
S=QK^T/\sqrt{k}+M,\qquad A=\operatorname{softmax}(S),\qquad Y=AV.
\]

S contains scores, M controls visibility, A contains attention weights, and Y contains outputs. Softmax acts on each row: a_j = exp(s_j) / sum_r exp(s_r). Its weights are nonnegative and sum to one. The superscript T means transpose.

## 1. Attention scaling (S01)

**What and why:** Divide each query-key dot product by sqrt(k). This controls score spread as vector width grows, helping avoid extremely concentrated softmax weights and small score gradients.

**Math:** If all q_l and k_l are independent, zero-mean and unit-variance:

\[
\operatorname{Var}(q\cdot k)
=\sum_{l=1}^{k}\operatorname{Var}(q_l k_l)=k,
\qquad \operatorname{Var}\left(\frac{q\cdot k}{\sqrt{k}}\right)=1.
\]

Variance measures spread around the mean. Dividing by a constant divides variance by its square.

**Evidence and limit:** Sample variances were 0.9659-1.0335 after scaling. This supports the calculation for our independent random inputs. Learned vectors may be correlated or have different variances; optimal scaling is untested.

## 2. Causal masking (S02)

**What and why:** Prevent a token position from reading future positions. Otherwise next-token training could expose the answer through the input.

**Math and operation:** Before softmax, add:

\[
M_{ij}=\begin{cases}0&j\leq i\\-\infty&j>i.\end{cases}
\qquad A_{ij}=0\text{ for }j>i.
\]

Since exp(-infinity)=0, forbidden values contribute nothing to Y_i=sum_j A_ij V_j. At least one key must remain visible. With input token t_i, the target is t_(i+1), so attending to the current input position is allowed.

**Evidence and limit:** Future edits changed earlier outputs by zero; removing the mask made them change. This checks one head. Correct target shifting and absence of other information leaks still need whole-model tests.

## 3. Permutation equivariance (S03)

**What and why:** A permutation reorders tokens. Without an order signal, unmasked attention follows that reordering rather than assigning meaning to absolute positions. This explains why sequence order needs special treatment.

**Math:** Let P reorder rows and F be position-free self-attention with shared projections:

\[
(PQ)(PK)^T=P(QK^T)P^T,\qquad
F(PX)=PF(X).
\]

Both score axes reorder; row softmax and value aggregation preserve the correspondence. This is **equivariance**: outputs reorder too. Invariance would mean outputs stayed unchanged.

**Evidence and limit:** Reordering error was zero; keeping positional signals fixed broke the equality. A fixed causal mask also introduces order structure, so the unmasked argument cannot be applied directly to causal models.

## 4. Softmax gradients (S04)

**What and why:** A gradient measures how an output changes when an input changes. Training needs these derivatives to propagate error back through attention scores.

**Math:** For a=softmax(s), the Jacobian (matrix of derivatives) is:

\[
\frac{\partial a_i}{\partial s_j}=a_i(\delta_{ij}-a_j),
\qquad \delta_{ij}=1\text{ if }i=j,\;0\text{ otherwise}.
\]

Increasing a score raises its own weight and lowers others. If one weight approaches one, these derivatives approach zero. We checked them numerically using [a_i(s+epsilon e_j)-a_i(s-epsilon e_j)]/(2 epsilon), where e_j changes only score j.

**Evidence and limit:** Maximum derivative error was 3.924e-11; a deliberately wrong formula disagreed. This validates the tested derivative, not gradient flow or training stability throughout a transformer.

## 5. Different attention, same output (S05)

**What and why:** Attention outputs are weighted sums. Different weights can produce the same sum, which limits what a heatmap alone can explain.

**Math:** For scalar values V=(0,1,2):

\[
(0.2,0.6,0.2)V^T=1,\qquad
(0.4,0.2,0.4)V^T=1.
\]

Both rows are valid softmax distributions: softmax(log(a))=a for positive weights summing to one.

**Evidence and limit:** Both outputs were 1. This proves that an output need not uniquely determine its attention distribution. It does not show how common this ambiguity is in trained models or that attention is always uninformative.

## 6. Streaming softmax (S06)

**What and why:** Compute a weighted result incrementally instead of retaining every attention weight. Subtracting a running maximum prevents exponential overflow while preserving the result.

**Math:** For scores seen so far, keep maximum m, denominator l=sum_j exp(s_j-m), and vector numerator u=sum_j exp(s_j-m)v_j. On receiving score s and value vector v:

\[
m'=\max(m,s),\quad c=e^{m-m'},\quad w=e^{s-m'},
\]
\[
l'=cl+w,\qquad u'=cu+wv,\qquad y=u/l\text{ after the final update}.
\]

Initialize m=-infinity, l=0, u=0. The factor c rescales earlier contributions when the maximum changes; numerator and denominator retain the same common scale.

**Evidence and limit:** Dense and streaming results differed by at most 5.552e-17. Our test supplies already-stored scores and checks arithmetic only. It neither demonstrates a memory-saving kernel nor measures FlashAttention performance; pairwise score computation remains.

## 7. Matrix reassociation (S07)

**What and why:** Matrix multiplication can be regrouped. The choice changes intermediate storage and computational cost, but a nonlinear operation cannot simply be moved or removed.

**Math:** For Q,K of shape n x k and V of shape n x v:

\[
(QK^T)V=Q(K^TV).
\]

The left intermediate is n x n; the right is k x v. However, generally:

\[
\operatorname{softmax}(QK^T)V\ne Q(K^TV).
\]

Softmax exponentiates and normalizes scores, changing their values. The same obstruction applies with score scaling.

**Evidence and limit:** Raw products agreed; inserting softmax produced a difference of 4.3576 from the reassociated raw product. This refutes deleting softmax as an equivalent shortcut, not correctly defined linear-attention alternatives.

## 8. Dense storage (S08)

**What and why:** Explicit dense attention stores one score for each query-key pair. Long sequences therefore make the score matrix expensive.

**Math:** For batch size B, h heads and b bytes per element, one dense score tensor requires:

\[
\text{elements}=Bhn^2,\qquad \text{bytes}=bBhn^2.
\]

Doubling n multiplies this storage by four. This excludes other tensors, gradients and allocator overhead.

**Evidence and limit:** Lengths 16, 32 and 64 produced 256, 1,024 and 4,096 elements for one head. This is a counting check, not a memory benchmark. Implementations need not materialize the entire matrix.

## What comes next

First explain a complete numerical transformer example, audit foundational paper claims, independently verify a tiny trainable implementation, then train controlled tasks and investigate learned behavior. Architecture proposals wait until that baseline evidence is reliable.

For broader context and source reading status, see the [architecture notes](architecture.md), [source register](../../reference/sources.md) and [interpretability discussion](interpretability.md).
