# Why repeating a history can change a memory read

`A:1, B:4` and `A:1, B:4, A:1, B:4` contain the same associations. The second history makes the reader process each write twice. Our diagnostic repeats whole histories in their original order, with unchanged weights and answers.

## Softmax: duplicated evidence cancels

For fixed query, keys and values, let s_t be the attention score:

\[
r=\frac{\sum_t e^{s_t}v_t}{\sum_t e^{s_t}}.
\]

Repeating **every** event f times multiplies both sums by f. The read stays identical. Our reader has no recency or positional operation, so its logits must also stay identical, apart from rounding. Duplicating only selected events would change their relative weight.

This follows from [scaled dot-product attention](https://arxiv.org/html/1706.03762v7), section 3.2.1. It is not an invariance claim for all Transformer models.

## Delta: every pass updates the shared matrix

The [delta correction rule](https://arxiv.org/html/2406.06484v2), section 2.2, is:

\[
S_t=(I-\beta k_tk_t^{\mathsf T})S_{t-1}+\beta k_tv_t^{\mathsf T}.
\]

Here k_t is a unit key, v_t a value vector, and beta the learned scalar write strength. A whole ordered block is an affine transformation `S -> A*S+B`. A contains the ordered product of key transitions; B contains their transformed writes. Starting from zero:

\[
S_1=B,\quad S_2=AB+B,\quad S_4=A^3B+A^2B+AB+B.
\]

Another pass can correct some reads and disturb others. These equations permit change; they do not guarantee better answers or a correct limiting state. The block expansion is our local derivation, not a reproduction of the paper's training results.

## Reading the evidence

A **rescue** turns an incorrect answer correct. A **harm** turns a correct answer incorrect. Net accuracy can hide both. Repetition adds processing and length, but no new associations. [Measured results](history-repetition-results.md) distinguish these effects from a trained repair.
