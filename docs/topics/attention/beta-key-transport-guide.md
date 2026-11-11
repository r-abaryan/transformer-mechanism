# Beta and key geometry do different jobs

Beta controls how strongly a write changes memory. Key geometry describes the directions of normalized keys and their overlaps. Both affect which old associations survive and which other values reach a query.

Our toy reader uses tied query/write keys, scalar beta and an affine classifier. For zero-start memory:

$$S_t=(I-\beta k_tk_t^T)S_{t-1}+\beta k_tv_t^T,\qquad r_q=q^TS_T=\sum_t a_{qt}v_t^T.$$

For a target written at position p,

$$a_{qp}=\beta q^TA_T\cdots A_{p+1}k_p,\qquad R_{qp}=\frac{a_{qp}}{\beta\|k_p\|^2},\quad A_t=I-\beta k_tk_t^T.$$

R is the target's final coefficient relative to its immediate coefficient. It measures signed sensitivity, not information retained. Lower beta also weakens the initial write, so a higher R need not mean a larger final target signal or better predictions.

## How we separated them

Take the beta and key table learned at step 3000, then those learned at 12000. Test all four combinations. Hold values and classifier fixed to one stage; repeat with them fixed to the other stage. This gives eight conditions per seed. No model is retrained.

| | Early beta | Late beta |
|---|---|---|
| Early keys | f00 | f01 |
| Late keys | f10 | f11 |

At fixed early keys, beta's effect is f01 - f00. At fixed late keys, it is f11 - f10. Their difference, f11 - f10 - f01 + f00, is the interaction. A nonzero interaction means the effect depends on the other factor. It prevents a unique additive percentage attribution.

Changing the key table changes queries and writes together. A common orthogonal rotation of all tied keys leaves zero-start reads unchanged: S transforms with the keys and q cancels the rotation. The meaningful change is their relative geometry, not their coordinates relative to values or the classifier.

## What happened

Lower beta substantially improves mean target coefficient retention. Later key geometry slightly lowers that mean, but helps accuracy at lower beta. Thus preserving the direct target term and controlling competing content are different requirements.

Accuracy and CE can disagree. Accuracy asks whether the correct class wins. CE also penalizes low probability on the correct class. With earlier values/head, the swaps gain accuracy while lower beta raises CE. With endpoint values/head, the native combination remains best among these four conditions. Rescuing selected errors does not guarantee net benefit.

The [results](beta-key-transport-results.md) isolate these computational effects on opened training histories. They do not establish why optimization learned them, a usable repair, or held-out transfer.

Sources inspected: [Yang et al. (2024), sections 2.2 and 3.2](https://arxiv.org/html/2406.06484v2) for published delta recurrence/unrolling; [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html) for the loss definition. Factorial contrasts and rotation invariance are local derivations. The paper uses input-dependent beta and a fuller architecture; it does not predict these scores.
