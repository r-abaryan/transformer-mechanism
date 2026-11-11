# Changing confidence while keeping memory fixed

The memory reader produces a vector r. A classifier turns it into six scores: z=Wr+b. Softmax turns scores into probabilities. Accuracy uses only the largest score; CE uses the probability assigned to the correct answer.

## What we changed

Multiply every class-score difference by the same positive number s. If s<1, probabilities become less sharp. The largest score stays largest, so no incorrect answer becomes correct. Some incorrect answers become less costly, while correct answers become less certain.

| Component | Role | In this control |
|---|---|---|
| Keys, values and beta | Write and read memory | Fixed at step 12000 |
| Readout W and bias b | Turn the read vector into class scores | Centered differences scaled together |
| Scale s | Controls probability sharpness | Derived once from saved weights |

For A=[W\|b], subtract the average class row to obtain C. That common component adds the same number to every score and disappears in softmax. Our local definition is

$$s=\frac{\|C_{3000}\|_F}{\|C_{12000}\|_F},\qquad A'=\operatorname{mean}_{classes}(A_{12000})+sC_{12000}.$$

Therefore $z'=\bar z+s(z-\bar z)$ and $\operatorname{softmax}(z')=\operatorname{softmax}(sz)$. Scaling W alone would leave b unchanged and can change the winner. The augmented norm includes the intercept and depends on the representation coordinates; it is one declared scale definition, not a unique measure of confidence.

## Why loss can improve without accuracy improving

$$CE_i(s)=-s z_{iy}+\log\sum_c e^{s z_{ic}}.$$

Softening can raise the true-class probability of an incorrect answer. It also lowers confidence in correct answers and can raise their CE. The net effect must be measured. An incorrect winner does not guarantee that its CE improves under every softening.

The [results](readout-scale-control-results.md) show much lower load-32 held-out CE, unchanged answers and worse training CE. This separates probability sharpness from answer selection. It does not identify a learned repair or the historical cause of the remaining mistakes.

[Guo et al. 2017, section 4.2, equation 9](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) uses positive temperature T: softmax(z/T), with unchanged argmax. Our s corresponds to T=1/s. Their temperature is selected using validation loss; ours comes only from weight norms. We inspected that equation and rank property, not reproduced their calibration experiments. No ECE or optimal temperature was measured. Class centering and norm selection above are local derivations.
