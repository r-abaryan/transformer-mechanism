# Better accuracy, worse probabilities

The continued reader answers slightly more test queries correctly, yet its average loss rises. The [saved-score diagnostic](feedback-loss-decomposition-results.md) locates that change without training again.

## Two different measurements

Accuracy checks whether the largest logit belongs to the true class. Cross-entropy checks how much probability the model assigns to that class:

$$p_y=\frac{e^{z_y}}{\sum_c e^{z_c}},\qquad\ell=-\log p_y.$$

If a still-wrong answer's true-class probability falls from .1 to .01, its loss rises from 2.30 to 4.61. Accuracy stays zero. This mathematical example does not assume that every actual error behaves that way. [PyTorch 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html) defines the class-index loss used here.

## Follow the same answers

| Group | Earlier / later prediction | What its loss change means |
|---|---|---|
| Correct at both | Correct / correct | Probability quality can improve or worsen without an accuracy change |
| Wrong at both | Wrong / wrong | Persistent errors can receive less probability on the correct class |
| Rescued | Wrong / correct | These contribute the accuracy gains |
| Newly wrong | Correct / wrong | These contribute the accuracy harms |

Let N be the total answer count and G one group. Its contribution is

$$C_G=\frac1N\sum_{i\in G}(\ell_i^{later}-\ell_i^{earlier}),\qquad\Delta\mathrm{CE}=\sum_G C_G.$$

A positive contribution raises overall loss. A negative one lowers it. The group mean alone omits how many answers belong to that group. Signed contributions can offset each other, so a positive contribution can exceed 100% of the net increase.

## Correct-answer probability is not winning confidence

Winning confidence is $\max_c p_c$. On a mistake, that probability belongs to a wrong class. We measure both it and p_y. They need not move oppositely: probability can move away from other classes toward both the winner and the true class.

Mean CE is the mean of -log p_y, not -log of the mean p_y. Very small true-class probabilities can raise CE sharply even when mean probability changes modestly.

The two test seeds show higher mean winning confidence in their fixed persistent-error cohorts, but some individual errors become less confident. Some also switch to another wrong class. Report those exceptions rather than treating all persistent errors as unchanged predictions.

[Guo et al. 2017, section 2](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) distinguishes confidence, calibration and NLL. Calibration asks whether a stated confidence matches observed correctness frequency across predictions. Our wrong-answer cohort analysis does not measure a calibration curve or ECE. It locates score deterioration, not its parameter cause.

The [results](feedback-loss-decomposition-results.md) contain actual counts, probabilities, contributions and traceable examples. Endpoint comparison does not establish a monotonic training trajectory, explain why the optimizer learned these parameters or repair retention. The four-way decomposition is local algebra, not a published prediction of these scores.
