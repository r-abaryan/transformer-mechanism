# More training or a different mechanism?

The [feedback experiment](query-feedback-training-results.md) produced real fitting gains but left errors. Its full training loss was still falling. The next question is whether more training with exactly that recipe is sufficient. [Frozen protocol](../../protocols/query-feedback-continuation.md) | [Measured results](query-feedback-continuation-results.md).

## What continues

| Kept | Why it matters |
|---|---|
| All model weights and the same architecture | No extra representation capacity is introduced |
| Adam moments and update count | The optimizer continues its learned update history |
| History and potential-query RNG streams | Training follows the next draws rather than restarting exposure |
| Batch, labels, objective, precision and learning rate | No simultaneous recipe change is selected |

More updates still mean more examples and answer targets. This study tests a larger budget under one recipe; it cannot separate update count from additional exposure or individual parameter changes. Only exhaustive feedback continues. It is selected by training CE, not its mixed earlier test scores.

## Why restoring the optimizer matters

Adam stores gradient averages. For a scalar gradient g at optimizer step t,

$$\mu_t=.9\mu_{t-1}+.1g_t,\qquad\nu_t=.999\nu_{t-1}+.001g_t^2.$$

Its update uses both moments and their step-dependent bias corrections. Starting new moments at step 3000 changes the update even if the weights are identical. The [earlier learning guide](retention-learning-replay-guide.md) explains the full update. We check the restored model/optimizer exactly and verify a short resumed run against uninterrupted training.

Sources inspected: [PyTorch 2.11 Adam](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html), moment and bias-correction equations; [official checkpoint tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training), model and optimizer restoration. RNG reconstruction is our local implementation, checked against independent draw replay.

## What the stopping rule means

Each seed may add up to 9000 updates. Check every full training load after each 1000. Stop at the first scheduled check with >=99% accuracy in every relevant training group, or at the budget cap. Validation and test are measured afterward and never select the stopping point.

A complete map requires every key in one history to be answered correctly. One wrong answer makes that map fail. Thus 99% individual-answer accuracy does not imply 99% complete maps. Both measurements remain visible.

| Outcome | Supported interpretation |
|---|---|
| Training reaches the fit rule | More training with this unchanged recipe is sufficient for that rule |
| Training fits but held-out scores remain weak | A generalization gap remains under these conditions |
| Fit is still inadequate at the cap | This larger budget remains insufficient; a hard capacity ceiling is unproved |
| CE is falling at the final check | Loss continues to improve across measured intervals; convergence is unestablished |

Earlier validation/test pools and seeds are reused. Results are development evidence, not fresh confirmation or longer-input transfer. Beta, geometry and readout learn jointly, so a changed beta does not by itself explain the outcome. [Yang et al. 2024, section 2.2](https://arxiv.org/html/2406.06484v2) defines the delta correction/read rule; our shared-beta 615-parameter reader is narrower than full DeltaNet. No cited source predicts these local scores.

## Why accuracy can improve while loss worsens

Accuracy checks the winning class. Cross-entropy checks the probability of the correct class:

$$\ell=-\log p(\text{correct answer}).$$

A still-wrong answer whose correct-class probability falls from .1 to .01 incurs loss 4.61 instead of 2.30. Prediction accuracy need not change. A few new correct answers can therefore coexist with higher average loss. This is a mathematical example, not a measured account of our individual errors. At load 32, the continuation does show higher test accuracy alongside worse CE; identifying which answers and learned parameters cause that pattern needs a separate saved-score diagnostic.
