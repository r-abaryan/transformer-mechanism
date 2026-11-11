# Frozen-model investigation protocol

Question: which attention computations contribute to seed 47's length-ten errors? This is exploratory analysis of already-observed test data, not independent confirmation.

Keep every checkpoint and the architecture fixed. Regenerate the original length-six test and length-ten examples with the recorded seeds. Inspect all five seeds. Before interpreting interventions, reconstruct the final-query computation explicitly and require logits to match the saved model within 1e-10.

Measure per-head attention to position zero, attention to the query itself, entropy, and the norm of each position's projected value contribution. Split these measurements by original correct/incorrect predictions; empty groups have no mean.

Interventions at the final query only: zero each head's output; force each head to read the first symbol; force it to read the last symbol as a positional control; replace each head's attention with a uniform distribution; remove the first position's value contribution and compare with removing the last position's contribution. Do not renormalize value-removal weights. These controls match the operation, not perturbation magnitude. Report paired accuracy, loss, corrected errors and broken correct predictions on the same examples, for every seed.

Additional diagnostic: replace only the length-ten query's positional vector with the training query's positional vector (position six), preserving all ten symbol positions. This intervention tests sensitivity to the changed query position, not whether positional encoding alone explains failures.

Forcing attention or altering positions can create states outside training. A rescue shows that an intervention can change this model's predictions; it does not prove the natural computation uses an interpretable rule. No architecture edits, retraining, checkpoint selection or general claims about transformers follow from these tests.
