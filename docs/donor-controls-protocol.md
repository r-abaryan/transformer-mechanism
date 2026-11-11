# Second-attention donor controls

Fixed before execution. Reuse five frozen models and 5,120 inspected six-derived crops. No training. Target only block 2 attention output at all token positions. Reuse the verified explicit forward.

Conditions: compact/shifted baselines; same-example compact donor into shifted recipient; same-example shifted donor into compact recipient (reverse); unrelated compact donors into shifted recipients with same answer and with different answer. Choose each unrelated donor deterministically by scanning cyclically from the next parent episode. Require a different parent, query key and sorted three-pair mapping. Save donor indices. Answer-matched selection is a label-informed control, not a practical algorithm. Donors may repeat; no independence assumption.

Report recipient accuracy/loss, donor-answer agreement, paired rescue/harm relative to the recipient's unpatched baseline, and all seeds. Predeclared descriptive checks: reverse intervention reduces compact accuracy by >=10 points in every seed; same-example rescue exceeds same-answer unrelated accuracy by >=10 points on average; different-answer donor agreement >=90% for every seed. These are diagnostic thresholds, not significance tests. Report failures without tuning.

Verify baseline and same-example rescue reproduction within 1e-12, same-state attention donor shams within 1e-10, label/donor restrictions, finite outputs, checkpoint hashes and paired arithmetic. Mixed activations are artificial. Reverse harm does not prove necessity; unrelated-donor effects do not prove that the attention output encodes only the answer. This experiment tests specificity before Q/K/value decomposition.
