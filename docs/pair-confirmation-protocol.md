# Frozen intervention confirmation across shifts

Fixed before execution. Generate 512 six-pair parents with seed 20261201, excluding canonical mapping/query identities in original, depth-confirmation and subset-confirmation six-pair sets. Enumerate all ten query-preserving three-pair crops per parent. Save exact parents and report cropped-identity overlap with earlier diagnostic crops and confirmation training. Fresh parents do not guarantee unseen model inputs.

Freeze original block-1 forcing: rows 3/5 in both heads read paired keys; row 1 unchanged. Test shifts 0,1,2,4,6,8 with original/self/earlier-nonpaired-key controls. Full forward execution, no compact donors or training; same five checkpoints. No tuning or new forcing sites.

Criteria: every seed >=90% paired-forcing accuracy at each positive shift, plus conjunction across shifts; mean paired advantage >=10 percentage points against each source control at each positive shift; compact loss <=1 point in every seed. Report each separately. Also report losses and paired rescue/harm, all seeds and parent-level every-crop-correct accuracy. Do not treat ten crops as independent parent trials.

Checks: parent exclusion, crop labels, causal masks/row sums, explicit original attention/forward versus unchanged implementation (<1e-10), finite outputs and checkpoint hashes. Save protocol, source and split hashes. These are shifted eight-token crops, not full long-sequence tests; additional shifts were previously examined without this intervention. Local confirmation is not independent replication or a universal repair claim.
