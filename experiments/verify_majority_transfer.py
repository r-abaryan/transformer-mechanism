"""Pretraining function/gradient, causal and ground-truth controls."""
import itertools
import json
import torch
from torch.nn import functional as F
from majority_transfer import ROOT, SEEDS, model_new, forward, provenance, hash_state
from audit_majority_transfer import independent, label


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    ids=torch.tensor([[0,1,0,1,0,1,1,1,0,2],[1,0,0,0,0,1,1,0,0,2]])
    targets=torch.tensor([1,0]); records=[]
    for seed in SEEDS:
        torch.manual_seed(seed); model=model_new(); a=forward(model,ids); b=independent(model,ids)
        error=(a-b).abs().max().item(); assert error<=1e-10
        parameters=list(model.parameters())
        ga=torch.autograd.grad(F.cross_entropy(a[:,-1],targets),parameters)
        gb=torch.autograd.grad(F.cross_entropy(b[:,-1],targets),parameters)
        gradient_error=max((x-y).abs().max().item() for x,y in zip(ga,gb)); assert gradient_error<=1e-10
        assert all(torch.isfinite(g).all() for g in ga) and max(g.abs().max().item() for g in ga)>1e-7
        changed=ids.clone(); changed[:,7]=1-changed[:,7]
        with torch.no_grad(): c=forward(model,changed)
        prefix=(a[:,:7]-c[:,:7]).abs().max().item(); suffix=(a[:,7:]-c[:,7:]).abs().max().item()
        assert prefix<=1e-10 and suffix>1e-6
        norms=[v for block in model.blocks for v in (block.norm1,block.norm2)]+[model.final_norm]
        assert all(n.eps==1e-5 and n.bias is not None for n in norms)
        records.append(dict(seed=seed,initial_sha256=hash_state(model.state_dict()),max_logit_error=error,
                            max_gradient_error=gradient_error,causal_prefix_error=prefix,changed_suffix_error=suffix))
    checked=0; wrong_last=0; wrong_flipped=0
    for n in (9,11):
        for bits in itertools.product((0,1),repeat=n):
            expected=int(2*sum(bits)>n); assert expected==label(bits)
            wrong_last+=int(bits[-1]!=expected); wrong_flipped+=int(1-expected!=expected); checked+=1
    assert checked==2560 and wrong_last>0 and wrong_flipped==checked
    result=dict(passed=True,models=records,labels_checked=checked,last_bit_control_errors=wrong_last,
                flipped_label_control_errors=wrong_flipped,provenance=provenance())
    path=ROOT/'results/majority_transfer_verification.json'; assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(dict(passed=True,models=len(records),labels_checked=checked,
                 max_gradient_error=max(r['max_gradient_error'] for r in records),last_bit_control_errors=wrong_last)))


if __name__=='__main__': main()
