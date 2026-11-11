"""Verify centering-only change and independent functions/gradients pretraining."""
import json
import torch
from torch.nn import functional as F
from rmsnorm_training import ROOT,SEEDS,model_new,forward,parameter_hash,provenance
from audit_rmsnorm_training import independent
from mqa_training import model_new as original_model
from position_distractors import digest


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    path=ROOT/'results/rmsnorm_training_verification.json';assert not path.exists()
    ids=torch.tensor([[0,12,1,13,0,14,18,0],[2,15,3,16,2,17,18,3]]);labels=torch.tensor([14,17]);records=[]
    for seed in SEEDS:
        hashes=[]
        for group in ('ln','rms'):
            torch.manual_seed(seed);m=model_new(group);hashes.append(parameter_hash(m))
            assert sum(p.numel() for p in m.parameters())==4464
            norms=[b.norm1 for b in m.blocks]+[b.norm2 for b in m.blocks]+[m.final_norm]
            assert all(n.eps==1e-5 and getattr(n,'bias',None) is None for n in norms)
            a=forward(m,ids);b=independent(m,ids,group)
            error=(a-b).abs().max().item();assert error<=1e-10
            params=list(m.parameters());ga=torch.autograd.grad(F.cross_entropy(a[:,-1],labels),params);gb=torch.autograd.grad(F.cross_entropy(b[:,-1],labels),params)
            ge=max((x-y).abs().max().item() for x,y in zip(ga,gb));assert ge<=1e-10
            assert all(torch.isfinite(g).all() for g in ga) and max(g.abs().max().item() for g in ga)>1e-7
            changed=ids.clone();changed[:,-1]=11
            with torch.no_grad():c=forward(m,changed)
            prefix=(a[:,:-1]-c[:,:-1]).abs().max().item();final=(a[:,-1]-c[:,-1]).abs().max().item()
            assert prefix<=1e-10 and final>1e-6
            wrong=(a-independent(m,ids,group,force_ln=True)).abs().max().item()
            if group=='rms':assert wrong>1e-6
            prior_error=None
            if group=='ln':
                torch.manual_seed(seed);prior=original_model('mqa');prior_error=(a-forward(prior,ids)).abs().max().item();assert prior_error<=1e-10
            records.append(dict(seed=seed,group=group,initial_sha256=hashes[-1],max_logit_error=error,max_gradient_error=ge,causal_prefix_error=prefix,changed_query_error=final,wrong_centering_error=wrong,initial_archived_ln_error=prior_error))
        assert hashes[0]==hashes[1]
    result=dict(passed=True,checks=records,provenance=provenance(),source_sha256=digest(ROOT/'experiments/verify_rmsnorm_training.py'))
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,models=len(records),max_gradient_error=max(x['max_gradient_error'] for x in records))))


if __name__=='__main__':main()
