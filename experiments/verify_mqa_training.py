"""Before training, corroborate functions, gradients, causal mask and savings."""
import json
import torch
from torch.nn import functional as F
from mqa_training import ROOT,model_new,template_model,shared_hash,retained_kv_hash,forward
from audit_mqa_training import independent
from positional_comparison import forward as previous_forward
from position_distractors import digest


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    ids=torch.tensor([[0,12,1,13,0,14,18,0],[2,15,3,16,2,17,18,3]])
    y=torch.tensor([14,17]);checks={}
    torch.manual_seed(1411);template=template_model()
    torch.manual_seed(1411);baseline=model_new('mha')
    old=previous_forward(template,ids,'rope');new=forward(baseline,ids)
    library_error=(old-new).abs().max().item();assert library_error<=1e-10
    pair={}
    for group in ('mha','mqa'):
        torch.manual_seed(1411);m=model_new(group);pair[group]=m
        a,cache=forward(m,ids,capture=True);b=independent(m,ids)
        error=(a-b).abs().max().item();assert error<=1e-10
        params=list(m.parameters())
        ga=torch.autograd.grad(F.cross_entropy(a[:,-1],y),params)
        gb=torch.autograd.grad(F.cross_entropy(b[:,-1],y),params)
        gradient_error=max((x-z).abs().max().item() for x,z in zip(ga,gb));assert gradient_error<=1e-10
        assert max(x.abs().max().item() for x in ga)>1e-7
        changed=ids.clone();changed[:,-1]=11
        with torch.no_grad():c=forward(m,changed)
        prefix_error=(a[:,:-1]-c[:,:-1]).abs().max().item();final_change=(a[:,-1]-c[:,-1]).abs().max().item()
        assert prefix_error<=1e-10 and final_change>1e-6
        count=sum(p.numel() for p in params);assert count==(5088 if group=='mha' else 4544)
        size=sum(t.untyped_storage().nbytes() for pair in cache for t in pair)
        assert size==2*2*m.groups*8*16*8
        checks[group]=dict(max_logit_error=error,max_gradient_error=gradient_error,causal_prefix_error=prefix_error,
                           changed_query_logit_difference=final_change,parameters=count,cache_bytes=size)
    assert shared_hash(pair['mha'])==shared_hash(pair['mqa']) and retained_kv_hash(pair['mha'])==retained_kv_hash(pair['mqa'])
    assert checks['mha']['cache_bytes']==2*checks['mqa']['cache_bytes']
    result=dict(passed=True,mha_previous_forward_error=library_error,checks=checks,
                source_sha256=digest(ROOT/'experiments/mqa_training.py'),
                auditor_sha256=digest(ROOT/'experiments/audit_mqa_training.py'),
                verification_source_sha256=digest(ROOT/'experiments/verify_mqa_training.py'))
    (ROOT/'results/mqa_training_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
