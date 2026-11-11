"""Width/count, independent gradients and function-containment fixtures."""
import copy
import json
from pathlib import Path
import torch
from ffn_width import ROOT,SEEDS,model_new,forward,shared_hash
from positional_comparison import forward as prior_forward
from audit_ffn_width import independent
from position_distractors import digest


def main():
    torch.set_num_threads(1)
    for seed in SEEDS:
        torch.manual_seed(seed); narrow=model_new('ff32')
        torch.manual_seed(seed); wide=model_new('ff64')
        assert shared_hash(narrow)==shared_hash(wide)
    assert sum(p.numel() for p in narrow.parameters())==5088
    assert sum(p.numel() for p in wide.parameters())==7200
    torch.manual_seed(20270106); ids=torch.randint(0,19,(2,8)); checks={}
    for group,model in [('ff32',narrow),('ff64',wide)]:
        ref=copy.deepcopy(model); a=forward(model,ids,group)[:,-1]
        b=prior_forward(ref,ids,'rope')[:,-1] if group=='ff32' else independent(ref,ids,group)
        error=(a-b).abs().max().item(); assert error<1e-10
        a.square().sum().backward();b.square().sum().backward()
        grad=max((p.grad-q.grad).abs().max().item() for p,q in zip(model.parameters(),ref.parameters())); assert grad<1e-9
        changed=ids.clone();changed[:,-1]=(changed[:,-1]+1)%19
        causal=(forward(model,ids,group)[:,:-1]-forward(model,changed,group)[:,:-1]).abs().max().item();assert causal<1e-10
        checks[group]=dict(output=error,gradient=grad,causal=causal)
    witness=copy.deepcopy(wide)
    with torch.no_grad():
        for a,b in zip(narrow.blocks,witness.blocks):
            b.linear1.weight.zero_();b.linear1.bias.zero_();b.linear2.weight.zero_()
            b.linear1.weight[:32].copy_(a.linear1.weight);b.linear1.bias[:32].copy_(a.linear1.bias)
            b.linear2.weight[:,:32].copy_(a.linear2.weight);b.linear2.bias.copy_(a.linear2.bias)
        containment=(forward(narrow,ids,'ff32')-forward(witness,ids,'ff64')).abs().max().item(); assert containment<1e-10
    result=dict(passed=True,checks=checks,containment_error=containment,parameters=[5088,7200],shared_initial_seeds=list(SEEDS),source_sha256=digest(ROOT/'experiments/ffn_width.py'),verification_sha256=digest(Path(__file__)))
    (ROOT/'results/ffn_width_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
