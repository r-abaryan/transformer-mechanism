"""Synthetic activation identities and model gradients before task training."""
import copy
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from gelu_comparison import ROOT, forward, exact_gelu
from positional_comparison import forward as prior_forward
from reserved_cycle import model_new
from position_distractors import digest


def main():
    torch.set_num_threads(1); torch.manual_seed(20270105)
    x=torch.linspace(-4,4,81,dtype=torch.float64,requires_grad=True)
    y=exact_gelu(x); expected=x*(1+torch.erf(x/math.sqrt(2)))/2
    output=(y-expected).abs().max().item(); assert output<1e-12
    grad=torch.autograd.grad(y.sum(),x)[0]
    derivative=(1+torch.erf(x/math.sqrt(2)))/2+x*torch.exp(-x*x/2)/math.sqrt(2*math.pi)
    gradient=(grad-derivative).abs().max().item(); assert gradient<1e-12
    relu_gradient=(torch.autograd.grad(F.relu(x).sum(),x)[0]-(x>0).double()).abs().max().item(); assert relu_gradient==0
    model=model_new(); reference=copy.deepcopy(model); ids=torch.randint(0,19,(2,8))
    a=forward(model,ids,'relu'); b=prior_forward(reference,ids,'rope')
    baseline_output=(a-b).abs().max().item(); assert baseline_output<1e-10
    a.square().sum().backward(); b.square().sum().backward()
    baseline_gradient=max((p.grad-q.grad).abs().max().item() for p,q in zip(model.parameters(),reference.parameters())); assert baseline_gradient<1e-9
    model.zero_grad(); g=forward(model,ids,'gelu'); changed=(g-a).abs().max().item(); assert changed>1e-6
    g.square().sum().backward(); p=model.blocks[0].linear1.weight; index=(0,0); analytic=p.grad[index].item(); h=1e-5
    with torch.no_grad():
        initial=p[index].item(); p[index]=initial+h; plus=forward(model,ids,'gelu').square().sum().item()
        p[index]=initial-h; minus=forward(model,ids,'gelu').square().sum().item(); p[index]=initial
    finite_difference=abs((plus-minus)/(2*h)-analytic); assert finite_difference<1e-6
    result=dict(passed=True,gelu_output=output,gelu_derivative=gradient,relu_derivative=relu_gradient,baseline_output=baseline_output,baseline_gradient=baseline_gradient,activation_changes_output=changed,finite_difference=finite_difference,source_sha256=digest(ROOT/'experiments/gelu_comparison.py'),verification_sha256=digest(Path(__file__)))
    (ROOT/'results/gelu_comparison_verification.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
