"""Pretraining checks with independent rotation matrices and finite differences."""
import copy,json,math
import torch
from positional_comparison import ROOT,forward,rotate,model_new,digest
from position_training import shifted

def main():
    torch.set_num_threads(1); torch.manual_seed(19)
    m=model_new(); other=copy.deepcopy(m); x=torch.tensor([[0,12,1,13,2,14,18,1]])
    a=forward(m,x,'baseline')[:,-1]; b=shifted(other,x,4)
    e=(a-b).abs().max().item(); assert e<1e-10
    a.square().sum().backward(); b.square().sum().backward()
    g=max((p.grad-q.grad).abs().max().item() for p,q in zip(m.parameters(),other.parameters())); assert g<1e-9
    q=torch.randn(1,2,8,8,dtype=torch.float64); k=torch.randn_like(q); pos=torch.arange(8)+4
    rot=rotate(q,pos); norm=(rot.norm(dim=-1)-q.norm(dim=-1)).abs().max().item(); assert norm<1e-10
    explicit=torch.empty_like(q)
    for i in range(8):
        for r in range(4):
            angle=float(pos[i])*10000**(-2*r/8); c=math.cos(angle); s=math.sin(angle)
            matrix=torch.tensor([[c,-s],[s,c]],dtype=torch.float64)
            explicit[:,:,i,2*r:2*r+2]=q[:,:,i,2*r:2*r+2]@matrix.T
    rotation=(rot-explicit).abs().max().item(); assert rotation<1e-10
    lhs=(rotate(q,pos)@rotate(k,pos).transpose(-1,-2))
    relative=0.
    for i in range(8):
        for j in range(8):
            rhs=(q[:,:,i]*rotate(k[:,:,j:j+1],torch.tensor([j-i]))[:,:,0]).sum(-1)
            relative=max(relative,(lhs[:,:,i,j]-rhs).abs().max().item())
    assert relative<1e-10
    offset=(forward(m,x,'rope',4)-forward(m,x,'rope',19)).abs().max().item(); assert offset<1e-10
    changed=x.clone(); changed[:,-1]=2
    causal=max((forward(m,x,mode)[:,:-1]-forward(m,changed,mode)[:,:-1]).abs().max().item() for mode in ['baseline','rope']); assert causal<1e-10
    m.zero_grad(); loss=forward(m,x,'rope').square().sum(); loss.backward()
    parameter=m.blocks[0].self_attn.in_proj_weight; index=(0,0); analytic=parameter.grad[index].item(); h=1e-5
    with torch.no_grad():
        original=parameter[index].item(); parameter[index]=original+h; plus=forward(m,x,'rope').square().sum().item()
        parameter[index]=original-h; minus=forward(m,x,'rope').square().sum().item(); parameter[index]=original
    finite=abs((plus-minus)/(2*h)-analytic); assert finite<1e-6
    report=dict(passed=True,baseline_output=e,baseline_gradient=g,rotation=rotation,norm=norm,relative=relative,offset=offset,causal=causal,finite_difference=finite,source_sha256=digest(ROOT/'experiments/positional_comparison.py'),verification_sha256=digest(ROOT/'experiments/verify_positional_comparison.py'))
    (ROOT/'results/positional_comparison_verification.json').write_text(json.dumps(report,indent=2)+'\n'); print(report)
if __name__=='__main__': main()
