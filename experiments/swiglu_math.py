"""Analytical, autograd, finite-difference and Decimal checks of a gated FFN."""
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path
import torch
from torch.nn import functional as F
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    scalar=[]
    with localcontext() as context:
        context.prec=60
        for a in (-20.,-3.,-1.,0.,1.,3.,20.):
            for u in (-2.,0.,2.):
                x=torch.tensor(a,dtype=torch.float64,requires_grad=True)
                v=torch.tensor(u,dtype=torch.float64,requires_grad=True)
                actual=F.silu(x)*v
                auto=torch.autograd.grad(actual,(x,v))
                s=1/(1+math.exp(-a));gate=a*s
                da=u*(s+a*s*(1-s));du=gate
                delta=1e-5
                fn=lambda aa,uu:aa/(1+math.exp(-aa))*uu
                fd=((fn(a+delta,u)-fn(a-delta,u))/(2*delta),(fn(a,u+delta)-fn(a,u-delta))/(2*delta))
                aa,uu=Decimal(str(a)),Decimal(str(u))
                ss=1/(1+(-aa).exp())
                dec=(aa*ss*uu,uu*(ss+aa*ss*(1-ss)),aa*ss)
                errors=[abs(actual.item()-float(dec[0])),abs(auto[0].item()-float(dec[1])),abs(auto[1].item()-float(dec[2]))]
                assert max(errors)<=1e-12
                assert max(abs(auto[0].item()-da),abs(auto[1].item()-du))<=1e-12
                assert max(abs(fd[0]-da),abs(fd[1]-du))<=1e-8
                scalar.append(dict(a=a,u=u,gate=gate,output=actual.item(),da=da,du=du,max_decimal_error=max(errors),max_fd_error=max(abs(fd[0]-da),abs(fd[1]-du))))
    g=torch.Generator().manual_seed(1769)
    matrices=[]
    for d,h in ((3,2),(4,5),(16,22)):
        for sample in range(3):
            x=torch.randn(2,d,generator=g,dtype=torch.float64,requires_grad=True)
            wg=torch.randn(d,h,generator=g,dtype=torch.float64,requires_grad=True)
            wu=torch.randn(d,h,generator=g,dtype=torch.float64,requires_grad=True)
            wo=torch.randn(h,d,generator=g,dtype=torch.float64,requires_grad=True)
            upstream=torch.randn(2,d,generator=g,dtype=torch.float64)
            a=x@wg;u=x@wu;s=a.sigmoid();gate=F.silu(a)
            out=(gate*u)@wo
            auto=torch.autograd.grad((out*upstream).sum(),(x,wg,wu,wo))
            back=upstream@wo.T
            da=back*u*(s+a*s*(1-s));du=back*gate
            analytical=(da@wg.T+du@wu.T,x.T@da,x.T@du,(gate*u).T@upstream)
            error=max((aa-bb).abs().max().item() for aa,bb in zip(auto,analytical))
            assert error<=1e-10
            params=(x,wg,wu,wo)
            direction=[torch.randn(t.shape,generator=g,dtype=torch.float64) for t in params]
            fn=lambda xs,gs,us,os:((F.silu(xs@gs)*(xs@us))@os*upstream).sum()
            eps=1e-6
            fd=(fn(*[t+eps*v for t,v in zip(params,direction)])-fn(*[t-eps*v for t,v in zip(params,direction)])).item()/(2*eps)
            slope=sum((aa*vv).sum().item() for aa,vv in zip(analytical,direction))
            relative=abs(fd-slope)/max(1.,abs(slope))
            assert relative<=1e-6
            matrices.append(dict(d=d,h=h,sample=sample,max_gradient_error=error,directional_fd_relative_error=relative))
    # Three false guarantees, with explicit witnesses.
    witnesses=dict(negative_gate=-1/(1+math.e),amplifying_gate=3/(1+math.exp(-3)),
                   gate_slope_at_zero=.5,zero_content_gate_gradient=0.,zero_gate_content_gradient=0.)
    assert witnesses['negative_gate']<0 and witnesses['amplifying_gate']>1
    a,u=1.,2.;correct=a/(1+math.exp(-a))*u
    # Use a=3 for sigmoid-only, since a=1 is an accidental identical-output point.
    wrong_sigmoid=abs(3/(1+math.exp(-3))*u-1/(1+math.exp(-3))*u)
    wrong_no_product=abs(correct-1/(1+math.exp(-1)))
    wrong_da=abs(2*(.5+0)-0.)
    assert min(wrong_sigmoid,wrong_no_product,wrong_da)>1e-6
    r=dict(passed=True,scalar_cases=scalar,matrix_cases=matrices,witnesses=witnesses,
           wrong_controls=dict(sigmoid_only_error=wrong_sigmoid,omitted_product_error=wrong_no_product,omitted_sigmoid_term_gradient_error=wrong_da),
           matched_cost=dict(relu_hidden=33,swiglu_hidden=22,weights_per_block=1056,projection_macs_per_token_block=1056),
           source_sha256=digest(Path(__file__)),protocol_sha256=digest(ROOT/'docs/decisions/0004-swiglu-matched-ffn.md'))
    assert 2*16*33==3*16*22==1056
    p=ROOT/'results/swiglu_math.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(passed=True,scalar_cases=len(scalar),matrix_cases=len(matrices),max_decimal_error=max(c['max_decimal_error'] for c in scalar),max_gradient_error=max(c['max_gradient_error'] for c in matrices),wrong_controls=r['wrong_controls'])))


if __name__=='__main__':main()
