"""Finite witnesses for centering, epsilon and local normalization gradients."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def fixtures():
    for d in (4,16):
        i=torch.arange(1,d+1,dtype=torch.float64)
        vectors=dict(ordinary=i,centered=i-(d+1)/2,constant=torch.full((d,),2.,dtype=torch.float64),
                     near_zero=torch.where(i.long()%2==1,i,-i)*1e-6,zero=torch.zeros(d,dtype=torch.float64))
        for name,x in vectors.items():
            for eps in (0.,1e-5):
                for method in ('ln','rms'):
                    z=x-x.mean() if method=='ln' else x
                    if eps==0 and z.square().mean()==0:continue
                    for gain,gamma in (('unit',torch.ones(d,dtype=torch.float64)),('varying',.5+i/d)):
                        yield dict(dimension=d,vector=name,method=method,epsilon=eps,gain=gain),x,gamma


def explicit(x,gamma,method,eps):
    z=x-x.mean() if method=='ln' else x
    return gamma*z/(z.square().mean()+eps).sqrt()


def jacobian(x,gamma,method,eps):
    d=len(x);identity=torch.eye(d,dtype=torch.float64)
    p=identity-torch.ones(d,d,dtype=torch.float64)/d if method=='ln' else identity
    z=p@x;r=(z.square().mean()+eps).sqrt()
    return gamma[:,None]*(p/r-torch.outer(z,z)/(d*r**3)),z,r


def native(x,gamma,method,eps):
    return F.layer_norm(x,(len(x),),gamma,None,eps) if method=='ln' else F.rms_norm(x,(len(x),),gamma,eps)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    target=ROOT/'results/rmsnorm_math.json';assert not target.exists(),'Preserve opened artifacts.'
    paths=('experiments/rmsnorm_math.py','experiments/audit_rmsnorm_math.py','docs/rmsnorm-math-protocol.md')
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',threads=1,
                provenance={p:digest(ROOT/p) for p in paths},cases=[])
    for key,x,gamma in fixtures():
        method,eps=key['method'],key['epsilon'];y=explicit(x,gamma,method,eps)
        j,z,r=jacobian(x,gamma,method,eps);scale=max(1.,float(j.abs().max()))
        library=native(x,gamma,method,eps)
        autograd=torch.autograd.functional.jacobian(lambda t:native(t,gamma,method,eps),x)
        h=max(float(x.abs().max()),1e-8)*1e-5;basis=torch.eye(len(x),dtype=torch.float64)
        fd=torch.stack([(explicit(x+h*e,gamma,method,eps)-explicit(x-h*e,gamma,method,eps))/(2*h) for e in basis],1)
        ones=torch.ones_like(x)
        checks=dict(native_error=float((library-y).abs().max()),jacobian_relative_error=float((j-autograd).abs().max())/scale,
                    finite_difference_relative_error=float((j-fd).abs().max())/scale,
                    radial_identity_relative_error=float((j@z-gamma*z*eps/r**3).abs().max())/scale)
        if method=='ln':checks['mean_null_relative_error']=float((j@ones).abs().max())/scale
        spectrum={}
        if key['gain']=='unit':
            expected=([0.,float(eps/r**3)]+[float(1/r)]*(len(x)-2)) if method=='ln' else [float(eps/r**3)]+[float(1/r)]*(len(x)-1)
            eigen=torch.linalg.eigvalsh(j);error=float((eigen-torch.tensor(sorted(expected),dtype=torch.float64)).abs().max())/scale
            checks['spectrum_relative_error']=error;spectrum=dict(measured=eigen.tolist(),expected=sorted(expected))
        transforms={name:explicit(t,gamma,method,eps).tolist() for name,t in (('shift',x+3),('positive_scale',2*x),('negative_scale',-x))}
        assert checks['native_error']<=1e-10 and checks['finite_difference_relative_error']<=1e-7
        assert all(v<=1e-10 for k,v in checks.items() if k not in ('native_error','finite_difference_relative_error'))
        if method=='ln':assert (torch.tensor(transforms['shift'],dtype=torch.float64)-y).abs().max()<=1e-8
        if eps==0:assert (torch.tensor(transforms['positive_scale'],dtype=torch.float64)-y).abs().max()<=1e-10
        assert (torch.tensor(transforms['negative_scale'],dtype=torch.float64)+y).abs().max()<=1e-10
        report['cases'].append(dict(**key,input=x.tolist(),gamma=gamma.tolist(),output=y.tolist(),jacobian=j.tolist(),
                                   autograd_jacobian=autograd.tolist(),finite_difference_jacobian=fd.tolist(),
                                   checks=checks,transformed_outputs=transforms,spectrum=spectrum))
    assert len(report['cases'])==68
    x=torch.tensor([-3.,-1.,1.,3.],dtype=torch.float64);gamma=torch.ones_like(x);eps=1e-5
    a,b=[explicit(x,gamma,m,eps) for m in ('ln','rms')]
    jl,_,_=jacobian(x,gamma,'ln',eps);jr,_,_=jacobian(x,gamma,'rms',eps)
    report['centered_witness']=dict(input=x.tolist(),epsilon=eps,output_error=float((a-b).abs().max()),
        ln_mean_direction=(jl@gamma).tolist(),rms_mean_direction=(jr@gamma).tolist(),jacobian_difference=(jr-jl).tolist())
    x=torch.arange(1,5,dtype=torch.float64)
    report['residual_witness']=dict(input=x.tolist(),shift=3.,epsilon=eps,
        differences={m:(x+3+explicit(x+3,gamma,m,eps)-x-explicit(x,gamma,m,eps)).tolist() for m in ('ln','rms')})
    report.update(phase='complete',passed=True)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(report['cases']),max_autograd_relative_error=max(c['checks']['jacobian_relative_error'] for c in report['cases']),
                         max_finite_difference_relative_error=max(c['checks']['finite_difference_relative_error'] for c in report['cases']))))


if __name__=='__main__':main()
