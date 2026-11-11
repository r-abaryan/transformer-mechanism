"""Distinguish score rank, probability rank, and log-odds representability."""
import hashlib
import json
import math
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def spectrum(t):
    s=torch.linalg.svdvals(t)
    return dict(singular_values=s.tolist(),rank=int((s>1e-10).sum()))


def main():
    p=ROOT/'results/head_width.json'; assert not p.exists()
    torch.set_num_threads(1)
    q=torch.arange(4,dtype=torch.float64)[:,None]; s=q@q.T; a=s.softmax(-1)
    assert spectrum(s)['rank']==1 and spectrum(a)['rank']==4
    x=q.squeeze().exp(); det=math.prod((x[j]-x[i]).item() for i in range(4) for j in range(i+1,4))/s.exp().sum(-1).prod().item()
    assert abs(det-torch.linalg.det(a).item())<1e-10
    desired=torch.eye(4,dtype=torch.float64).softmax(-1)
    b=(desired[:,:3]/desired[:,3:]).log()
    exact=torch.cat([torch.eye(3,dtype=torch.float64),-torch.ones(1,3,dtype=torch.float64)])
    assert (b-exact).abs().max()<1e-10
    assert (b.T@b-(torch.eye(3)+torch.ones(3,3))).abs().max()<1e-10
    assert spectrum(b)['rank']==3
    u,sv,vh=torch.linalg.svd(b,full_matrices=False)
    approximations=[]
    for width,expected in [(1,math.sqrt(2)),(2,1.)]:
        low=(u[:,:width]*sv[:width])@vh[:width]
        error=(b-low).norm().item(); assert abs(error-expected)<1e-10
        approximations.append(dict(width=width,log_odds=low.tolist(),frobenius_error=error))
    keys=torch.cat([torch.eye(3,dtype=torch.float64),torch.zeros(1,3,dtype=torch.float64)])
    queries=math.sqrt(3)*b
    recovered=(queries@keys.T/math.sqrt(3)).softmax(-1)
    recovery_error=(recovered-desired).abs().max().item(); assert recovery_error<1e-10
    causal=torch.zeros(4,4,dtype=torch.float64).masked_fill(torch.ones(4,4,dtype=torch.bool).triu(1),float('-inf')).softmax(-1)
    assert spectrum(causal)['rank']==4 and abs(torch.linalg.det(causal).item()-1/24)<1e-10
    r=dict(passed=True,torch=torch.__version__,dtype='float64',hashes={f:digest(ROOT/f) for f in ['experiments/head_width.py','docs/head-width-protocol.md']},
        width1=dict(q=q.tolist(),scores=s.tolist(),attention=a.tolist(),score_spectrum=spectrum(s),attention_spectrum=spectrum(a),determinant=torch.linalg.det(a).item(),vandermonde_determinant=det),
        target=dict(attention=desired.tolist(),log_odds=b.tolist(),spectrum=spectrum(b),gram=(b.T@b).tolist(),approximations=approximations),
        width3=dict(q=queries.tolist(),k=keys.tolist(),attention=recovered.tolist(),max_error=recovery_error),
        causal=dict(attention=causal.tolist(),spectrum=spectrum(causal),determinant=torch.linalg.det(causal).item()))
    p.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps({k:v for k,v in r.items() if k in ['passed','width1','width3','causal']}))


if __name__=='__main__': main()
