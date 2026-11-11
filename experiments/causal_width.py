"""A common-visible-key causal width obstruction and two output witnesses."""
import hashlib
import json
import math
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    path=ROOT/'results/causal_width.json'; assert not path.exists()
    k=torch.tensor([[1,0],[0,1],[0,0],[0,0]],dtype=torch.float64)
    q=math.sqrt(2)*torch.tensor([[0,0],[0,0],[1,0],[0,1]],dtype=torch.float64)
    scores=q@k.T/math.sqrt(2); mask=torch.ones(4,4,dtype=torch.bool).triu(1)
    a=scores.masked_fill(mask,float('-inf')).softmax(-1)
    assert a[mask].eq(0).all() and (a[~mask]>0).all() and (a.sum(-1)-1).abs().max()<1e-10
    expected=torch.tensor([[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,1,0,0]],dtype=torch.float64)
    assert (scores-expected).abs().max()<1e-10
    b=(a[2:,:2]/a[2:,2:3]).log()
    assert (b-torch.eye(2)).abs().max()<1e-10
    assert torch.linalg.matrix_rank(b,atol=1e-10)==2
    v=torch.arange(4,dtype=torch.float64)[:,None]; target=a@v; queries=[]
    for i in range(4):
        if i==0: queries.append(0.); continue
        keys=v[:i+1,0]
        def f(t): return (keys*t).softmax(0)@keys
        lo=-10.; hi=10.; assert f(lo)<target[i,0]<f(hi)
        for _ in range(80):
            mid=(lo+hi)/2
            if f(mid)<target[i,0]: lo=mid
            else: hi=mid
        queries.append((lo+hi)/2)
    narrow_q=torch.tensor(queries,dtype=torch.float64)[:,None]
    narrow=(narrow_q@v.T).masked_fill(mask,float('-inf')).softmax(-1)
    err=(narrow@v-target).abs().max().item(); diff=(narrow-a).abs().max().item()
    assert err<1e-10 and diff>.01
    identifiable=torch.cat([torch.eye(3,dtype=torch.float64),torch.zeros(1,3,dtype=torch.float64)])
    out=a@identifiable; recovered=torch.cat([out,1-out.sum(-1,keepdim=True)],-1)
    recovery=(recovered-a).abs().max().item(); mismatch=(narrow@identifiable-out).abs().max().item()
    assert recovery<1e-10 and mismatch>.01
    r=dict(passed=True,torch=torch.__version__,hashes={p:digest(ROOT/p) for p in ['experiments/causal_width.py','docs/causal-width-protocol.md']},
           q=q.tolist(),k=k.tolist(),scores=scores.tolist(),attention=a.tolist(),log_odds=b.tolist(),log_odds_singular_values=torch.linalg.svdvals(b).tolist(),
           attention_rank=int(torch.linalg.matrix_rank(a,atol=1e-10)),
           scalar=dict(values=v.tolist(),q=narrow_q.tolist(),k=v.tolist(),attention=narrow.tolist(),output=(narrow@v).tolist(),target_output=target.tolist(),max_output_error=err,max_attention_difference=diff),
           identifiable=dict(values=identifiable.tolist(),output=out.tolist(),recovered=recovered.tolist(),max_recovery_error=recovery,negative_control_error=mismatch))
    path.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(dict(output_error=err,attention_difference=diff,recovery_error=recovery,negative_control_error=mismatch)))


if __name__=='__main__': main()
