"""Two missing witnesses connecting attention-pattern and output capacity."""
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    path=ROOT/'results/output_capacity.json'; assert not path.exists()
    hp=ROOT/'results/head_width.json'; old=json.loads(hp.read_text()); assert old['passed']
    for name,h in old['hashes'].items(): assert digest(ROOT/name)==h
    sp=ROOT/'results/attention_spikes.json'; spikes=json.loads(sp.read_text())
    witness=next(x for x in spikes['records'] if x['id']=='S05'); assert witness['status']=='pass'
    a=torch.tensor(old['target']['attention'],dtype=torch.float64)
    v=torch.arange(4,dtype=torch.float64)[:,None]; desired=a@v
    qs=[]
    for target in desired[:,0]:
        low=-10.; high=10.
        def output(q): return (v[:,0]*q).softmax(0)@v[:,0]
        assert output(low)<target<output(high)
        for _ in range(80):
            mid=(low+high)/2
            if output(mid)<target: low=mid
            else: high=mid
        qs.append((low+high)/2)
    q=torch.tensor(qs,dtype=torch.float64)[:,None]; other=(q@v.T).softmax(-1)
    output_error=(other@v-desired).abs().max().item(); attention_error=(other-a).abs().max().item()
    assert output_error<1e-10 and attention_error>.01
    unique=torch.cat([torch.eye(3,dtype=torch.float64),torch.zeros(1,3,dtype=torch.float64)])
    unique_out=a@unique; recovered=torch.cat([unique_out,1-unique_out.sum(-1,keepdim=True)],-1)
    recovery_error=(recovered-a).abs().max().item(); control=(other@unique-unique_out).abs().max().item()
    assert recovery_error<1e-10 and control>.01
    aug=torch.cat([unique,torch.ones(4,1,dtype=torch.float64)],-1)
    assert torch.linalg.matrix_rank(aug,atol=1e-10)==4
    r=dict(passed=True,hashes={p:digest(ROOT/p) for p in ['experiments/output_capacity.py','docs/output-capacity-protocol.md','results/head_width.json','results/attention_spikes.json']},
           torch=torch.__version__,target_attention=a.tolist(),reused_s05=witness['metrics'],
           scalar=dict(values=v.tolist(),q=q.tolist(),k=v.tolist(),attention=other.tolist(),target_output=desired.tolist(),actual_output=(other@v).tolist(),max_output_error=output_error,max_attention_difference=attention_error,augmented_rank=int(torch.linalg.matrix_rank(torch.cat([v,torch.ones_like(v)],1),atol=1e-10))),
           identifiable=dict(values=unique.tolist(),output=unique_out.tolist(),recovered_attention=recovered.tolist(),max_recovery_error=recovery_error,wrong_attention_output_error=control,value_rank=3,augmented_rank=4))
    path.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(output_error=output_error,attention_difference=attention_error,recovery_error=recovery_error,negative_control_error=control)))


if __name__=='__main__': main()
