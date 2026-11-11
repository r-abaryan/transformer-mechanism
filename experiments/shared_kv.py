"""Four fixed witnesses for shared-KV storage, equivalence and restrictions."""
import json
import math
from pathlib import Path

import torch
from head_width import digest, spectrum

ROOT = Path(__file__).resolve().parents[1]


def attn(q,k,v,causal=False):
    scores = q@k.T/math.sqrt(q.shape[-1])
    if causal:
        scores = scores.masked_fill(torch.ones_like(scores,dtype=torch.bool).triu(1),-torch.inf)
    a = scores.softmax(-1)
    return a,a@v


def main():
    torch.set_num_threads(1)
    path = ROOT/'results/shared_kv.json'
    assert not path.exists(), 'Preserve opened results.'
    frozen = ['experiments/shared_kv.py','experiments/audit_shared_kv.py',
              'experiments/head_width.py','docs/shared-kv-protocol.md']
    hashes = {p:digest(ROOT/p) for p in frozen}
    prior_path = ROOT/'results/head_width.json'
    prior = json.loads(prior_path.read_text()); assert prior['passed']
    for p,h in prior['hashes'].items(): assert digest(ROOT/p)==h
    prior_audit_path = ROOT/'results/head_width_audit.json'
    prior_audit = json.loads(prior_audit_path.read_text())
    assert prior_audit['passed'] and prior_audit['report_sha256']==digest(prior_path)
    storage = []
    for groups in (4,2,1):
        cache = [(torch.zeros(2,groups,128,2,dtype=torch.float64),
                  torch.zeros(2,groups,128,3,dtype=torch.float64)) for layer in range(2)]
        assert len({t.data_ptr() for pair in cache for t in pair})==4
        byte_count = sum(t.untyped_storage().nbytes() for pair in cache for t in pair)
        assert byte_count==2*2*128*groups*(2+3)*8
        projections = [torch.zeros(shape,dtype=torch.float64) for shape in
                       ((8,4*2),(8,groups*2),(8,groups*3),(4*3,8))]
        count = sum(p.numel() for p in projections)
        assert count==8*4*2+8*groups*(2+3)+4*3*8
        storage.append(dict(groups=groups,cache_bytes=byte_count,projection_parameters=count,
                            k_shape=list(cache[0][0].shape),v_shape=list(cache[0][1].shape)))
    equivalent = []
    q = (torch.arange(40,dtype=torch.float64).reshape(4,5,2)%7-3)/3
    for groups in (4,2,1):
        keys = (torch.arange(groups*10,dtype=torch.float64).reshape(groups,5,2)%5-2)/2
        values = (torch.arange(groups*15,dtype=torch.float64).reshape(groups,5,3)%11-5)/4
        dense_k = keys.repeat_interleave(4//groups,0)
        dense_v = values.repeat_interleave(4//groups,0)
        for causal in (False,True):
            outputs,weights = [],[]
            for head in range(4):
                g = head//(4//groups)
                a,y = attn(q[head],keys[g],values[g],causal)
                weights.append(a);outputs.append(y)
            a,y = torch.stack(weights),torch.stack(outputs)
            dense_s = q@dense_k.transpose(-1,-2)/math.sqrt(2)
            if causal:
                dense_s = dense_s.masked_fill(torch.ones(5,5,dtype=torch.bool).triu(1),-torch.inf)
            dense_a = dense_s.softmax(-1);dense_y = dense_a@dense_v
            error = max((a-dense_a).abs().max().item(),(y-dense_y).abs().max().item())
            assert error<=1e-10
            different = (a[0]-a[1]).abs().max().item()
            if groups==1 and not causal: assert different>1e-3
            equivalent.append(dict(groups=groups,causal=causal,q=q.tolist(),k=keys.tolist(),v=values.tolist(),
                                   attention=a.tolist(),output=y.tolist(),dense_attention=dense_a.tolist(),
                                   dense_output=dense_y.tolist(),max_error=error,head01_attention_difference=different))
    scores = torch.tensor([[1.,0.,0.],[0.,1.,0.]],dtype=torch.float64)
    target = scores.softmax(-1)
    odds = (target[:,:2]/target[:,2:]).log()
    assert (odds-torch.eye(2,dtype=torch.float64)).abs().max()<=1e-10
    assert spectrum(odds)['rank']==2
    shared_k = torch.tensor([[1.,0.],[0.,1.],[0.,0.]],dtype=torch.float64)
    shared_q = math.sqrt(2)*torch.eye(2,dtype=torch.float64)
    recovered = (shared_q@shared_k.T/math.sqrt(2)).softmax(-1)
    error = (recovered-target).abs().max().item();assert error<=1e-10
    routing = dict(scores=scores.tolist(),attention=target.tolist(),joint_log_odds=odds.tolist(),
                   spectrum=spectrum(odds),shared_width2_q=shared_q.tolist(),shared_width2_k=shared_k.tolist(),
                   recovered_attention=recovered.tolist(),max_error=error)
    one = torch.ones(1,1,dtype=torch.float64)
    k = torch.tensor([[1.,0.,0.],[0.,1.,0.]],dtype=torch.float64).unsqueeze(-1)
    v = torch.tensor([[0.,1.,2.],[3.,4.,5.]],dtype=torch.float64).unsqueeze(-1)
    before = torch.stack([attn(one,k[h],v[h])[1] for h in range(2)])
    mean_k,mean_v = k.mean(0),v.mean(0)
    pooled = attn(one,mean_k,mean_v)[1].repeat(2,1,1)
    pool_error = (pooled-before).abs().max().item();assert pool_error>1e-3
    identical = torch.stack([attn(one,k[0],v[0])[1] for h in range(2)])
    control = attn(one,torch.stack([k[0]]*2).mean(0),torch.stack([v[0]]*2).mean(0))[1].repeat(2,1,1)
    control_error = (identical-control).abs().max().item();assert control_error<=1e-10
    pooling = dict(q=one.tolist(),k=k.tolist(),v=v.tolist(),original_outputs=before.flatten().tolist(),
                   pooled_outputs=pooled.flatten().tolist(),max_output_error=pool_error,
                   identical_head_output=identical.flatten().tolist(),identical_control_error=control_error)
    report = dict(passed=True,torch=torch.__version__,device='cpu',dtype='float64',hashes=hashes,
                  prior_head_width_sha256=digest(prior_path),prior_head_width_audit_sha256=digest(prior_audit_path),
                  storage=storage,equivalence=equivalent,
                  routing=routing,pooling=pooling)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,storage=storage,max_equivalence_error=max(c['max_error'] for c in equivalent),
                         joint_rank=routing['spectrum']['rank'],pooling_error=pool_error)))


if __name__=='__main__':
    main()
