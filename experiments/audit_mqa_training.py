"""Independently verify matched data, initialization and trained predictions."""
import hashlib
import itertools
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from audit_kv_cache_equivalence import norm, rotation
from audit_ffn_width import audit_inputs as family_audit
from mqa_training import ROOT,SEEDS,PRIORS,model_new,shared_hash,retained_kv_hash,expand_rows,measure,decision,forward
from position_training import weight_hash
from position_distractors import digest


def audit_inputs(payload):
    checks=family_audit(payload);assert set(payload['prior_hashes'])==set(PRIORS)
    count=0
    for rows in payload['datasets'].values():
        expanded=expand_rows(rows);n=(len(rows[0][0])-2)//2
        assert len(expanded)==len(rows)*(n-1)
        for i,(x,y) in enumerate(expanded):
            old,_=rows[i//(n-1)];assert x[:-1]==old[:-1]
            pairs=list(zip(x[:-2:2],x[1:-2:2]))
            assert y==next(v for k,v in reversed(pairs) if k==x[-1])
            expected=[old[-1]]+sorted({k for k,v in pairs}-{old[-1]})
            assert x[-1]==expected[i%(n-1)];count+=1
    return dict(**checks,expanded_rows_checked=count)


def independent(model,ids,group=None):
    x=model.embedding.weight[ids];n=ids.shape[1];positions=torch.arange(n)+4
    for block in model.blocks:
        z=norm(x,block.norm1);m=block.attn
        q,k,v=[z@getattr(m,t+'_weight').T+getattr(m,t+'_bias') for t in ('q','k','v')]
        q=q.reshape(len(ids),n,2,8).permute(0,2,1,3)
        k,v=[t.reshape(len(ids),n,model.groups,8).permute(0,2,1,3).expand(len(ids),2,n,8) for t in (k,v)]
        q,k=rotation(q,positions),rotation(k,positions)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True).permute(0,2,1,3).reshape(len(ids),n,16)
        x=x+y@m.out_proj.weight.T+m.out_proj.bias
        y=norm(x,block.norm2)@block.linear1.weight.T+block.linear1.bias
        x=x+y.clamp_min(0)@block.linear2.weight.T+block.linear2.bias
    return norm(x,model.final_norm)@model.output.weight.T


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rp=ROOT/'results/mqa_training.json';r=json.loads(rp.read_text());assert r['phase']=='complete'
    for name,h in r['provenance'].items():assert digest(ROOT/('docs/mqa-training-protocol.md' if name=='protocol' else 'experiments/'+name))==h
    ip=ROOT/'results/mqa_training_data.json';assert digest(ip)==r['input_sha256'];payload=json.loads(ip.read_text());inputs=audit_inputs(payload)
    assert {(x['seed'],x['group']) for x in r['runs']}==set(itertools.product(SEEDS,('mha','mqa')))
    for seed in SEEDS:
        pair=[x for x in r['runs'] if x['seed']==seed]
        for key in ('shared_initial_sha256','retained_kv_initial_sha256','stream3_sha256','stream4_sha256'):
            assert len({x[key] for x in pair})==1
        for run in pair:
            torch.manual_seed(seed);model=model_new(run['group'])
            assert weight_hash(model)==run['initial_sha256']
            assert shared_hash(model)==run['shared_initial_sha256'] and retained_kv_hash(model)==run['retained_kv_initial_sha256']
            assert sum(p.numel() for p in model.parameters())==run['parameters']==(5088 if run['group']=='mha' else 4544)
        for n,offset in ((3,1000),(4,2000)):
            g=torch.Generator().manual_seed(seed+offset);stream=hashlib.sha256();repeated=0
            for step in range(1,6001):
                idx=torch.randint(len(payload['datasets'][f'train_{n}'])*(n-1),(64,),generator=g)
                stream.update(json.dumps(idx.tolist()).encode())
                if (n==3 and step%2) or (n==4 and not step%2):repeated+=int((idx%(n-1)==0).sum())
            for run in pair:
                assert run[f'stream{n}_sha256']==stream.hexdigest()
                assert run['query_counts'][str(n)]==dict(repeated=repeated,unique=192000-repeated)
    dp=ROOT/'results/mqa_training_scores.pt';assert digest(dp)==r['scores_sha256'];saved=torch.load(dp,weights_only=True);errors=[]
    for run in r['runs']:
        if run['status']!='trained':continue
        cp=ROOT/run['checkpoint'];assert digest(cp)==run['checkpoint_sha256'];state=torch.load(cp,weights_only=True)
        assert all(v['step'].item()==6000 for v in state['optimizer']['state'].values())
        assert run['tokens']==3456000 and run['attention_pairs']==31488000
        model=model_new(run['group']).eval();model.load_state_dict(state['model'])
        detail=next(d for d in saved if (d['seed'],d['group'])==(run['seed'],run['group']))
        for n in (3,4,6):
            rows=expand_rows(payload['datasets'][f'test_{n}']);x=torch.tensor([x for x,y in rows]);y=torch.tensor([y for x,y in rows])
            with torch.no_grad():scores=torch.cat([independent(model,c)[:,-1] for c in x.split(256)])
            error=(scores-detail['scores'][str(n)]).abs().max().item();assert error<=1e-10;errors.append(error)
            actual=measure(scores,y,x);stored=run['test'][str(n)]
            for key in actual:assert (torch.tensor(actual[key],dtype=torch.float64)-torch.tensor(stored[key],dtype=torch.float64)).abs().max()<=1e-10
            with torch.no_grad():_,cache=forward(model,x[:1],capture=True)
            assert all(list(t.shape)==[1,model.groups,2*n+2,8] for pair in cache for t in pair)
            size=sum(t.untyped_storage().nbytes() for pair in cache for t in pair)
            assert size==run['cache_bytes'][str(n)]==2*model.groups*(2*n+2)*16*8
    assert decision(r['runs'])==r['decision']
    verification=json.loads((ROOT/'results/mqa_training_verification.json').read_text())
    assert verification['passed'] and verification['source_sha256']==digest(ROOT/'experiments/mqa_training.py')
    result=dict(passed=True,evaluations=len(errors),max_logit_error=max(errors),input_checks=inputs,
                report_sha256=digest(rp),source_sha256=digest(Path(__file__)))
    (ROOT/'results/mqa_training_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
