"""Audit mapping isolation, matched streams and explicit checkpoint reloads."""
import hashlib
import collections
import itertools
import json
from pathlib import Path
import torch
from activation_paths import forward as baseline_forward
from torch.nn import functional as F
from query_exposure import ROOT, SEEDS, measure, decision, model_new, shared_hash, expand_rows, training_batch
from audit_ffn_width import audit_inputs as width_input_audit
from audit_gelu_comparison import independent as activation_independent


from position_training import weight_hash
from position_distractors import digest


def audit_inputs(payload):
    checks=width_input_audit(payload)
    assert set(payload['prior_hashes'])=={'repeated_key_data.json','gelu_comparison_data.json','ffn_width_data.json'}
    changed=0
    for rows in payload['datasets'].values():
        expanded=expand_rows(rows)
        n=(len(rows[0][0])-2)//2
        for i,(x,y) in enumerate(expanded):
            original,label=rows[i//(n-1)]; assert x[:-1]==original[:-1]
            assert y==next(v for k,v in reversed(list(zip(x[:-2:2],x[1:-2:2]))) if k==x[-1])
            if i%(n-1):assert x[-1]!=original[-1] and y!=label;changed+=1
            else:assert x==original and y==label
        tx=torch.tensor([x for x,y in expanded]);ty=torch.tensor([y for x,y in expanded])
        bx,by=training_batch(tx,ty,'repeated')
        expected=torch.tensor([x for x,y in rows]).repeat_interleave(n-1,0)
        labels=torch.tensor([y for x,y in rows]).repeat_interleave(n-1,0)
        assert torch.equal(bx,expected) and torch.equal(by,labels)
        ax,ay=training_batch(tx,ty,'all');assert torch.equal(ax,tx) and torch.equal(ay,ty)
    return dict(**checks,changed_unique_examples=changed,training_context_matching=True)


def independent(model,ids,group):
    return activation_independent(model,ids,'relu')


def main():
    torch.set_num_threads(1)
    rp=ROOT/'results/query_exposure.json'; r=json.loads(rp.read_text()); assert r['phase']=='complete'
    for n,h in r['provenance'].items(): assert digest(ROOT/('docs/query-exposure-protocol.md' if n=='protocol' else 'experiments/'+n))==h
    ip=ROOT/'results/query_exposure_data.json'; assert digest(ip)==r['input_sha256']; payload=json.loads(ip.read_text()); counts=audit_inputs(payload)
    assert {(x['seed'],x['group']) for x in r['runs']}==set(itertools.product(SEEDS,('repeated','all')))
    for seed in SEEDS:
        pair=[x for x in r['runs'] if x['seed']==seed]
        for key in ('initial_sha256','stream3_sha256','stream4_sha256'): assert len({x[key] for x in pair})==1
        for run in pair:
            torch.manual_seed(seed); m=model_new(run['group'])
            assert weight_hash(m)==run['initial_sha256'] and shared_hash(m)==run['shared_initial_sha256']
            assert sum(p.numel() for p in m.parameters())==run['parameters']==5088
        for n,offset in ((3,1000),(4,2000)):
            generator=torch.Generator().manual_seed(seed+offset); h=hashlib.sha256()
            count=0
            for step in range(1,6001):
                idx=torch.randint(len(payload['datasets'][f'train_{n}'])*(n-1),(64,),generator=generator)
                h.update(json.dumps(idx.tolist()).encode())
                if (n==3 and step%2==1) or (n==4 and step%2==0):count+=int((idx%(n-1)==0).sum())
            for run in pair:
                actual=count if run['group']=='all' else 192000
                assert run['query_counts'][str(n)]==dict(repeated=actual,unique=192000-actual)
            assert h.hexdigest()==pair[0][f'stream{n}_sha256']
    dp=ROOT/'results/query_exposure_scores.pt'; assert digest(dp)==r['scores_sha256']; saved=torch.load(dp,weights_only=True); errors=[]
    for run in r['runs']:
        if run['status']!='trained': continue
        cp=ROOT/run['checkpoint']; assert digest(cp)==run['checkpoint_sha256']; state=torch.load(cp,weights_only=True)
        assert all(v['step'].item()==6000 for v in state['optimizer']['state'].values())
        assert run['tokens']==3456000
        assert run['attention_pairs']==31488000
        model=model_new(run['group']).eval(); model.load_state_dict(state['model'])
        detail=next(d for d in saved if (d['seed'],d['group'])==(run['seed'],run['group']))
        for n in (3,4,6):
            rows=expand_rows(payload['datasets'][f'test_{n}']); x=torch.tensor([x for x,y in rows]); y=torch.tensor([y for x,y in rows])
            with torch.no_grad(): s=torch.cat([independent(model,c,run['group']) for c in x.split(256)])
            err=(s-detail['scores'][str(n)]).abs().max().item(); assert err<1e-10; errors.append(err)
            actual=measure(s,y,x); stored=run['test'][str(n)]
            for key in actual:
                assert (torch.tensor(actual[key],dtype=torch.float64)-torch.tensor(stored[key],dtype=torch.float64)).abs().max()<1e-10
            assert measure(detail['scores'][str(n)],y,x)==stored
    assert decision(r['runs'])==r['decision']
    verification=json.loads((ROOT/'results/ffn_width_verification.json').read_text())
    assert verification['passed'] and verification['source_sha256']==digest(ROOT/'experiments/ffn_width.py')
    result=dict(passed=True,evaluations=len(errors),max_logit_error=max(errors),input_checks=counts,report_sha256=digest(rp),source_sha256=digest(Path(__file__)))
    (ROOT/'results/query_exposure_audit.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
