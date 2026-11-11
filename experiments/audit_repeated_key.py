"""Audit mapping isolation, matched streams and explicit checkpoint reloads."""
import hashlib
import collections
import itertools
import json
from pathlib import Path
import torch
from activation_paths import forward as baseline_forward
from torch.nn import functional as F
from repeated_key import ROOT, SEEDS, measure, decision
from audit_positional_comparison import independent
from reserved_cycle import model_new
from position_training import weight_hash
from position_distractors import digest


def audit_inputs(payload):
    seen=set(); counts={}; heuristic={}
    for name,rows in payload['datasets'].items():
        n=int(name.split('_')[1]); signatures=[]
        for start in range(0,len(rows),36):
            chunk=rows[start:start+36]; identities=set()
            assert len(chunk)==36 and len({tuple(x) for x,y in chunk})==36
            assert collections.Counter(y for x,y in chunk)=={v:6 for v in range(12,18)}
            for x,y in chunk:
                assert len(x)==2*n+2 and x[-2]==18
                pairs=list(zip(x[:-2:2],x[1:-2:2])); q=x[-1]
                assert all(0<=k<12 and 12<=v<18 for k,v in pairs)
                counts_k=collections.Counter(k for k,v in pairs)
                assert counts_k[q]==2 and len(counts_k)==n-1 and sorted(counts_k.values())==[1]*(n-2)+[2]
                assert len({v for k,v in pairs})==n
                matches=[v for k,v in pairs if k==q]
                assert y==matches[-1] and y!=matches[0]
                identities.add(min(tuple(sorted((k,12+(v-12+t)%6) for k,v in pairs)) for t in range(6)))
            assert len(identities)==1
            signature=identities.pop(); assert signature not in seen; seen.add(signature); signatures.append(signature)
            for j in range(0,36,2):
                x,y=chunk[j]; z,w=chunk[j+1]
                assert list(zip(x[:-2:2],x[1:-2:2]))[::-1]==list(zip(z[:-2:2],z[1:-2:2]))
                assert x[-1]==z[-1] and y!=w
        assert signatures==[tuple(tuple(pair) for pair in f) for f in payload['families'][name]]
        counts[name]=len(signatures)
        heuristic[name]=sum(x[-3]==y for x,y in rows)/len(rows)
    assert counts==dict(train_3=800,validation_3=100,test_3=200,train_4=800,validation_4=100,test_4=200,test_6=200)
    return dict(family_counts=counts,last_pair_heuristic=heuristic,first_match_accuracy=0.0)


def main():
    torch.set_num_threads(1)
    rp=ROOT/'results/repeated_key.json'; r=json.loads(rp.read_text()); assert r['phase']=='complete'
    for n,h in r['provenance'].items(): assert digest(ROOT/('docs/repeated-key-protocol.md' if n=='protocol' else 'experiments/'+n))==h
    ip=ROOT/'results/repeated_key_data.json'; assert digest(ip)==r['input_sha256']; payload=json.loads(ip.read_text()); counts=audit_inputs(payload)
    assert {(x['seed'],x['group']) for x in r['runs']}==set(itertools.product(SEEDS,('baseline','rope')))
    for seed in SEEDS:
        pair=[x for x in r['runs'] if x['seed']==seed]
        for key in ('initial_sha256','stream3_sha256','stream4_sha256'): assert len({x[key] for x in pair})==1
        torch.manual_seed(seed); assert weight_hash(model_new())==pair[0]['initial_sha256']
        for n,offset in ((3,1000),(4,2000)):
            generator=torch.Generator().manual_seed(seed+offset); h=hashlib.sha256()
            for _ in range(6000):
                h.update(json.dumps(torch.randint(len(payload['datasets'][f'train_{n}']),(64,),generator=generator).tolist()).encode())
            assert h.hexdigest()==pair[0][f'stream{n}_sha256']
    dp=ROOT/'results/repeated_key_scores.pt'; assert digest(dp)==r['scores_sha256']; saved=torch.load(dp,weights_only=True); errors=[]
    for run in r['runs']:
        if run['status']!='trained': continue
        cp=ROOT/run['checkpoint']; assert digest(cp)==run['checkpoint_sha256']; state=torch.load(cp,weights_only=True)
        assert all(v['step'].item()==6000 for v in state['optimizer']['state'].values())
        assert run['tokens']==3456000
        assert run['attention_pairs']==31488000
        model=model_new().eval(); model.load_state_dict(state['model'])
        detail=next(d for d in saved if (d['seed'],d['group'])==(run['seed'],run['group']))
        for n in (3,4,6):
            rows=payload['datasets'][f'test_{n}']; x=torch.tensor([x for x,y in rows]); y=torch.tensor([y for x,y in rows])
            with torch.no_grad(): s=torch.cat([independent(model,c,run['group']) for c in x.split(256)])
            err=(s-detail['scores'][str(n)]).abs().max().item(); assert err<1e-10; errors.append(err)
            actual=measure(s,y,x); stored=run['test'][str(n)]
            for key in actual:
                assert (torch.tensor(actual[key],dtype=torch.float64)-torch.tensor(stored[key],dtype=torch.float64)).abs().max()<1e-10
            assert measure(detail['scores'][str(n)],y,x)==stored
    assert decision(r['runs'])==r['decision']
    verification=json.loads((ROOT/'results/positional_comparison_verification.json').read_text())
    assert verification['passed'] and verification['source_sha256']==digest(ROOT/'experiments/positional_comparison.py')
    result=dict(passed=True,evaluations=len(errors),max_logit_error=max(errors),input_checks=counts,report_sha256=digest(rp),source_sha256=digest(Path(__file__)))
    (ROOT/'results/repeated_key_audit.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
