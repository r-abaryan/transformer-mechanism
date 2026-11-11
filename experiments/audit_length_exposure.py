"""Audit mapping isolation, matched streams and explicit checkpoint reloads."""
import collections
import itertools
import json
from pathlib import Path
import torch
from activation_paths import forward
from length_exposure import ROOT, SEEDS, excluded, identity, measure, decision
from reserved_cycle import model_new
from position_distractors import digest


def audit_inputs(payload):
    old,hashes=excluded(); assert hashes==payload['exclusion_hashes']
    assert old=={tuple(tuple(pair) for pair in m) for m in payload['excluded_mappings']}
    seen={}; counts={}
    for name,rows in payload['datasets'].items():
        n=int(name.split('_')[1]); groups=collections.defaultdict(list)
        for x,y in rows:
            m=identity(x); assert len(m)==n and len(set(k for k,v in m))==n and len(set(v for k,v in m))==n
            assert x[-2]==18 and dict(m)[x[-1]]==y
            groups[m].append((tuple(x),y))
        assert not set(groups)&old
        for m,variants in groups.items():
            assert len(variants)==6*n and len(set(variants))==6*n
            orders=collections.Counter(x[:-2] for x,y in variants); assert len(orders)==6 and set(orders.values())=={n}
            assert collections.Counter(x[-1] for x,y in variants)==collections.Counter({k:6 for k,v in m})
        assert set(collections.Counter(y for x,y in rows).values())=={len(rows)//6}
        assert set(collections.Counter(x[:-2:2].index(x[-1]) for x,y in rows).values())=={len(rows)//n}
        assert not set(groups)&seen.get(n,set()); seen.setdefault(n,set()).update(groups); counts[name]=len(groups)
    assert counts==dict(train_3=1600,validation_3=200,test_3=400,train_4=1605,validation_4=210,test_4=420,test_6=420)
    return counts


def main():
    torch.set_num_threads(1)
    rp=ROOT/'results/length_exposure.json'; r=json.loads(rp.read_text()); assert r['phase']=='complete'
    for n,h in r['provenance'].items(): assert digest(ROOT/('docs/length-exposure-protocol.md' if n=='protocol' else 'experiments/'+n))==h
    ip=ROOT/'results/length_exposure_inputs.json'; assert digest(ip)==r['input_sha256']; payload=json.loads(ip.read_text()); counts=audit_inputs(payload)
    assert {(x['seed'],x['group']) for x in r['runs']}==set(itertools.product(SEEDS,('baseline','mixed')))
    for seed in SEEDS:
        pair=[x for x in r['runs'] if x['seed']==seed]
        for key in ('initial_sha256','stream3_sha256','stream4_sha256'): assert len({x[key] for x in pair})==1
    dp=ROOT/'results/length_exposure_scores.pt'; assert digest(dp)==r['scores_sha256']; saved=torch.load(dp,weights_only=True); errors=[]
    for run in r['runs']:
        if run['status']!='trained': continue
        cp=ROOT/run['checkpoint']; assert digest(cp)==run['checkpoint_sha256']; state=torch.load(cp,weights_only=True)
        assert all(v['step'].item()==6000 for v in state['optimizer']['state'].values())
        assert run['tokens']==(3072000 if run['group']=='baseline' else 3456000)
        assert run['attention_pairs']==(24576000 if run['group']=='baseline' else 31488000)
        model=model_new().eval(); model.load_state_dict(state['model'])
        detail=next(d for d in saved if (d['seed'],d['group'])==(run['seed'],run['group']))
        for n in (3,4,6):
            rows=payload['datasets'][f'test_{n}']; x=torch.tensor([x for x,y in rows]); y=torch.tensor([y for x,y in rows])
            with torch.no_grad(): s=torch.cat([forward(model,c,4)[0] for c in x.split(256)])
            err=(s-detail['scores'][str(n)]).abs().max().item(); assert err<1e-10; errors.append(err)
            actual=measure(s,y,x); stored=run['test'][str(n)]
            for key in actual:
                assert (torch.tensor(actual[key],dtype=torch.float64)-torch.tensor(stored[key],dtype=torch.float64)).abs().max()<1e-10
            assert measure(detail['scores'][str(n)],y,x)==stored
    assert decision(r['runs'])==r['decision']
    result=dict(passed=True,evaluations=len(errors),max_logit_error=max(errors),mapping_counts=counts,report_sha256=digest(rp),source_sha256=digest(Path(__file__)))
    (ROOT/'results/length_exposure_audit.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
