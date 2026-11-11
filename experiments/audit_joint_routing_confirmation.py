"""Audit fresh mappings, saved logits, criteria and immutable provenance."""
import itertools
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    path=ROOT/'results/joint_routing_confirmation.json'; r=json.loads(path.read_text())
    assert r['phase']=='complete' and len(r['runs'])==5
    for name,h in r['source_sha256'].items(): assert digest(ROOT/'experiments'/name)==h
    assert digest(ROOT/'docs/joint-routing-confirmation-protocol.md')==r['protocol_sha256']
    assert digest(ROOT/'results/position_training.json')==r['prior_sha256']
    def key(ids): return tuple(sorted((ids[i],ids[i+1]) for i in (0,2,4)))
    excluded=set()
    for name,h in r['exclusion_hashes'].items():
        p=ROOT/'results'/name; assert digest(p)==h
        excluded.update(key(x) for rows in json.loads(p.read_text()).values() for x,y in rows)
    assert len(excluded)==4800
    sp=ROOT/'results/joint_routing_confirmation_inputs.json'; assert digest(sp)==r['inputs_sha256']
    rows=json.loads(sp.read_text()); assert len(rows)==7200
    maps={key(x) for x,y in rows}; assert len(maps)==400 and not maps&excluded
    for start in range(0,7200,18):
        m=key(rows[start][0]); expected={(tuple(t for pair in order for t in pair)+(18,q),v) for order in itertools.permutations(m) for q,v in m}
        assert {(tuple(x),y) for x,y in rows[start:start+18]}==expected
    assert all(sum(y==v for x,y in rows)==1200 for v in range(12,18))
    assert all(sum(x[:6:2].index(x[-1])==i for x,y in rows)==2400 for i in range(3))
    assert all(sum(tuple(sorted(v for k,v in m))==vs for m in maps)==20 for vs in itertools.combinations(range(12,18),3))
    dp=ROOT/'results/joint_routing_confirmation_scores.json'; assert digest(dp)==r['scores_sha256']
    details=json.loads(dp.read_text()); assert len(details)==5
    y=torch.tensor([y for x,y in rows]); error=0.; count=0
    for run,d,checkpoint in zip(r['runs'],details,r['checkpoints']):
        assert run['seed']==d['seed']==checkpoint['seed']
        assert digest(ROOT/checkpoint['checkpoint'])==checkpoint['checkpoint_sha256']
        assert run['max_control_error']<1e-10
        assert set(run['conditions'])=={'4_ordinary','8_ordinary','12_ordinary','8_joint','12_joint'}
        for name,m in run['conditions'].items():
            s=torch.tensor(d['scores'][name],dtype=torch.float64); assert s.shape==(7200,19) and torch.isfinite(s).all()
            now=s.argmax(-1)==y; shift=name.split('_')[0]
            before=torch.tensor(d['scores'][shift+'_ordinary'],dtype=torch.float64).argmax(-1)==y
            other=s.clone(); other[torch.arange(7200),y]=float('-inf')
            observed=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(s,y).item(),
                          mean_margin=(s[torch.arange(7200),y]-other.max(1).values).mean().item(),
                          rescued=int((now&~before).sum()),harmed=int((~now&before).sum()),
                          every_order_query_correct=now.reshape(400,18).all(1).double().mean().item())
            for k,v in observed.items(): error=max(error,abs(v-m[k]))
            count+=1
    assert error<1e-12
    for shift,threshold,min_gain in (('8',.95,.10),('12',.90,.05)):
        c=r['criteria'][shift]
        gain=sum(run['conditions'][shift+'_joint']['accuracy']-run['conditions'][shift+'_ordinary']['accuracy'] for run in r['runs'])/5
        all_pass=all(run['conditions'][shift+'_joint']['accuracy']>=threshold for run in r['runs'])
        assert abs(gain-c['mean_gain'])<1e-12 and c['joint_all_accuracy']==all_pass and c['gain_pass']==(gain>=min_gain)
        assert c['joint_pass']==(all_pass and gain>=min_gain)
        assert c['ordinary_all_90']==all(run['conditions'][shift+'_ordinary']['accuracy']>=.9 for run in r['runs'])
    result=dict(passed=True,report_sha256=digest(path),audit_source_sha256=digest(Path(__file__)),conditions=count,
                maximum_metric_error=error,mapping_isolation=True,criteria_verified=True,
                scope='Saved-logit metric recomputation, dataset and provenance checks; shared PyTorch, no independent intervention forward rerun.')
    out=ROOT/'results/joint_routing_confirmation_audit.json'; assert not out.exists(); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
