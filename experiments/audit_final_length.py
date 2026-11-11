"""Audit length inputs, saved logits, rank metrics and provenance."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    path=ROOT/'results/final_length.json'; r=json.loads(path.read_text()); assert r['phase']=='complete'
    for name,h in r['source_sha256'].items(): assert digest(ROOT/'experiments'/name)==h
    assert digest(ROOT/'docs/final-length-protocol.md')==r['protocol_sha256']
    assert digest(ROOT/'results/position_training.json')==r['prior_sha256']
    def key(x): return tuple(sorted(zip(x[:-2:2],x[1:-2:2])))
    excluded=set()
    for name,h in r['exclusion_hashes'].items():
        p=ROOT/'results'/name; assert digest(p)==h; d=json.loads(p.read_text())
        for rows in (d.values() if isinstance(d,dict) else [d]): excluded.update(key(x) for x,y in rows)
    assert len(excluded)==5600
    sp=ROOT/'results/final_length_inputs.json'; assert digest(sp)==r['inputs_sha256']; data=json.loads(sp.read_text())
    assert set(data)=={'3','4','6'}
    for name,rows in data.items():
        n=int(name); assert len(rows)==420*6*n
        maps={key(x) for x,y in rows}; assert len(maps)==420 and not maps&excluded
        for start in range(0,len(rows),6*n):
            block=rows[start:start+6*n]; m=key(block[0][0]); orders={tuple(x[:-2]) for x,y in block}; assert len(orders)==6
            assert all(key(x)==m and dict(m)[x[-1]]==y and x[-2]==18 and len(set(x[1:-2:2]))==n for x,y in block)
            assert {(tuple(x),y) for x,y in block}=={(order+(18,q),v) for order in orders for q,v in m}
        assert all(sum(y==v for x,y in rows)==len(rows)//6 for v in range(12,18))
        assert all(sum(x[:-2:2].index(x[-1])==i for x,y in rows)==len(rows)//n for i in range(n))
    dp=ROOT/'results/final_length_scores.json'; assert digest(dp)==r['scores_sha256']; ds=json.loads(dp.read_text())
    assert len(ds)==len(r['runs'])==len(r['checkpoints'])==5
    error=0.; count=0
    for run,d,cp in zip(r['runs'],ds,r['checkpoints']):
        assert run['seed']==d['seed']==cp['seed'] and digest(ROOT/cp['checkpoint'])==cp['checkpoint_sha256']
        assert run['max_control_error']<1e-10 and set(run['conditions'])==set(data)
        for name,m in run['conditions'].items():
            rows=data[name]; n=int(name); y=torch.tensor([y for x,y in rows]); rank=torch.tensor([x[:-2:2].index(x[-1]) for x,y in rows])
            s=torch.tensor(d['scores'][name],dtype=torch.float64); assert s.shape==(len(rows),19) and torch.isfinite(s).all()
            correct=s.argmax(-1)==y; other=s.clone(); other[torch.arange(len(rows)),y]=float('-inf')
            expected=dict(accuracy=correct.double().mean().item(),loss=F.cross_entropy(s,y).item(),mean_margin=(s[torch.arange(len(rows)),y]-other.max(1).values).mean().item(),all_sampled_variants_correct=correct.reshape(420,6*n).all(1).double().mean().item())
            for k,v in expected.items(): error=max(error,abs(v-m[k]))
            for i,v in enumerate(m['accuracy_by_rank']): error=max(error,abs(v-correct[rank==i].double().mean().item()))
            assert m['sequences']==len(rows); count+=1
    assert error<1e-12
    for n,t in (('3',.99),('4',.9),('6',.9)): assert r['criteria'][n]==all(run['conditions'][n]['accuracy']>=t for run in r['runs'])
    result=dict(passed=True,conditions=count,maximum_metric_error=error,report_sha256=digest(path),audit_source_sha256=digest(Path(__file__)),scope='Saved-logit metrics, rank breakdown, mapping isolation, sampling and provenance; no independent forward implementation.')
    out=ROOT/'results/final_length_audit.json'; assert not out.exists(); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
