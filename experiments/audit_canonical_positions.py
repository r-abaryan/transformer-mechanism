"""Audit canonical-position data, logits, adapter and criteria."""
import itertools
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from canonical_positions import canonical_positions
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    path=ROOT/'results/canonical_positions.json'; r=json.loads(path.read_text()); assert r['phase']=='complete'
    for n,h in r['source_sha256'].items(): assert digest(ROOT/'experiments'/n)==h
    assert digest(ROOT/'docs/canonical-position-protocol.md')==r['protocol_sha256']
    assert digest(ROOT/'results/position_training.json')==r['prior_sha256']
    def key(x): return tuple(sorted(zip(x[:6:2],x[1:6:2])))
    excluded=set()
    for name,h in r['exclusion_hashes'].items():
        p=ROOT/'results'/name; assert digest(p)==h; d=json.loads(p.read_text())
        for rows in (d.values() if isinstance(d,dict) else [d]): excluded.update(key(x) for x,y in rows)
    assert len(excluded)==5200
    sp=ROOT/'results/canonical_positions_inputs.json'; assert digest(sp)==r['inputs_sha256']
    rows=json.loads(sp.read_text()); maps={key(x) for x,y in rows}
    assert len(rows)==7200 and len(maps)==400 and not maps&excluded
    for start in range(0,7200,18):
        m=key(rows[start][0]); expected={(tuple(t for pair in order for t in pair)+(18,q),v) for order in itertools.permutations(m) for q,v in m}
        assert {(tuple(x),y) for x,y in rows[start:start+18]}==expected
    assert all(sum(y==v for x,y in rows)==1200 for v in range(12,18))
    assert all(sum(x[:6:2].index(x[-1])==i for x,y in rows)==2400 for i in range(3))
    assert all(sum(tuple(sorted(v for k,v in m))==vs for m in maps)==20 for vs in itertools.combinations(range(12,18),3))
    for shift in (-10,0,4,8,12,100): assert torch.equal(canonical_positions((torch.arange(8)+shift)[None]),torch.arange(4,12)[None])
    for bad in (torch.tensor([[0,2,3]]),torch.tensor([[0,0,1]])):
        try: canonical_positions(bad)
        except ValueError: pass
        else: raise AssertionError('Invalid positions accepted')
    dp=ROOT/'results/canonical_positions_scores.json'; assert digest(dp)==r['scores_sha256']; details=json.loads(dp.read_text())
    assert len(details)==len(r['runs'])==len(r['checkpoints'])==5
    y=torch.tensor([y for x,y in rows]); error=0.; identity=0.; count=0
    for run,d,cp in zip(r['runs'],details,r['checkpoints']):
        assert run['seed']==d['seed']==cp['seed'] and digest(ROOT/cp['checkpoint'])==cp['checkpoint_sha256']
        assert run['max_control_error']<1e-10
        assert set(run['conditions'])=={f'{s}_{m}' for s in (0,4,8,12) for m in ('ordinary','canonical')}
        ref=torch.tensor(d['scores']['4_ordinary'],dtype=torch.float64)
        for name,m in run['conditions'].items():
            s=torch.tensor(d['scores'][name],dtype=torch.float64); assert s.shape==(7200,19) and torch.isfinite(s).all()
            if name.endswith('canonical'): identity=max(identity,(s-ref).abs().max().item())
            correct=s.argmax(-1)==y; other=s.clone(); other[torch.arange(7200),y]=float('-inf')
            measured=dict(accuracy=correct.double().mean().item(),loss=F.cross_entropy(s,y).item(),mean_margin=(s[torch.arange(7200),y]-other.max(1).values).mean().item(),every_order_query_correct=correct.reshape(400,18).all(1).double().mean().item())
            for k,v in measured.items(): error=max(error,abs(v-m[k]))
            count+=1
    assert error<1e-12 and identity<1e-10
    c=r['criteria']; assert c['offset_identity'] and r['gapped_positions_rejected']
    assert c['canonical_all_99']==all(run['conditions']['0_canonical']['accuracy']>=.99 for run in r['runs'])
    assert c['zero_preserved']==all(run['conditions']['0_ordinary']['accuracy']-run['conditions']['0_canonical']['accuracy']<=.01 for run in r['runs'])
    result=dict(passed=True,conditions=count,maximum_metric_error=error,canonical_logit_identity_error=identity,report_sha256=digest(path),audit_source_sha256=digest(Path(__file__)),scope='Saved-logit audit, provenance, mapping isolation and adapter boundary checks; no independent model implementation.')
    out=ROOT/'results/canonical_positions_audit.json'; assert not out.exists(); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
