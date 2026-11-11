"""Recompute query/key diagnostics from saved logits."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    path=ROOT/'results/query_key.json'; r=json.loads(path.read_text())
    for file,key in [('query_key_scores.json','scores_sha256'),('position_training_splits.json','split_sha256'),('position_routing.json','prior_sha256')]: assert digest(ROOT/'results'/file)==r[key]
    for n,h in r['source_sha256'].items(): assert digest(ROOT/'experiments'/n)==h
    assert digest(ROOT/'docs/query-key-protocol.md')==r['protocol_sha256']
    rows=json.loads((ROOT/'results/position_training_splits.json').read_text())['test']; y=torch.tensor([y for x,y in rows])
    details=json.loads((ROOT/'results/query_key_scores.json').read_text()); assert len(details)==len(r['runs'])==5
    error=0.; count=0
    for run,d in zip(r['runs'],details):
        assert run['seed']==d['seed'] and digest(ROOT/run['checkpoint'])==run['checkpoint_sha256']
        assert run['max_control_error']<1e-10
        assert set(run['conditions'])=={f'{a}_to_{b}_{m}' for a,b in ((4,8),(8,4)) for m in ('Q','K','QK')}
        for name,m in {**run['references'],**run['conditions']}.items():
            s=torch.tensor(d['scores'][name],dtype=torch.float64); assert s.shape==(7200,19) and torch.isfinite(s).all()
            pred=s.argmax(-1); now=pred==y; other=s.clone(); other[torch.arange(7200),y]=float('-inf')
            expected=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(s,y).item(),mean_margin=(s[torch.arange(7200),y]-other.max(1).values).mean().item())
            if name not in ('4','8'):
                recipient=name.split('_')[2]; before=torch.tensor(d['scores'][recipient],dtype=torch.float64).argmax(-1)==y
                expected.update(rescued=int((now&~before).sum()),harmed=int((~now&before).sum()))
            for k,v in expected.items(): error=max(error,abs(v-m[k]))
            count+=1
        for prefix,value in run['accuracy_interaction'].items():
            recipient=prefix.split('_')[2]; c=run['conditions']
            expected=c[prefix+'_QK']['accuracy']-c[prefix+'_Q']['accuracy']-c[prefix+'_K']['accuracy']+run['references'][recipient]['accuracy']
            error=max(error,abs(value-expected))
    assert error<1e-12
    result=dict(passed=True,conditions=count,maximum_metric_error=error,report_sha256=digest(path),audit_source_sha256=digest(Path(__file__)),scope='Recomputed metrics from saved logits and verified provenance; no independent forward rerun.')
    out=ROOT/'results/query_key_audit.json'; assert not out.exists(); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
