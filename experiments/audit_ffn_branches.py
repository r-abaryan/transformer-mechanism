"""Recompute FFN intervention summaries from saved logits."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1); path=ROOT/'results/ffn_branches.json'; r=json.loads(path.read_text()); assert r['phase']=='complete'
    for n,h in r['source_sha256'].items(): assert digest(ROOT/'experiments'/n)==h
    for name,key in [('canonical_positions_inputs.json','inputs_sha256'),('ffn_branches_scores.json','scores_sha256'),('ffn_branches_permutation.json','permutation_sha256'),('position_training.json','prior_sha256')]: assert digest(ROOT/'results'/name)==r[key]
    assert digest(ROOT/'docs/ffn-branch-protocol.md')==r['protocol_sha256']
    rows=json.loads((ROOT/'results/canonical_positions_inputs.json').read_text()); y=torch.tensor([y for x,y in rows]); size=len(y)
    idx=torch.tensor(json.loads((ROOT/'results/ffn_branches_permutation.json').read_text())); assert torch.equal(idx.sort().values,torch.arange(size))
    assert r['fixed_points']==int((idx==torch.arange(size)).sum()) and abs(r['same_answer']-(y==y[idx]).double().mean().item())<1e-12
    details=json.loads((ROOT/'results/ffn_branches_scores.json').read_text()); assert len(details)==len(r['runs'])==5
    error=0.; count=0
    for run,d in zip(r['runs'],details):
        assert run['seed']==d['seed'] and digest(ROOT/run['checkpoint'])==run['checkpoint_sha256']
        assert max(run['errors'].values())<1e-10 and all(t['error']<1e-6 for t in run['derivative'])
        assert set(run['conditions'])=={'ordinary','sham','zero_first','zero_second','zero_both','permute_first','permute_second'}
        base=torch.tensor(d['scores']['ordinary'],dtype=torch.float64); before=base.argmax(-1)==y
        assert (base-torch.tensor(d['scores']['sham'],dtype=torch.float64)).abs().max()<1e-10
        for name,m in run['conditions'].items():
            s=torch.tensor(d['scores'][name],dtype=torch.float64); assert s.shape==(size,19) and torch.isfinite(s).all()
            pred=s.argmax(-1); now=pred==y; other=s.clone(); other[torch.arange(size),y]=float('-inf')
            expected=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(s,y).item(),mean_margin=(s[torch.arange(size),y]-other.max(1).values).mean().item(),rescued=int((now&~before).sum()),harmed=int((~now&before).sum()))
            if name.startswith('permute'): expected['donor_answer_agreement']=(pred==y[idx]).double().mean().item()
            for k,v in expected.items(): error=max(error,abs(v-m[k]))
            count+=1
    assert error<1e-12
    result=dict(passed=True,conditions=count,maximum_metric_error=error,report_sha256=digest(path),audit_source_sha256=digest(Path(__file__)),scope='Saved-logit summaries and provenance; math checks verified as recorded, not independently rerun.')
    out=ROOT/'results/ffn_branches_audit.json'; assert not out.exists(); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
