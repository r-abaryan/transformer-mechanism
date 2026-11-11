"""Recompute donor-condition metrics from saved full logits and labels."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    path=ROOT/'results/donor_slots.json'; r=json.loads(path.read_text())
    for file,key in [('donor_slots_scores.json','scores_sha256'),('donor_slots_indices.json','indices_sha256'),('position_training_splits.json','split_sha256'),('position_routing.json','prior_sha256')]:
        assert digest(ROOT/'results'/file)==r[key]
    for name,sha in r['source_sha256'].items(): assert digest(ROOT/'experiments'/name)==sha
    assert digest(ROOT/'docs/donor-slot-protocol.md')==r['protocol_sha256']
    rows=json.loads((ROOT/'results/position_training_splits.json').read_text())['test']
    x=torch.tensor([x for x,y in rows]); y=torch.tensor([y for x,y in rows])
    ranks=torch.tensor([ids[:6:2].index(ids[-1]) for ids,target in rows])
    maps=[tuple(sorted(zip(ids[:6:2],ids[1:6:2]))) for ids,target in rows]
    indices=json.loads((ROOT/'results/donor_slots_indices.json').read_text())
    assert set(indices)=={'same_slot_same_answer','same_slot_different_answer','different_slot_same_answer','different_slot_different_answer'}
    for name,selection in indices.items():
        assert len(selection)==7200
        idx=torch.tensor(selection); assert ((idx>=0)&(idx<7200)).all()
        assert all(maps[i]!=maps[j] for i,j in enumerate(selection))
        assert ((ranks==ranks[idx])==name.startswith('same_slot')).all()
        assert ((y==y[idx])==name.endswith('_same_answer')).all()
        slot_y=x[torch.arange(7200),2*ranks[idx]+1]
        expected=r['donors'][name]
        assert expected['unique_donor_rows']==len(set(selection))
        assert expected['donor_rank_counts']==torch.bincount(ranks[idx],minlength=3).tolist()
        assert abs(expected['slot_value_equals_donor_answer']-(slot_y==y[idx]).double().mean().item())<1e-12
    prior=json.loads((ROOT/'results/position_routing.json').read_text())
    pp=ROOT/'results/position_routing_details.json'; assert digest(pp)==prior['details_sha256']
    prior_details=json.loads(pp.read_text())['runs']
    details=json.loads((ROOT/'results/donor_slots_scores.json').read_text())
    assert len(details)==len(r['runs'])==10
    error=0.; count=0
    for run,d,old in zip(r['runs'],details,prior_details):
        assert (run['seed'],run['group'])==(d['seed'],d['group'])==(old['seed'],old['group'])
        assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256'] and run['max_control_error']<1e-10
        before=torch.tensor(d['scores']['baseline'],dtype=torch.float64).argmax(-1)==y
        donor_pred=torch.tensor(old['conditions']['4']['predictions'])
        assert set(run['conditions'])==set(indices)|{'matched'}
        for name,m in run['conditions'].items():
            s=torch.tensor(d['scores'][name],dtype=torch.float64); assert s.shape==(7200,19) and torch.isfinite(s).all()
            idx=torch.tensor(indices[name]) if name!='matched' else torch.arange(7200)
            pred=s.argmax(-1); now=pred==y; slot_y=x[torch.arange(7200),2*ranks[idx]+1]; distinct=slot_y!=y[idx]
            other=s.clone(); other[torch.arange(7200),y]=float('-inf')
            expected=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(s,y).item(),
                          mean_margin=(s[torch.arange(7200),y]-other.max(1).values).mean().item(),
                          rescued=int((now&~before).sum()),harmed=int((~now&before).sum()),
                          donor_answer_agreement=(pred==y[idx]).double().mean().item(),
                          recipient_donor_slot_agreement=(pred==slot_y).double().mean().item(),
                          donor_original_accuracy=(donor_pred[idx]==y[idx]).double().mean().item(),
                          distinct_labels_count=int(distinct.sum()),
                          distinct_labels_slot_agreement=(pred[distinct]==slot_y[distinct]).double().mean().item() if distinct.any() else None,
                          distinct_labels_donor_agreement=(pred[distinct]==y[idx][distinct]).double().mean().item() if distinct.any() else None)
            for k,v in expected.items():
                if v is None: assert m[k] is None
                else: error=max(error,abs(v-m[k]))
            count+=1
    assert error<1e-12
    result=dict(passed=True,report_sha256=digest(path),audit_source_sha256=digest(Path(__file__)),conditions=count,maximum_metric_error=error,
                scope='All intervention summaries recomputed from saved logits; donor baseline checked against hashed prior predictions. No independent forward implementation.')
    out=ROOT/'results/donor_slots_audit.json'; assert not out.exists(); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
