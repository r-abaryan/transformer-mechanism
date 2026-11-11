"""Audit saved example-level predictions, routes, margins and provenance."""
import json
from pathlib import Path
import torch
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    path=ROOT/'results/position_routing.json'
    r=json.loads(path.read_text())
    dp=ROOT/'results/position_routing_details.json'
    assert digest(dp)==r['details_sha256']
    d=json.loads(dp.read_text())
    assert digest(ROOT/'results/position_training.json')==r['prior_sha256']
    assert digest(ROOT/'results/position_training_splits.json')==r['split_sha256']
    assert digest(ROOT/'docs/position-routing-protocol.md')==r['protocol_sha256']
    for name,sha in r['source_sha256'].items(): assert digest(ROOT/'experiments'/name)==sha
    y=torch.tensor(d['labels']); idx=torch.tensor(d['unrelated_indices'])
    assert len(y)==7200 and torch.equal(idx,(torch.arange(7200)+18)%7200)
    assert abs((y==y[idx]).double().mean().item()-r['unrelated_same_answer'])<1e-12
    assert len(r['runs'])==len(d['runs'])==10
    max_error=0.; count=0
    for run,detail in zip(r['runs'],d['runs']):
        assert (run['seed'],run['group'])==(detail['seed'],detail['group'])
        assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256']
        assert run['max_control_error']<1e-10
        assert len(run['references'])==2 and len(run['conditions'])==12
        for name,m in {**run['references'],**run['conditions']}.items():
            row=detail['conditions'][name]
            pred=torch.tensor(row['predictions']); margin=torch.tensor(row['margins'],dtype=torch.float64)
            assert len(pred)==len(margin)==7200 and torch.isfinite(margin).all()
            assert (pred>=0).all() and (pred<19).all()
            now=pred==y
            # Strict margin signs agree with predictions unless an exact tie occurs.
            assert ((margin>0)<=now).all() and ((margin<0)<=~now).all()
            max_error=max(max_error,abs(now.double().mean().item()-m['accuracy']),abs(margin.mean().item()-m['mean_margin']))
            if name in ('4','8'):
                for raw,summary in [('block1_paired_key','block1_paired_key_by_head_row'),('block2_target','block2_target_by_head')]:
                    a=torch.tensor(row[raw],dtype=torch.float64)
                    assert ((a>=0)&(a<=1)).all()
                    max_error=max(max_error,(a.mean(0)-torch.tensor(m[summary],dtype=torch.float64)).abs().max().item())
            else:
                donor,_,recipient,source,mode=name.split('_')
                before=torch.tensor(detail['conditions'][recipient]['predictions'])==y
                assert m['rescued']==int((now&~before).sum()) and m['harmed']==int((~now&before).sum())
                labels=y if source=='matched' else y[idx]
                max_error=max(max_error,abs((pred==labels).double().mean().item()-m['donor_answer_agreement']))
                assert abs(m['accuracy']-run['references'][recipient]['accuracy']-(m['rescued']-m['harmed'])/7200)<1e-12
            count+=1
    assert max_error<1e-12
    result=dict(passed=True,report_sha256=digest(path),audit_source_sha256=digest(Path(__file__)),conditions=count,
                maximum_summary_error=max_error,checkpoint_hashes_unchanged=True,
                scope='Saved predictions, margins, route summaries, rescue/harm and provenance; losses are not independently recomputed by this audit.')
    out=ROOT/'results/position_routing_audit.json'; assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
