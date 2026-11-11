"""Saved-output localization, before choosing one intervention. No model forwards."""
import json
from pathlib import Path

import torch

from copy_position_sites import ROOT, digest, write
from audit_copy_position_exposure import close

OUT=ROOT/'results/copy_site_failure'
PARENT=ROOT/'results/copy_position_sites'
MODELS=((3767,'cross_only'),(3779,'cross_only'),(3793,'cross_only'),(3767,'both'))
POOLS=('train_6','validation_6','test_6','test_8','test_10','uniform_8','uniform_10')


def reduce(rows,z,generated,scalar=False):
    n=len(rows[0]);assert z.shape==(len(rows),n+1,16) and len(generated)==len(rows)
    pred=[[max(range(16),key=v.__getitem__) for v in seq] for seq in z.tolist()] if scalar else z.argmax(-1).tolist()
    records=[]
    for i,(row,answer) in enumerate(zip(rows,generated)):
        gold=row+[2]
        j=next((j for j in range(n+1) if j>=len(answer) or answer[j]!=gold[j]),n+1)
        first_teacher=next((j for j in range(n+1) if pred[i][j]!=gold[j]),n+1)
        assert j==first_teacher, 'First divergence must agree before prefixes differ.'
        if j==n+1:
            assert answer==gold;kind='correct';token=None;margin=None
        else:
            assert j<len(answer) and pred[i][j]==answer[j]
            token=answer[j]
            kind='premature_eos' if token==2 else ('missed_eos' if j==n else 'wrong_content')
            vals=z[i,j].tolist();margin=vals[gold[j]]-max(v for k,v in enumerate(vals) if k!=gold[j])
        records.append(dict(index=i,group='distinct' if len(set(row))==n else 'repeated',
            first_error_position=None if j==n+1 else j+1,kind=kind,predicted=token,
            correct_label=None if j==n+1 else gold[j],correct_margin=margin,
            teacher_content_errors=sum(a!=b for a,b in zip(pred[i][:n],row)),
            teacher_eos_correct=pred[i][n]==2,teacher_predictions=pred[i],generated=answer))
    groups={}
    for g in ('all','distinct','repeated'):
        part=[r for r in records if g=='all' or r['group']==g]
        groups[g]=None if not part else dict(count=len(part),correct=sum(r['kind']=='correct' for r in part),
            categories={k:sum(r['kind']==k for r in part) for k in ('correct','premature_eos','wrong_content','missed_eos')},
            first_error_counts={str(j):sum(r['first_error_position']==j for r in part) for j in range(1,n+2)},
            teacher_content_errors=sum(r['teacher_content_errors'] for r in part),
            teacher_eos_correct=sum(r['teacher_eos_correct'] for r in part))
    return dict(groups=groups,records=records)


def checks():
    rows=[[3,4,5],[4,4,4],[5,6,7],[6,7,8]]
    preds=[[3,4,5,2],[4,2,4,2],[5,3,7,2],[6,7,8,3]]
    z=torch.full((4,4,16),-5.,dtype=torch.float64)
    z.scatter_(-1,torch.tensor(preds)[:,:,None],5.)
    outputs=[preds[0],[4,2],[5,3,7,2],[6,7,8,3,2]]
    result=reduce(rows,z,outputs);close(result,reduce(rows,z,outputs,True))
    assert result['groups']['all']['categories']==dict(correct=1,premature_eos=1,wrong_content=1,missed_eos=1)
    bad=[list(r) for r in outputs];bad[0][0]=4
    try:reduce(rows,z,bad)
    except AssertionError:pass
    else:raise AssertionError('Wrong saved answer escaped detection.')
    assert reduce([[3]*3],z[:1],[[3,4,5,2]])['groups']['distinct'] is None


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'report.json').exists();checks()
    parent=json.loads((PARENT/'report.json').read_text());audit=json.loads((PARENT/'audit.json').read_text())
    assert audit['passed'] and audit['report_sha256']==digest(PARENT/'report.json')
    runs=[next(r for r in parent['confirmation'] if (r['seed'],r['arm'])==pair) for pair in MODELS]
    paths=[Path(__file__),ROOT/'docs/protocols/copy-site-failure.md',PARENT/'report.json',PARENT/'audit.json',PARENT/'data.json']
    paths += [ROOT/r[key] for r in runs for key in ('checkpoint','scores')]
    protected={str(p.relative_to(ROOT)):digest(p) for p in paths}
    assert not (OUT/'plan.json').exists()
    write(OUT/'plan.json',dict(models=MODELS,pools=POOLS,protected=protected,model_forwards=0,training_updates=0,
        selection='Inspect first error position/type and correct-prefix failures; choose at most one intervention in a separate frozen protocol. No metric-selected sweeps.'))
    data=json.loads((PARENT/'data.json').read_text())['pools'];cells=[]
    for run in runs:
        assert digest(ROOT/run['scores'])==run['scores_sha256']
        stored=torch.load(ROOT/run['scores'],weights_only=True)
        for pool in POOLS:
            s=stored[pool];result=reduce(data[pool],s['logits'],s['generated'])
            close(result,reduce(data[pool],s['logits'],s['generated'],True))
            for g,m in result['groups'].items():
                old=run['metrics'][pool][g]
                assert (m is None)==(old is None)
                if m:assert m['correct']/m['count']==old['exact_accuracy']
            cells.append(dict(seed=run['seed'],arm=run['arm'],pool=pool,**result))
    assert len(cells)==len(MODELS)*len(POOLS)==28
    assert all(digest(ROOT/p)==sha for p,sha in protected.items())
    write(OUT/'report.json',dict(cells=cells,checked=True,model_forwards=0,training_updates=0,
        plan_sha256=digest(OUT/'plan.json'),protected_files=len(protected),
        limitation='Scalar and tensor reductions agree on previously audited saved predictions; no new model audit or fresh confirmation.'))
    for c in cells:
        if c['pool'] in ('test_8','test_10'):print(json.dumps({k:c[k] for k in ('seed','arm','pool')}|c['groups']['all']))


if __name__=='__main__':
    torch.set_num_threads(1);main()
