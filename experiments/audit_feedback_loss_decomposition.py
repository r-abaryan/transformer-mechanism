"""Scalar log-partition, independent answer transitions and exact contribution sums."""
import json
import math

import torch
import feedback_loss_decomposition as study
import audit_query_feedback_training as previous


def scalar(z,y):
    high=max(z);den=math.fsum(math.exp(v-high) for v in z);prediction=max(range(6),key=lambda k:z[k])
    return dict(ce=(high-z[y])+math.log(den),true_probability=math.exp(z[y]-high)/den,
                winner_confidence=1/den,true_margin=z[y]-max(v for k,v in enumerate(z) if k!=y),
                correct=prediction==y,prediction=prediction)


def reconstruct(a,b):
    assert a['logits'].shape==b['logits'].shape and a['logits'].shape[:-1]==a['labels'].shape==a['repeat'].shape
    assert torch.equal(a['labels'],b['labels']) and torch.equal(a['repeat'],b['repeat']),'Unaligned targets'
    values=[]
    for za,zb,y,repeat in zip(a['logits'].reshape(-1,6).tolist(),b['logits'].reshape(-1,6).tolist(),
                            a['labels'].flatten().tolist(),a['repeat'].flatten().tolist()):
        old,new=scalar(za,y),scalar(zb,y)
        if old['correct'] and new['correct']:name='correct_both'
        elif not old['correct'] and not new['correct']:name='wrong_both'
        elif new['correct']:name='rescued'
        else:name='newly_wrong'
        values.append(dict(old=old,new=new,repeat=repeat,transition=name,change=new['ce']-old['ce']))
    result={}
    for group in ('all','unique','repeat'):
        ids=[i for i,v in enumerate(values) if group=='all' or v['repeat']==(group=='repeat')];n=len(ids);parts={}
        for name in ('correct_both','wrong_both','rescued','newly_wrong'):
            selected=[i for i in ids if values[i]['transition']==name];count=len(selected);means={}
            for endpoint in ('old','new'):
                for f in ('ce','true_probability','winner_confidence','true_margin'):
                    means[endpoint+'_'+f]=math.fsum(values[i][endpoint][f] for i in selected)/count if count else None
            parts[name]=dict(count=count,fraction=count/n if n else None,**means,
                mean_ce_change=math.fsum(values[i]['change'] for i in selected)/count if count else None,
                ce_contribution=math.fsum(values[i]['change'] for i in selected)/n if n else 0.,
                loss_increased=sum(values[i]['change']>1e-12 for i in selected),
                winner_confidence_increased=sum(values[i]['new']['winner_confidence']>values[i]['old']['winner_confidence']+1e-12 for i in selected),
                prediction_changed=sum(values[i]['new']['prediction']!=values[i]['old']['prediction'] for i in selected),
                old_confidence_ge_90=sum(values[i]['old']['winner_confidence']>=.9 for i in selected),
                new_confidence_ge_90=sum(values[i]['new']['winner_confidence']>=.9 for i in selected))
        result[group]=dict(count=n,old_accuracy=sum(values[i]['old']['correct'] for i in ids)/n if n else None,
            new_accuracy=sum(values[i]['new']['correct'] for i in ids)/n if n else None,
            old_ce=math.fsum(values[i]['old']['ce'] for i in ids)/n if n else None,
            new_ce=math.fsum(values[i]['new']['ce'] for i in ids)/n if n else None,
            ce_change=math.fsum(values[i]['change'] for i in ids)/n if n else None,transitions=parts)
        assert sum(p['count'] for p in parts.values())==n
        if n:
            assert abs(math.fsum(p['ce_contribution'] for p in parts.values())-(result[group]['new_ce']-result[group]['old_ce']))<1e-9
            assert result[group]['new_accuracy']-result[group]['old_accuracy']==(parts['rescued']['count']-parts['newly_wrong']['count'])/n or abs(result[group]['new_accuracy']-result[group]['old_accuracy']-(parts['rescued']['count']-parts['newly_wrong']['count'])/n)<1e-12
    return result,values


def compare(actual,expected):
    if isinstance(expected,dict):
        assert actual.keys()==expected.keys(),'Field mismatch'
        return max([compare(actual[k],expected[k]) for k in expected],default=0.)
    if expected is None:assert actual is None;return 0.
    if isinstance(expected,(int,str,bool)):assert actual==expected,'Count or identity mismatch';return 0.
    assert actual is not None and math.isfinite(actual)
    error=abs(actual-expected);assert error<=1e-9,'Independent scalar disagreement'
    return error


def audit_pair(a,b,rows,actual):
    _,q,y,repeat=previous.independent.inputs(rows,'all_queries')
    assert torch.equal(a['labels'],y) and torch.equal(a['repeat'],repeat),'Wrong labels/order'
    expected,values=reconstruct(a,b);error=compare(actual['groups'],expected)
    ranked=sorted((i for i,v in enumerate(values) if v['change']>1e-12),key=lambda i:(-round(values[i]['change'],10),i))[:5]
    assert [v['flat_index'] for v in actual['top_loss_increases']]==ranked,'Wrong top examples'
    for example,i in zip(actual['top_loss_increases'],ranked):
        h,j=divmod(i,y.shape[1]);v=values[i]
        expected=dict(flat_index=i,history_index=h,key=int(q[h,j]),label=int(y[h,j]),old_prediction=v['old']['prediction'],
            new_prediction=v['new']['prediction'],transition=v['transition'],old_ce=v['old']['ce'],new_ce=v['new']['ce'],ce_change=v['change'])
        error=max(error,compare(example,expected))
    return error


def main():
    torch.set_num_threads(1);report=json.loads((study.OUT/'report.json').read_text());lock=json.loads((study.OUT/'lock.json').read_text())
    assert report['status']=='complete' and report['training_updates']==0
    assert report['lock_sha256']==study.parent.parent.replay.digest(study.OUT/'lock.json')
    study.parent.parent.replay.check_files(lock['files'])
    pairs,old_report,new_report=study.inputs()
    expected_ids={(s,f'{split}_{m}') for s in (3461,3463) for split in ('train','validation','test') for m in (2,8,16,32)}
    assert len(pairs)==len(report['cells'])==24
    assert {(c['seed'],c['pool']) for c in report['cells']}==expected_ids
    errors=[]
    for (a,b,rows),record in zip(pairs,report['cells']):
        assert (record['seed'],record['pool'],record['old_step'],record['new_step'])==(a['seed'],a['pool'],3000,12000)
        assert record['old_checkpoint']==a['checkpoint'] and record['new_checkpoint']==b['checkpoint']
        for cell,source in ((a,old_report),(b,new_report)):
            source_record=next(c for c in source['cells'] if (c['seed'],c['arm'],c['step'],c['pool'])==(cell['seed'],cell['arm'],cell['step'],cell['pool']))
            assert source_record=={k:cell[k] for k in ('seed','arm','step','pool','checkpoint','metrics')}
            previous.check_metrics(cell['metrics'],previous.metrics(cell['logits'],cell['labels'],cell['repeat']))
        errors.append(audit_pair(a,b,rows,record))
    assert report['paired_answers']==sum(len(rows)*int(a['pool'].split('_')[-1]) for a,b,rows in pairs)==155904
    assert report['endpoint_vectors']==311808 and report['cells_count']==24
    study.parent.parent.replay.check_files(lock['files'])
    result=dict(passed=True,report_sha256=study.parent.parent.replay.digest(study.OUT/'report.json'),cells=24,
        paired_answers=155904,endpoint_vectors=311808,training_updates=0,max_scalar_error=max(errors),
        protected_files=len(lock['files']),forward_audits_reused_by_hash=True)
    path=study.OUT/'audit.json';assert not path.exists();study.parent.parent.replay.write(path,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
