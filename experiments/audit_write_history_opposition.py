"""Independent chronological prefix cross terms, without subtraction for B."""
import csv
import json
import torch
import write_history_opposition as study
import audit_shared_state_feasibility as ref


def independent(g,ids,supervised,seed,group):
    h,t,v=g.shape;local=torch.zeros(h,6,v,dtype=torch.float64);global_prefix=torch.zeros(6,v,dtype=torch.float64)
    within=torch.zeros(h,dtype=torch.float64);base_h=torch.zeros(h,dtype=torch.float64)
    b=0.;p=0.;position_energy=0.;positions=[];indices=torch.arange(h)
    for step in range(t):
        x=g[:,step];value=ids[:,step]
        same=local[indices,value];others=global_prefix[value]-same
        within+=2*(x*same).sum(-1);b+=2*float((x*others).sum())
        base_h+=x.square().sum(-1);pooled=torch.zeros(6,v,dtype=torch.float64)
        for label in range(6):
            selected=x[value==label];pooled[label]=selected.sum(0)
            # Other histories at this position, each unordered pair once.
            if len(selected):p+=2*float((selected*(selected.cumsum(0)-selected)).sum())
        position_energy+=float(pooled.square().sum());positions.append(pooled)
        local[indices,value]+=x;global_prefix+=pooled
    w=float(within.sum());base=float(base_h.sum());joint=float(global_prefix.square().sum());c=w+b
    rows=[]
    for i in range(h):
        val=float(within[i]);sgn='negative' if val < -1e-14 else 'positive' if val > 1e-14 else 'unresolved'
        rows.append(dict(seed=seed,group=group,history=i,supervised=bool(supervised[i]),event_energy=float(base_h[i]),
            map_energy=float(local[i].square().sum()),within=val,sign=sgn))
    summary=dict(seed=seed,group=group,histories=h,positions=t,supervised_histories=int(supervised.sum()),
        event_energy=base,position_energy=position_energy,joint_energy=joint,within=w,between_temporal=b,
        between_same_position=p,temporal_cross=c,within_over_event=w/base if base else None,
        between_temporal_over_event=b/base if base else None,between_same_position_over_event=p/base if base else None,
        supervised_negative=len([r for r in rows if r['supervised'] and r['sign']=='negative']),
        supervised_positive=len([r for r in rows if r['supervised'] and r['sign']=='positive']),
        supervised_unresolved=len([r for r in rows if r['supervised'] and r['sign']=='unresolved']),
        unsupervised_histories=len([r for r in rows if not r['supervised']]))
    ref.close(joint,base+w+b+p,scale=1e-12)
    return summary,rows,torch.stack(positions)


def audit(out,persist=True):
    lock=json.loads((out/'lock.json').read_text());report=json.loads((out/'report.json').read_text())
    assert report['lock_sha256']==study.primary.prior.digest(out/'lock.json')
    assert report['histories_sha256']==study.primary.prior.digest(out/'histories.csv')
    study.primary.prior.check_files(lock['files']);source=study.ROOT/lock['source']
    upstream=json.loads((source/'lock.json').read_text());up_report=json.loads((source/'report.json').read_text());up_audit=json.loads((source/'audit.json').read_text())
    assert up_audit['passed'] and up_audit['report_sha256']==study.primary.prior.digest(source/'report.json')
    assert up_report['gradients_sha256']==study.primary.prior.digest(source/'gradients.pt')
    assert lock['cases']==upstream['cases'];bundles=torch.load(source/'gradients.pt',weights_only=True)
    assert len(bundles)==len(lock['cases']);summaries=[];history=[];maximum=0.
    for case,bundle in zip(lock['cases'],bundles):
        assert bundle['seed']==case['seed'] and tuple(bundle['gradients'])==study.primary.GROUPS
        rows=json.loads((study.ROOT/case['data']).read_text(encoding='utf-8-sig'))['pools'][case['pool']]
        ids=torch.tensor([[event[1] for event in row['events']] for row in rows]);assert ids.shape[0]==len(bundle['labels'])
        assert bundle['masks'].keys()==bundle['gradients'].keys()
        for group in ('correct','error','total'):
            g=bundle['gradients'][group];assert g['events'].shape[:2]==ids.shape
            assert bool(torch.isfinite(g['events']).all())
            mask=bundle['masks'][group];assert mask.shape==bundle['labels'].shape
            expected,records,positions=independent(g['events'],ids,mask.any(1),case['seed'],group)
            torch.testing.assert_close(positions,g['positions'],atol=1e-12,rtol=1e-12)
            torch.testing.assert_close(positions.sum(0),g['embedding'],atol=1e-12,rtol=1e-12)
            summaries.append(expected);history.extend(records)
    assert len(report['summary'])==len(summaries)
    for actual,expected in zip(report['summary'],summaries):
        assert actual.keys()==expected.keys()
        for key,value in expected.items():
            if isinstance(value,float):
                maximum=max(maximum,abs(actual[key]-value));ref.close(actual[key],value,scale=1e-12)
            else:assert actual[key]==value,key
    if not lock['fixture']:
        assert len(summaries)==2*3 and len(history)==2*3*768
        old=json.loads((source/'cross_terms.json').read_text())
        for row,before in zip(summaries,old['rows']):
            assert (row['seed'],row['group'])==(before['seed'],before['group'])
            ref.close(row['temporal_cross'],before['cross_energy'],scale=1e-12)
            ref.close(row['joint_energy'],before['joint_energy'],scale=1e-12)
    with (out/'histories.csv').open(newline='',encoding='utf-8') as f:actual=list(csv.DictReader(f))
    assert len(actual)==len(history) and report['history_rows']==len(history) and report['cohorts']==len(summaries)
    for row,expected in zip(actual,history):
        assert row.keys()==expected.keys()
        for key,value in expected.items():
            if isinstance(value,float):ref.close(float(row[key]),value,scale=1e-12)
            else:assert row[key]==str(value),key
    assert report['model_evaluations']==report['backward_calls']==report['optimizer_calls']==report['training_updates']==0
    study.primary.prior.check_files(lock['files'])
    result=dict(passed=True,report_sha256=study.primary.prior.digest(out/'report.json'),cohorts=len(summaries),
        history_rows=len(history),max_summary_error=maximum,protected_files=len(lock['files']))
    if persist:
        assert not (out/'audit.json').exists();study.primary.prior.write(out/'audit.json',result)
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':torch.set_num_threads(1);audit(study.OUT)
