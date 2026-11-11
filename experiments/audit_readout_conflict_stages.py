"""Independent checkpoint labels/recurrence and exact stage certificates."""
import copy
import csv
import json
import numpy as np
import torch
import readout_conflict_stages as study
import audit_shared_readout_recovery as coordinate_check
import audit_shared_readout_feasibility as proof
import query_feedback_training as history


@torch.no_grad()
def check_recovery(source,rec,model):
    x=torch.cat([b['reads'].flatten(0,1) for b in source['blocks']]);y=torch.cat([b['labels'].flatten() for b in source['blocks']])
    record=rec['record'];coordinate=rec['coordinate'];outcome=record['outcome'];cells=[];maximum=0.
    assert rec['seed']==source['seed'] and (record['constraints'],record['variables'])==(len(y)*5+1,46)
    if coordinate is None:
        assert outcome=='unresolved' and record['solver_calls']==0 and rec['candidate'] is None and not rec['certificate']
        return cells,maximum
    assert record['solver_calls']==1 and coordinate_check.nonsingular(coordinate['matrix'])
    np.testing.assert_allclose(coordinate['transformed'],coordinate_check.independent_transform(x,coordinate),atol=1e-9,rtol=1e-9)
    maximum=proof.row_check(coordinate['transformed'],y)
    candidate=rec['candidate'];nativecorrect=None
    if candidate is not None and 'logits' in candidate:
        lower=proof.independent_margin(x,candidate['weight'],candidate['bias'],y);proof.ref.close(record['positive_lower_margin'],lower)
        private=copy.deepcopy(model);private.output.weight.copy_(candidate['weight']);private.output.bias.copy_(candidate['bias']);nativecorrect=True
        assert len(candidate['logits'])==len(source['blocks'])
        for block,z in zip(source['blocks'],candidate['logits']):
            n,m=block['queries'].shape;states=block['states'][:,None].expand(-1,m,-1,-1).reshape(n*m,16,8)
            actual=private.read((states,),block['queries'].flatten()).reshape(n,m,6)
            torch.testing.assert_close(z,actual,atol=1e-9,rtol=1e-9);nativecorrect &= bool((actual.argmax(-1)==block['labels']).all())
            cells.append(dict(seed=source['seed'],pool=block['pool'],arm='candidate',metrics=proof.reduced(actual,block['labels'],block['repeat'])))
        assert record['candidate_queries']==len(y) and record['native_candidate_all_correct']==nativecorrect
    else:assert record['candidate_queries']==0
    if outcome=='verified_feasible':assert lower>study.original.MARGIN and nativecorrect and not rec['certificate'] and record['witness_queries']==len(y)
    elif outcome=='certified_infeasible':assert proof.exact_certificate(x.numpy(),y.numpy(),rec['certificate']) and record['witness_queries']==0
    else:assert outcome=='unresolved' and not proof.exact_certificate(x.numpy(),y.numpy(),rec['certificate']) and record['witness_queries']==0
    assert record['certificate_support']==len(rec['certificate']);proof.ref.compare(rec['cells'],cells)
    pools={b['pool']:0 for b in source['blocks']};labels={str(c):0 for c in range(6)}
    for row in rec['certificate']:
        left=row['index']//5
        for block in source['blocks']:
            if left<block['labels'].numel():
                pools[block['pool']]+=1;labels[str(int(block['labels'].flatten()[left]))]+=1;break
            left-=block['labels'].numel()
        else:raise AssertionError('support index')
    assert record['support_groups']==dict(pools=pools,labels=labels)
    return cells,maximum


@torch.no_grad()
def audit(out,persist=True):
    prior=study.original.prior;lock=json.loads((out/'lock.json').read_text());report=json.loads((out/'report.json').read_text());prior.check_files(lock['files'])
    for field,name in (('lock_sha256','lock.json'),('stages_sha256','stages.pt'),('metrics_sha256','metrics.csv')):assert report[field]==prior.digest(out/name)
    bundles=torch.load(out/'stages.pt',weights_only=True);assert len(bundles)==len(lock['cases']);cells=[];summaries=[];maximum=0.;fresh=0;total=0;histories=0;calls=0;prefills=0
    historical=None;continuation=None;cached_source=None;cached_rec=None
    if not lock['fixture']:
        assert lock['cases']==study.cases()
        historical=json.loads((study.ROOT/'results/query_feedback_training/report.json').read_text());continuation=json.loads((study.ROOT/'results/query_feedback_continuation/report.json').read_text())
        cached_source=torch.load(study.original.OUT/'witnesses.pt',weights_only=True);cached_rec=torch.load(study.recovery.OUT/'witnesses.pt',weights_only=True)
    for case,bundle in zip(lock['cases'],bundles):
        assert bundle['case']==case and bundle['source']['seed']==case['seed'];source=bundle['source'];rec=bundle['recovery']
        cp=torch.load(study.ROOT/case['checkpoint'],weights_only=True);assert cp['step']==case['step']
        model=prior.historical.parent.previous.load_model(study.ROOT/case['checkpoint']).eval();payload=json.loads((study.ROOT/case['data']).read_text(encoding='utf-8-sig'))
        if historical:
            run=next(r for r in historical['runs'] if r['seed']==case['seed'] and r['arm']=='exhaustive')
            if case['step']==0:assert history.parent.study.parent.state_hash(cp['model'])==run['initial_sha256']
            continuation_run=next(r for r in continuation['runs'] if r['seed']==case['seed'])
            assert continuation_run['start']==3000 and continuation_run['parent_sha256']==prior.digest(study.ROOT/f"results/query_feedback_training/{case['seed']}_exhaustive_3000.pt")
        assert len(source['blocks'])==len(case['pools'])
        if case['cached']:
            assert not lock['fixture']
            assert history.replay.equal(source,next(b for b in cached_source if b['seed']==case['seed']))
            assert history.replay.equal(rec,next(b for b in cached_rec if b['seed']==case['seed']))
        for pool,block in zip(case['pools'],source['blocks']):
            assert block['pool']==pool;rows=payload['pools'][pool]
            if not case['cached']:
                q,y,rep,s,x,z=proof.independent_features(model,rows)
                for field,expected in (('queries',q),('labels',y),('repeat',rep)):assert torch.equal(block[field],expected)
                for field,expected in (('states',s),('reads',x),('native_logits',z)):
                    torch.testing.assert_close(block[field],expected,atol=1e-9,rtol=1e-9);maximum=max(maximum,float((block[field]-expected).abs().max()))
                metric=proof.reduced(z,y,rep);proof.ref.compare(block['native_metrics'],metric)
                fresh+=y.numel();prefills+=2
            else:metric=block['native_metrics'];y=block['labels']
            total+=y.numel();histories+=len(rows);cells.append(dict(seed=case['seed'],step=case['step'],pool=pool,arm='native',metrics=metric))
            if historical:
                before=next(c for c in (continuation if case['cached'] else historical)['cells'] if c['seed']==case['seed'] and c['step']==case['step'] and c['pool']==pool and c['arm']=='exhaustive')
                proof.ref.compare(metric,{**before['metrics']['groups'],'complete_maps':before['metrics']['complete_maps']['correct']})
        candidate,error=check_recovery(source,rec,model);maximum=max(maximum,error)
        cells.extend(dict(step=case['step'],**c) for c in candidate)
        summaries.append(dict(seed=case['seed'],step=case['step'],cached=case['cached'],**rec['record']))
        if not case['cached']:calls+=rec['record']['solver_calls']
    proof.ref.compare(report['cells'],cells);proof.ref.compare(report['summary'],summaries)
    assert (report['fresh_native_queries'],report['reported_native_queries'],report['histories'],report['new_solver_calls'],report['fresh_native_prefills'])==(fresh,total,histories,calls,prefills)
    assert report['training_updates']==report['optimizer_calls']==0
    if not lock['fixture']:assert (fresh,total,histories,calls,prefills)==(178176,267264,18432,4,32)
    with (out/'metrics.csv').open(newline='',encoding='utf-8') as f:actual=list(csv.DictReader(f))
    assert len(actual)==len(cells)
    for row,cell in zip(actual,cells):
        expected={k:v for k,v in cell.items() if k!='metrics'}
        for group,stats in cell['metrics'].items():
            if group=='complete_maps':expected[group]=stats
            else:
                for k,v in stats.items():expected[group+'_'+k]=v
        assert row.keys()==expected.keys()
        for k,v in expected.items():
            if v is None:assert row[k]==''
            elif isinstance(v,float):proof.ref.close(float(row[k]),v)
            else:assert row[k]==str(v)
    prior.check_files(lock['files']);result=dict(passed=True,report_sha256=prior.digest(out/'report.json'),stages=len(bundles),cells=len(cells),
        fresh_native_queries=fresh,reported_native_queries=total,new_solver_calls=calls,max_equation_error=maximum,protected_files=len(lock['files']))
    if persist:
        assert not (out/'audit.json').exists();prior.write(out/'audit.json',result)
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':torch.set_num_threads(1);audit(study.OUT)
