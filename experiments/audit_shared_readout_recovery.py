"""Original-input certificates and independent recovery-coordinate arithmetic."""
import copy
import csv
import json
from decimal import Decimal,localcontext
from fractions import Fraction
import numpy as np
import torch
import shared_readout_recovery as study
import audit_shared_readout_feasibility as proof


def nonsingular(matrix):
    a=[[Fraction.from_float(float(v)) for v in row] for row in matrix]
    if len(a)!=8 or any(len(row)!=8 for row in a):return False
    for j in range(8):
        pivot=next((i for i in range(j,8) if a[i][j]),None)
        if pivot is None:return False
        a[j],a[pivot]=a[pivot],a[j];p=a[j][j]
        for i in range(j+1,8):
            factor=a[i][j]/p
            for k in range(j,8):a[i][k]-=factor*a[j][k]
    return True


def independent_transform(x,coordinate):
    with localcontext() as context:
        context.prec=100;D=Decimal.from_float
        t=[[D(float(v)) for v in row] for row in coordinate['matrix']];mean=[D(float(v)) for v in coordinate['mean']]
        offsets=[sum((mean[i]*t[i][j] for i in range(8)),Decimal(0)) for j in range(8)]
        return np.array([[float(sum((D(float(row[i]))*t[i][j] for i in range(8)),Decimal(0))-offsets[j]) for j in range(8)] for row in x])


@torch.no_grad()
def audit(out,persist=True):
    prior=study.original.prior;lock=json.loads((out/'lock.json').read_text());report=json.loads((out/'report.json').read_text())
    prior.check_files(lock['files']);origin=study.ROOT/lock['origin'];oldreport=json.loads((origin/'report.json').read_text())
    oldaudit=json.loads((origin/'audit.json').read_text());assert oldaudit['passed'] and oldaudit['report_sha256']==prior.digest(origin/'report.json')
    for field,name in (('lock_sha256','lock.json'),('witnesses_sha256','witnesses.pt'),('metrics_sha256','metrics.csv')):assert report[field]==prior.digest(out/name)
    original=torch.load(origin/'witnesses.pt',weights_only=True);recovered=torch.load(out/'witnesses.pt',weights_only=True)
    assert len(recovered)==len(original)==len(lock['cases']);cells=[];summary=[];maximum=0.;certificates=0;checked=0;candidate_queries=0
    for source,bundle,case in zip(original,recovered,lock['cases']):
        assert source['seed']==bundle['seed']==case['seed'];x=torch.cat([b['reads'].flatten(0,1) for b in source['blocks']]);y=torch.cat([b['labels'].flatten() for b in source['blocks']])
        checked+=y.numel();coordinate=bundle['coordinate'];record=bundle['record'];outcome=record['outcome']
        assert (record['constraints'],record['variables'])==(y.numel()*5+1,46)
        model=prior.historical.parent.previous.load_model(study.ROOT/case['checkpoint']).eval();values=model.value.weight.detach()
        q,_=torch.linalg.qr(values.T,mode='complete');residual=float((x@q[:,6:]).abs().max())
        assert residual<=1e-12 and bundle['geometry']['max_outside_value_span']<=1e-12
        np.testing.assert_allclose(bundle['geometry']['coordinate_sd'],x.numpy().std(0),atol=1e-9,rtol=1e-9)
        proof.ref.compare(bundle['geometry']['read_range'],[float(x.min()),float(x.max())])
        if coordinate is None:
            assert outcome=='unresolved' and record['solver_calls']==0 and bundle['candidate'] is None and not bundle['certificate']
        else:
            assert coordinate['matrix'].shape==(8,8) and nonsingular(coordinate['matrix']) and record['solver_calls']==1
            np.testing.assert_allclose(coordinate['mean'],x.numpy().mean(0),atol=1e-9,rtol=1e-9)
            singular=np.linalg.svd(x.numpy()-x.numpy().mean(0),compute_uv=False)
            np.testing.assert_allclose(coordinate['singular_values'],singular,atol=1e-9,rtol=1e-9)
            assert coordinate['exponents']==np.floor(np.log2(np.array(coordinate['singular_values'])/np.sqrt(len(x)))).astype(int).tolist()
            expected=independent_transform(x,coordinate);np.testing.assert_allclose(coordinate['transformed'],expected,atol=1e-9,rtol=1e-9)
            maximum=max(maximum,float(np.max(np.abs(coordinate['transformed'].numpy()-expected))))
            maximum=max(maximum,proof.row_check(coordinate['transformed'],y))
            candidate=bundle['candidate']
            if candidate is not None and 'logits' in candidate:
                w=candidate['weight'];bias=candidate['bias'];lower=proof.independent_margin(x,w,bias,y)
                proof.ref.close(record['positive_lower_margin'],lower);private=copy.deepcopy(model)
                private.output.weight.copy_(w);private.output.bias.copy_(bias);allcorrect=True
                assert len(candidate['logits'])==len(source['blocks'])
                for block,z in zip(source['blocks'],candidate['logits']):
                    n,m=block['queries'].shape
                    # Per-history reads avoid the producer's flattened state batching.
                    actual=torch.stack([private.read((state[None].expand(m,-1,-1),),query) for state,query in zip(block['states'],block['queries'])])
                    torch.testing.assert_close(z,actual,atol=1e-9,rtol=1e-9);allcorrect &= bool((actual.argmax(-1)==block['labels']).all())
                    cells.append(dict(seed=case['seed'],pool=block['pool'],arm='candidate',metrics=proof.reduced(actual,block['labels'],block['repeat'])))
                assert record['native_candidate_all_correct']==allcorrect and record['candidate_queries']==y.numel();candidate_queries+=y.numel()
                if outcome=='verified_feasible':assert lower>study.original.MARGIN and allcorrect and record['witness_queries']==y.numel()
            else:assert outcome!='verified_feasible' and record['candidate_queries']==0
            if outcome=='certified_infeasible':
                assert proof.exact_certificate(x.numpy(),y.numpy(),bundle['certificate']);certificates+=1
            elif outcome=='unresolved':assert not proof.exact_certificate(x.numpy(),y.numpy(),bundle['certificate'])
            else:assert outcome=='verified_feasible' and not bundle['certificate']
            assert record['certificate_support']==len(bundle['certificate'])
            groups={b['pool']:0 for b in source['blocks']};labels={str(c):0 for c in range(6)}
            for item in bundle['certificate']:
                sample=item['index']//5;left=sample
                for block in source['blocks']:
                    if left<block['labels'].numel():groups[block['pool']]+=1;labels[str(int(block['labels'].flatten()[left]))]+=1;break
                    left-=block['labels'].numel()
                else:raise AssertionError('support index')
            assert record['support_groups']==dict(pools=groups,labels=labels)
        proof.ref.compare(bundle['cells'],[c for c in cells if c['seed']==case['seed']])
        summary.append(dict(seed=case['seed'],geometry=bundle['geometry'],**record))
    proof.ref.compare(report['summary'],summary);proof.ref.compare(report['cells'],cells);proof.ref.compare(report['native_reference'],oldreport['cells'])
    assert report['original_queries']==oldreport['native_queries']==checked and report['histories']==oldreport['histories']
    assert report['solver_calls']==sum(b['record']['solver_calls'] for b in recovered)<=len(recovered)
    assert report['training_updates']==report['optimizer_calls']==report['native_prefills']==0
    with (out/'metrics.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    assert len(rows)==len(cells)
    for row,cell in zip(rows,cells):
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
    prior.check_files(lock['files']);result=dict(passed=True,report_sha256=prior.digest(out/'report.json'),original_queries=checked,
        candidate_queries=candidate_queries,certified_seeds=certificates,max_equation_error=maximum,protected_files=len(lock['files']))
    if persist:
        assert not (out/'audit.json').exists();prior.write(out/'audit.json',result)
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':torch.set_num_threads(1);audit(study.OUT)
