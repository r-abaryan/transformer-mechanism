"""Independent scalar reads, labels, exact dual identities and map counts."""
import argparse
import csv
import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction

import torch
import shared_state_feasibility as study
import audit_query_feedback_training as existing


def close(a, b, scale=1e-9):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert abs(float(a)-float(b)) <= scale*(1+abs(float(b))), (a,b)


def compare(a, b):
    if isinstance(b, dict):
        assert isinstance(a, dict) and a.keys() == b.keys(), 'Dictionary coverage'
        for k in b: compare(a[k], b[k])
    elif isinstance(b, list):
        assert isinstance(a, list) and len(a) == len(b), 'List coverage'
        for x, y in zip(a,b): compare(x,y)
    elif isinstance(b, float): close(a,b)
    else: assert a == b, (a,b)


def oracle_read(q, s, w, b, y):
    with localcontext() as context:
        context.prec = 80
        D = lambda v: Decimal.from_float(float(v))
        output = []; lower = []
        for vector, target in zip(q, y):
            value = []; abs_value = []
            for j in range(len(w[0])):
                terms = [D(vector[d])*D(s[d][j]) for d in range(len(vector))]
                value.append(sum(terms, Decimal(0))); abs_value.append(sum(map(abs,terms), Decimal(0)))
            logits = []; guards = []
            for row, bias in zip(w,b):
                logits.append(D(bias)+sum((D(row[j])*value[j] for j in range(len(row))), Decimal(0)))
                guards.append(Decimal('1e-50')*(1+abs(D(bias))+sum(
                    (abs(D(row[j]))*abs_value[j] for j in range(len(row))), Decimal(0))))
            output.append([float(x) for x in logits])
            lower.extend(logits[target]-logits[c]-guards[target]-guards[c]
                         for c in range(len(b)) if c != target)
        return torch.tensor(output,dtype=torch.float64), float(min(lower))


def exact_upper(q,w,b,y,dual):
    if dual is None: return False, None
    F = lambda v: Fraction.from_float(float(v))
    lam = [F(v) for v in dual]
    if len(lam) != len(y)*(len(b)-1)+1 or min(lam) < 0 or sum(lam) != 1: return False, None
    # First aggregate coefficients by query/class, then multiply by the head.
    class_weights = [[Fraction(0) for c in b] for i in y]; n=0
    upper = lam[-1]
    for i,target in enumerate(y):
        for c in range(len(b)):
            if c == target: continue
            class_weights[i][target] += lam[n]; class_weights[i][c] -= lam[n]
            n += 1
    for i in range(len(y)):
        upper += sum((coefficient*F(b[c]) for c,coefficient in enumerate(class_weights[i])),Fraction(0))
    for d in range(len(q[0])):
        for j in range(len(w[0])):
            total = sum((F(q[i][d])*sum((class_weights[i][c]*F(w[c][j])
                for c in range(len(b))),Fraction(0)) for i in range(len(y))),Fraction(0))
            if total != 0: return False, float(upper)
    return upper <= 0, float(upper)


def aggregate(records):
    output = []
    for seed in sorted(set(r['seed'] for r in records)):
        for name in ('all','native_complete','native_incomplete'):
            chosen = [r for r in records if r['seed']==seed and
                      (name=='all' or bool(r['native_complete'])==(name=='native_complete'))]
            counts = dict(seed=seed,group=name,histories=0,positive=0,certified_nonpositive=0,
                          unresolved=0,native_complete=0,native_correct=0,queries=0,
                          oracle_scored_queries=0,oracle_correct=0)
            for r in chosen:
                counts['histories']+=1; counts[r['outcome']]+=1
                for k in ('native_complete','native_correct','queries'): counts[k]+=r[k]
                if r['oracle_correct'] is not None:
                    counts['oracle_scored_queries']+=r['queries']; counts['oracle_correct']+=r['oracle_correct']
            output.append(counts)
    return output


def audit(out, persist=True):
    lock=json.loads((out/'lock.json').read_text()); report=json.loads((out/'report.json').read_text())
    study.check_files(lock['files'])
    assert report['lock_sha256']==study.digest(out/'lock.json')
    assert report['witnesses_sha256']==study.digest(out/'witnesses.pt')
    assert report['metrics_sha256']==study.digest(out/'metrics.csv')
    assert report['status']=='complete' and report['optimizer_calls']==report['training_updates']==0
    compare(report['versions'],lock['versions']); assert report['versions']==study.versions()
    assert lock['method']=='highs-ds' and lock['options']==study.OPTIONS and lock['strict_margin']==study.MARGIN
    bundles=torch.load(out/'witnesses.pt',map_location='cpu',weights_only=True)
    assert len(bundles)==len(lock['cases']), 'Missing seed'
    records=[]; errors=[]; native_metrics=[]; checked=0; oracle_outputs=0
    for case,bundle in zip(lock['cases'],bundles):
        rows=json.loads((study.ROOT/case['data']).read_text())['pools'][case['pool']]
        assert bundle['seed']==case['seed'] and len(bundle['records'])==len(rows)
        ids=[]; labels=[]; repeat=[]
        for row in rows:
            mapping={}; counts={}
            for key,value in row['events']:
                assert 0<=key<32 and 0<=value<6
                assert key not in mapping or mapping[key]==value, 'Unstable label'
                mapping[key]=value; counts[key]=counts.get(key,0)+1
            assert mapping[row['query']]==row['label'], 'Assigned label'
            keys=sorted(counts); ids.append(keys); labels.append([mapping[q] for q in keys]); repeat.append([counts[q]>1 for q in keys])
        ids=torch.tensor(ids); y=torch.tensor(labels); rs=torch.tensor(repeat)
        assert torch.equal(bundle['queries'],ids) and torch.equal(bundle['labels'],y), 'Labels/queries'
        assert torch.equal(bundle['repeat'],rs), 'Groups'
        model=study.historical.parent.previous.load_model(study.ROOT/case['checkpoint'])
        w=model.output.weight.detach().tolist(); b=model.output.bias.detach().tolist()
        e=torch.tensor([r['events'] for r in rows])
        with torch.no_grad():
            z,_=existing.independent.forward_tangent(model,e,ids)
            raw=model.key.weight[ids]; manual_q=raw/raw.square().sum(-1,keepdim=True).clamp_min(1e-24).sqrt()
        errors.append(float((z-bundle['native_logits']).abs().max()))
        assert torch.allclose(z,bundle['native_logits'],atol=1e-9,rtol=1e-9), 'Native logits'
        assert torch.equal(z.argmax(-1),bundle['native_logits'].argmax(-1)), 'Native winners'
        assert torch.allclose(manual_q,bundle['normalized_queries'],atol=1e-14,rtol=1e-14), 'Normalized queries'
        expected_metrics=existing.metrics(z,y,rs)
        expected_metrics={**expected_metrics['groups'],'complete_maps':expected_metrics['complete_maps']['correct']}
        compare(bundle['native_metrics'],expected_metrics)
        if not lock['fixture']:
            original=json.loads((study.historical.OUT/'report.json').read_text())
            cell=next(c for c in original['cells'] if c['seed']==case['seed'] and c['step']==12000 and c['pool']=='train_32')
            assert cell['checkpoint']==case['checkpoint']
            compare(expected_metrics,{**cell['metrics']['groups'],'complete_maps':cell['metrics']['complete_maps']['correct']})
        native_metrics.append(dict(seed=case['seed'],metrics=expected_metrics))
        for i,record in enumerate(bundle['records']):
            assert record['history']==i and record['seed']==case['seed'] and record['queries']==len(ids[i])
            correct=int((z[i].argmax(-1)==y[i]).sum())
            assert record['native_correct']==correct and record['native_complete']==(correct==len(ids[i]))
            native_gaps=[min(float(z[i,j,target]-z[i,j,c]) for c in range(6) if c!=target) for j,target in enumerate(y[i].tolist())]
            close(record['native_min_margin'],min(native_gaps))
            assert record['native_target_ties']==sum(gap==0 for gap in native_gaps)
            native_ce=existing.metrics(z[i:i+1],y[i:i+1],rs[i:i+1])['groups']['all']['ce']
            close(record['native_ce'],native_ce)
            q=bundle['normalized_queries'][i].tolist(); s=bundle['states'][i]; yi=y[i].tolist()
            expected='unresolved'
            if torch.isfinite(s).all():
                scalar,lower=oracle_read(q,s.tolist(),w,b,yi)
                native=model.read((s[None].expand(len(yi),-1,-1),),ids[i]).detach()
                err=float((scalar-native).abs().max()); errors.append(err)
                assert err<=1e-9*(1+float(scalar.abs().max())), 'Oracle scalar/native'
                assert torch.allclose(native,bundle['oracle_logits'][i],atol=1e-9,rtol=1e-9), 'Saved oracle logits'
                close(record['decimal_guarded_margin'],lower)
                minimum=min(float(native[j,target]-native[j,c]) for j,target in enumerate(yi) for c in range(6) if c!=target)
                close(record['oracle_min_margin'],minimum)
                oracle_correct=int((native.argmax(-1)==y[i]).sum())
                assert record['oracle_correct']==oracle_correct and record['oracle_complete']==(oracle_correct==len(yi))
                oracle_ce=existing.metrics(scalar[None],y[i:i+1],rs[i:i+1])['groups']['all']['ce']
                close(record['oracle_ce'],oracle_ce)
                gaps=[min(float(native[j,target]-native[j,c]) for c in range(6) if c!=target) for j,target in enumerate(yi)]
                assert record['oracle_target_ties']==sum(gap==0 for gap in gaps)
                close(record['oracle_state_norm'],float(s.norm()))
                close(record['native_state_norm'],float(bundle['native_states'][i].norm()))
                close(record['oracle_to_native_norm'],float(s.norm()/bundle['native_states'][i].norm()))
                if lower>study.MARGIN and minimum>study.MARGIN: expected='positive'
                oracle_outputs+=len(yi)
            else:
                assert torch.isnan(bundle['oracle_logits'][i]).all(), 'Missing state logits'
            if expected!='positive':
                certificate,bound=exact_upper(q,w,b,yi,bundle['duals'][i])
                if certificate: expected='certified_nonpositive'
                if record['exact_dual_bound'] is not None and bound is not None: close(record['exact_dual_bound'],bound)
            assert record['outcome']==expected, 'Certificate outcome'
            if expected=='positive': assert torch.equal(bundle['oracle_logits'][i].argmax(-1),y[i]), 'Positive winners'
            records.append(record); checked+=len(yi)
    compare(report['summary'],aggregate(records)); compare(report['native_metrics'],native_metrics)
    with (out/'metrics.csv').open(newline='',encoding='utf-8') as file: csv_rows=list(csv.DictReader(file))
    assert len(csv_rows)==len(records), 'CSV coverage'
    for saved, record in zip(csv_rows,records):
        assert saved.keys()==record.keys()
        for key,value in record.items():
            if value is None: assert saved[key]=='', key
            elif isinstance(value,float): close(float(saved[key]),value)
            else: assert saved[key]==str(value), ('CSV',key)
    assert report['histories']==len(records) and report['native_answers']==checked
    assert report['oracle_answer_outputs']==oracle_outputs
    if not lock['fixture']:
        assert [c['seed'] for c in lock['cases']]==[3461,3463]
        assert len(records)==2*768 and checked==2*768*32 and all(r['queries']==32 for r in records)
    study.check_files(lock['files'])
    result=dict(passed=True,report_sha256=study.digest(out/'report.json'),histories=len(records),
                native_answers=checked,oracle_answer_outputs=oracle_outputs,max_logit_error=max(errors),
                protected_files=len(lock['files']),training_updates=0)
    if persist:
        assert not (out/'audit.json').exists(); study.write(out/'audit.json',result)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=str)
    args=parser.parse_args(); audit(study.OUT if args.out is None else study.ROOT/args.out)
