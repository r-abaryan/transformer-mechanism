"""Independent fixed reads, scalar margins and exact class-balanced certificate."""
import csv
import json
from decimal import Decimal,localcontext
from fractions import Fraction
import numpy as np
import torch
import shared_readout_feasibility as study
import audit_shared_state_feasibility as ref


def exact_certificate(reads,labels,certificate):
    if not certificate:return False
    weights=[Fraction(r['weight']) for r in certificate];indices=[r['index'] for r in certificate]
    if min(weights)<0 or sum(weights)!=1 or len(set(indices))!=len(indices):return False
    if min(indices)<0 or max(indices)>=len(labels)*5:return False
    totals=[[Fraction(0) for _ in range(reads.shape[1]+1)] for c in range(6)];strict=False
    for item,lam in zip(certificate,weights):
        i,slot=divmod(item['index'],5);target=int(labels[i]);competitor=next(c for j,c in enumerate(c for c in range(6) if c!=target) if j==slot)
        x=[Fraction.from_float(float(v)) for v in reads[i]]+[Fraction(1)]
        for d,value in enumerate(x):totals[target][d]+=lam*value;totals[competitor][d]-=lam*value
        if lam>0 and target>competitor:strict=True
    return strict and all(v==0 for row in totals for v in row)


def independent_margin(reads,w,b,y):
    with localcontext() as context:
        context.prec=80;D=Decimal.from_float;weights=[[D(float(v)) for v in row] for row in w];bias=[D(float(v)) for v in b];lower=None
        for row,target in zip(reads,y):
            x=[D(float(v)) for v in row];target=int(target)
            for c in range(6):
                if c==target:continue
                terms=[(weights[target][j]-weights[c][j])*x[j] for j in range(len(x))]
                value=bias[target]-bias[c]+sum(terms,Decimal(0))
                magnitude=1+abs(bias[target])+abs(bias[c])+sum((abs(weights[target][j]*x[j])+abs(weights[c][j]*x[j]) for j in range(len(x))),Decimal(0))
                bound=value-Decimal('1e-60')*magnitude;lower=bound if lower is None else min(lower,bound)
        return float(lower)


@torch.no_grad()
def independent_features(model,rows):
    order=[];ys=[];repeat=[]
    for row in rows:
        mapping={}
        for k,v in row['events']:
            assert k not in mapping or mapping[k]==v;mapping[k]=v
        assert mapping[row['query']]==row['label'];keys=sorted(mapping);order.append(keys);ys.append([mapping[k] for k in keys]);repeat.append([sum(k==key for key,value in row['events'])>1 for k in keys])
    q=torch.tensor(order);y=torch.tensor(ys);rep=torch.tensor(repeat);e=torch.tensor([r['events'] for r in rows])
    raw=model.key.weight.detach();keys=raw/raw.square().sum(-1).sqrt().clamp_min(1e-12)[:,None]
    k=keys[e[...,0]];v=model.value.weight.detach()[e[...,1]];beta=float(model.write_logit.sigmoid());s=torch.zeros(len(rows),16,8,dtype=torch.float64)
    for t in range(e.shape[1]):s=(torch.eye(16,dtype=torch.float64)-beta*k[:,t,:,None]*k[:,t,None,:])@s+beta*k[:,t,:,None]*v[:,t,None,:]
    reads=keys[q]@s;z=reads@model.output.weight.detach().T+model.output.bias.detach()
    return q,y,rep,s,reads,z


def reduced(z,y,repeat):
    metric=ref.existing.metrics(z,y,repeat);result=dict(metric['groups']);result['complete_maps']=metric['complete_maps']['correct'];return result


def row_check(reads,labels):
    a,rhs,cost=study.program(reads.numpy(),labels.numpy());n,d=reads.shape;theta=np.linspace(-.73,.91,5*(d+1));gamma=.37
    vectors=np.vstack((np.zeros(d+1),theta.reshape(5,d+1)));scores=np.column_stack((reads.numpy(),np.ones(n)))@vectors.T
    expected=np.array([gamma-scores[i,int(target)]+scores[i,c] for i,target in enumerate(labels.tolist()) for c in range(6) if c!=target]+[gamma])
    actual=a@np.r_[theta,gamma];assert np.allclose(actual,expected,atol=1e-9,rtol=1e-9)
    assert a.shape==(n*5+1,5*(d+1)+1) and np.array_equal(rhs,np.r_[np.zeros(n*5),1.])
    assert np.array_equal(cost,np.r_[np.zeros(5*(d+1)),-1.])
    return float(np.max(np.abs(actual-expected)))


def audit(out,persist=True):
    lock=json.loads((out/'lock.json').read_text());report=json.loads((out/'report.json').read_text());study.prior.check_files(lock['files'])
    for field,name in (('lock_sha256','lock.json'),('witnesses_sha256','witnesses.pt'),('metrics_sha256','metrics.csv')):assert report[field]==study.prior.digest(out/name)
    bundles=torch.load(out/'witnesses.pt',weights_only=True);assert len(bundles)==len(lock['cases']);cells=[];summaries=[];native=0;witness=0;histories=0;maximum=0.;certificate_rows=0;bounds=[]
    historical=None if lock['fixture'] else json.loads((study.ROOT/'results/query_feedback_continuation/report.json').read_text())
    for case,bundle in zip(lock['cases'],bundles):
        assert bundle['seed']==case['seed'];payload=json.loads((study.ROOT/case['data']).read_text(encoding='utf-8-sig'))
        model=study.prior.historical.parent.previous.load_model(study.ROOT/case['checkpoint']).eval();reads=[];labels=[]
        assert len(bundle['blocks'])==len(case['pools'])
        for pool,block in zip(case['pools'],bundle['blocks']):
            assert block['pool']==pool;rows=payload['pools'][pool];q,y,repeat,s,x,z=independent_features(model,rows)
            for field,expected in (('queries',q),('labels',y),('repeat',repeat)):
                assert torch.equal(block[field],expected),field
            for field,expected in (('states',s),('reads',x),('native_logits',z)):
                actual=block[field];assert actual.shape==expected.shape;maximum=max(maximum,float((actual-expected).abs().max()))
                torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-9)
            metric=reduced(z,y,repeat);ref.compare(block['native_metrics'],metric)
            cells.append(dict(seed=case['seed'],pool=pool,arm='native',metrics=metric));reads.append(block['reads'].flatten(0,1));labels.append(y.flatten())
            native+=y.numel();histories+=len(y)
            if historical:
                before=next(c for c in historical['cells'] if c['seed']==case['seed'] and c['pool']==pool and c['step']==12000)
                # Existing full-pool metric has nested group/map fields.
                ref.compare(metric,{**before['metrics']['groups'],'complete_maps':before['metrics']['complete_maps']['correct']})
        x=torch.cat(reads);y=torch.cat(labels);maximum=max(maximum,row_check(x,y));record=bundle['record'];outcome=record['outcome']
        assert (record['constraints'],record['variables'])==(y.numel()*5+1,46)
        if outcome=='verified_feasible':
            assert bundle['head'] is not None and not bundle['certificate'];w=bundle['head']['weight'];b=bundle['head']['bias']
            assert w.shape==(6,8) and b.shape==(6,) and bool(torch.isfinite(w).all()) and bool(torch.isfinite(b).all())
            lower=independent_margin(x,w,b,y);assert lower>study.MARGIN and record['positive_lower_margin']>study.MARGIN;bounds.append(dict(seed=case['seed'],lower_margin=lower))
            ref.close(record['positive_lower_margin'],lower);ref.close(record['witness_weight_norm'],float(w.norm()));ref.close(record['witness_bias_norm'],float(b.norm()))
            for block in bundle['blocks']:
                z=block['reads']@w.T+b;torch.testing.assert_close(z,block['witness_logits'],atol=1e-9,rtol=1e-9)
                assert bool((z.argmax(-1)==block['labels']).all())
                cells.append(dict(seed=case['seed'],pool=block['pool'],arm='witness',metrics=reduced(z,block['labels'],block['repeat'])));witness+=block['labels'].numel()
        elif outcome=='certified_infeasible':
            assert bundle['head'] is None and exact_certificate(x.numpy(),y.numpy(),bundle['certificate'])
            assert record['certificate_support']==len(bundle['certificate']);certificate_rows+=len(bundle['certificate'])
        else:
            assert outcome=='unresolved' and bundle['head'] is None and not exact_certificate(x.numpy(),y.numpy(),bundle['certificate'])
        summaries.append(dict(seed=case['seed'],**record))
    ref.compare(report['cells'],cells);ref.compare(report['summary'],summaries)
    assert (report['native_queries'],report['witness_queries'],report['histories'],report['primal_solver_calls'])==(native,witness,histories,len(bundles))
    assert report['training_updates']==report['optimizer_calls']==0
    if not lock['fixture']:assert native==2*768*sum((2,8,16,32)) and histories==2*4*768 and len(bundles)==2
    with (out/'metrics.csv').open(newline='',encoding='utf-8') as f:actual=list(csv.DictReader(f))
    assert len(actual)==len(cells)
    for row,cell in zip(actual,cells):
        expected={k:v for k,v in cell.items() if k!='metrics'}
        for group,stats in cell['metrics'].items():
            if group=='complete_maps':expected[group]=stats
            else:
                for key,value in stats.items():expected[group+'_'+key]=value
        assert row.keys()==expected.keys()
        for key,value in expected.items():
            if value is None:assert row[key]==''
            elif isinstance(value,float):ref.close(float(row[key]),value)
            else:assert row[key]==str(value),key
    study.prior.check_files(lock['files']);result=dict(passed=True,report_sha256=study.prior.digest(out/'report.json'),
        cells=len(cells),native_queries=native,witness_queries=witness,certificate_rows=certificate_rows,
        max_equation_error=maximum,independent_margins=bounds,protected_files=len(lock['files']))
    if persist:
        assert not (out/'audit.json').exists();study.prior.write(out/'audit.json',result)
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':torch.set_num_threads(1);audit(study.OUT)
