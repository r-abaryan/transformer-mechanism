"""One unrestricted shared affine head on fixed learned training reads."""
import argparse
import copy
import csv
import importlib.util
import json
import time
import unittest
from decimal import Decimal,localcontext
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import torch
import shared_state_feasibility as prior

ROOT=prior.ROOT
OUT=ROOT/'results/shared_readout_feasibility'
PROTOCOL='docs/protocols/shared-readout-feasibility.md'
SOURCES=('experiments/shared_readout_feasibility.py','experiments/audit_shared_readout_feasibility.py',
         'tests/test_shared_readout_feasibility.py')
LOADS=(2,8,16,32)
MARGIN=1e-7
OPTIONS=dict(presolve=True,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,
             maxiter=100000,time_limit=60.)


def program(reads,labels):
    x=np.asarray(reads,dtype=np.float64);y=np.asarray(labels,dtype=np.int64);n,d=x.shape
    assert y.shape==(n,) and d>0 and n>0 and np.isfinite(x).all() and np.all((y>=0)&(y<6))
    phi=np.column_stack((x,np.ones(n)));width=d+1;variables=5*width
    competitor=np.array([[c for c in range(6) if c!=target] for target in y],dtype=np.int64).flatten()
    target=np.repeat(y,5);sample=np.repeat(np.arange(n),5);constraint=np.arange(n*5)
    row=[];col=[];value=[]
    for classes,sign in ((target,-1.),(competitor,1.)):
        selected=classes>0
        row.append(np.repeat(constraint[selected],width));col.append(((classes[selected,None]-1)*width+np.arange(width)).flatten())
        value.append((sign*phi[sample[selected]]).flatten())
    row.append(np.arange(n*5+1));col.append(np.full(n*5+1,variables));value.append(np.ones(n*5+1))
    a=coo_matrix((np.concatenate(value),(np.concatenate(row),np.concatenate(col))),shape=(n*5+1,variables+1)).tocsr()
    rhs=np.r_[np.zeros(n*5),1.];cost=np.r_[np.zeros(variables),-1.]
    return a,rhs,cost


def rational_row(reads,labels,index):
    sample,slot=divmod(index,5);target=int(labels[sample]);competitor=[c for c in range(6) if c!=target][slot]
    phi=[Fraction.from_float(float(v)) for v in reads[sample]]+[Fraction(1)];width=len(phi)
    result=[Fraction(0)]*(5*width)
    for label,sign in ((target,1),(competitor,-1)):
        if label:
            for j,x in enumerate(phi):result[(label-1)*width+j]+=sign*x
    return result


def reconstruct(reads,labels,dual):
    support=np.where(np.asarray(dual[:-1])>0)[0].tolist()
    info=dict(support=len(support),attempted=False,reason='support_limit_or_empty')
    if not support or len(support)>64:return [],info
    info['attempted']=True;columns=[rational_row(reads,labels,index) for index in support];count=len(support)
    rows=[[column[j] for column in columns]+[Fraction(0)] for j in range(len(columns[0]))]
    rows.append([Fraction(1)]*count+[Fraction(1)])
    pivots=[];rank=0
    for col in range(count):
        pivot=next((i for i in range(rank,len(rows)) if rows[i][col]),None)
        if pivot is None:continue
        rows[rank],rows[pivot]=rows[pivot],rows[rank];factor=rows[rank][col];rows[rank]=[v/factor for v in rows[rank]]
        for i in range(len(rows)):
            if i==rank or not rows[i][col]:continue
            factor=rows[i][col];rows[i]=[a-factor*b for a,b in zip(rows[i],rows[rank])]
        pivots.append(col);rank+=1
        if rank==len(rows):break
    info['rank']=rank
    if any(not any(row[:-1]) and row[-1] for row in rows):info['reason']='inconsistent_exact_system';return [],info
    total=sum(float(dual[i]) for i in support)
    solution=[Fraction.from_float(float(dual[i])/total) for i in support]
    free=set(range(count))-set(pivots)
    for i,col in enumerate(pivots):solution[col]=rows[i][-1]-sum((rows[i][j]*solution[j] for j in free),Fraction(0))
    if min(solution)<0:info['reason']='negative_exact_weight';return [],info
    certificate=[dict(index=index,weight=str(weight)) for index,weight in zip(support,solution) if weight]
    if not certificate_valid(reads,labels,certificate):info['reason']='exact_check_failed';return [],info
    info['reason']='verified';return certificate,info


def certificate_valid(reads,labels,certificate):
    if not certificate:return False
    weights=[Fraction(row['weight']) for row in certificate]
    indices=[row['index'] for row in certificate]
    if len(set(indices))!=len(indices) or min(indices)<0 or max(indices)>=len(labels)*5:return False
    if min(weights)<0 or sum(weights)!=1:return False
    total=[Fraction(0)]*(5*(reads.shape[1]+1));strict=False
    for item,weight in zip(certificate,weights):
        coeff=rational_row(reads,labels,item['index']);total=[a+weight*b for a,b in zip(total,coeff)]
        sample,slot=divmod(item['index'],5);target=int(labels[sample]);other=[c for c in range(6) if c!=target][slot]
        strict|=bool(weight>0 and target>other)
    return strict and all(value==0 for value in total)


def decimal_margin(reads,w,b,labels,precision=60,guard='1e-45'):
    with localcontext() as context:
        context.prec=precision;D=Decimal.from_float;dw=[[D(float(v)) for v in row] for row in w];db=[D(float(v)) for v in b]
        minimum=None
        for vector,target in zip(reads,labels):
            x=[D(float(v)) for v in vector];scores=[];errors=[]
            for row,bias in zip(dw,db):
                scores.append(bias+sum((a*v for a,v in zip(row,x)),Decimal(0)))
                errors.append(Decimal(guard)*(1+abs(bias)+sum((abs(a*v) for a,v in zip(row,x)),Decimal(0))))
            target=int(target)
            for c in range(6):
                if c==target:continue
                lower=scores[target]-scores[c]-errors[target]-errors[c]
                minimum=lower if minimum is None else min(minimum,lower)
        return float(minimum)


def solve(reads,labels):
    a,rhs,cost=program(reads,labels);start=time.perf_counter()
    result=linprog(cost,A_ub=a,b_ub=rhs,bounds=[(None,None)]*len(cost),method='highs-ds',options=OPTIONS)
    elapsed=time.perf_counter()-start;dual=None if result.ineqlin.marginals is None else -result.ineqlin.marginals
    outcome='unresolved';head=None;certificate=[];reconstruction=dict(attempted=False,reason='not_needed');lower=None
    if result.x is not None and np.isfinite(result.x).all():
        parameters=np.vstack((np.zeros(reads.shape[1]+1),result.x[:-1].reshape(5,reads.shape[1]+1)))
        w=parameters[:,:-1];b=parameters[:,-1]
        lower=decimal_margin(reads,w,b,labels)
        if lower>MARGIN:outcome='verified_feasible';head=dict(weight=torch.from_numpy(w),bias=torch.from_numpy(b))
    if outcome!='verified_feasible' and dual is not None and np.isfinite(dual).all():
        certificate,reconstruction=reconstruct(reads,labels,dual)
        if certificate:outcome='certified_infeasible'
    residual=None if dual is None else float(np.max(np.abs(a.T@dual+cost)))
    record=dict(outcome=outcome,solver_status=int(result.status),solver_message=str(result.message),iterations=int(result.nit),
        solver_seconds=elapsed,solver_margin=None if result.x is None else float(result.x[-1]),
        positive_lower_margin=lower,approximate_dual_residual=residual,certificate_support=len(certificate),reconstruction=reconstruction,
        constraints=a.shape[0],variables=a.shape[1])
    return head,certificate,record,torch.from_numpy(dual) if dual is not None else None


@torch.no_grad()
def features(model,rows,pool):
    q,y,repeat=prior.labels(rows);e=torch.tensor([r['events'] for r in rows]);s=model.prefill(e)[0];read=model.keys(q)@s
    z=prior.historical.parent.forward_queries(model,e,q);torch.testing.assert_close(model.output(read),z,atol=1e-9,rtol=1e-9)
    return dict(pool=pool,states=s,queries=q,labels=y,repeat=repeat,reads=read,native_logits=z,native_metrics=prior.metrics(z,y,repeat))


@torch.no_grad()
def evaluate(model,pools,seed):
    before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    blocks=[features(model,rows,pool) for pool,rows in pools.items()];reads=torch.cat([p['reads'].flatten(0,1) for p in blocks]);labels=torch.cat([p['labels'].flatten() for p in blocks])
    head,certificate,record,dual=solve(reads.numpy(),labels.numpy());cells=[]
    for block in blocks:
        cells.append(dict(seed=seed,pool=block['pool'],arm='native',metrics=block['native_metrics']))
    if head:
        candidate=copy.deepcopy(model)
        with torch.no_grad():candidate.output.weight.copy_(head['weight']);candidate.output.bias.copy_(head['bias'])
        for block in blocks:
            with torch.no_grad():
                q=block['queries'];s=block['states'];n,m=q.shape
                z=candidate.read((s[:,None].expand(-1,m,-1,-1).reshape(n*m,16,8),),q.flatten()).reshape(n,m,6)
            torch.testing.assert_close(z,candidate.output(block['reads']),atol=1e-9,rtol=1e-9)
            assert bool((z.argmax(-1)==block['labels']).all());block['witness_logits']=z
            cells.append(dict(seed=seed,pool=block['pool'],arm='witness',metrics=prior.metrics(z,block['labels'],block['repeat'])))
        record['witness_weight_norm']=float(head['weight'].norm());record['witness_bias_norm']=float(head['bias'].norm())
    else:record.update(witness_weight_norm=None,witness_bias_norm=None)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()) and torch.equal(rng,torch.get_rng_state())
    return dict(seed=seed,blocks=blocks,head=head,certificate=certificate,dual=dual,record=record),cells


def execute(out,cases,fixture=False):
    assert not out.exists(),'Preserve evidence'
    files={p:prior.digest(ROOT/p) for p in (*SOURCES,PROTOCOL,'experiments/shared_state_feasibility.py','requirements-analysis.txt')}
    for case in cases:
        for name in ('checkpoint','data'):files[case[name]]=prior.digest(ROOT/case[name])
    if not fixture:
        pre=ROOT/'results/shared_readout_feasibility_preflight.json';result=json.loads(pre.read_text())
        assert result['passed'] and result['sources']=={p:prior.digest(ROOT/p) for p in SOURCES};files[pre.relative_to(ROOT).as_posix()]=prior.digest(pre)
        for folder in ('query_feedback_continuation','shared_state_feasibility','single_query_write_opposition'):
            source=ROOT/'results'/folder;report=json.loads((source/'report.json').read_text());audit=json.loads((source/'audit.json').read_text())
            assert audit['passed'] and audit['report_sha256']==prior.digest(source/'report.json')
            for name in ('lock.json','report.json','audit.json'):
                p=source/name;files[p.relative_to(ROOT).as_posix()]=prior.digest(p)
    prior.check_files(files);out.mkdir(parents=True)
    prior.write(out/'lock.json',dict(cases=cases,fixture=fixture,files=files,versions=prior.versions(),options=OPTIONS,torch_threads=torch.get_num_threads(),
        solver_threads='HiGHS default',feature_contract='stored binary float64 reads interpreted exactly'))
    start=time.perf_counter();bundles=[];cells=[]
    for case in cases:
        payload=json.loads((ROOT/case['data']).read_text(encoding='utf-8-sig'));pools={p:payload['pools'][p] for p in case['pools']}
        model=prior.historical.parent.previous.load_model(ROOT/case['checkpoint']).eval();bundle,records=evaluate(model,pools,case['seed'])
        bundles.append(bundle);cells.extend(records);torch.save(bundles,out/'witnesses.pt')
        print(json.dumps(dict(seed=case['seed'],record=bundle['record'])),flush=True)
    flat=[]
    for cell in cells:
        row={k:v for k,v in cell.items() if k!='metrics'}
        for group,stats in cell['metrics'].items():
            if group=='complete_maps':row[group]=stats
            else:
                for name,value in stats.items():row[group+'_'+name]=value
        flat.append(row)
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    report=dict(status='complete',fixture=fixture,cells=cells,summary=[dict(seed=b['seed'],**b['record']) for b in bundles],
        histories=sum(len(p['labels']) for b in bundles for p in b['blocks']),native_queries=sum(p['labels'].numel() for b in bundles for p in b['blocks']),
        witness_queries=sum(p['labels'].numel() for b in bundles if b['head'] for p in b['blocks']),
        primal_solver_calls=len(bundles),training_updates=0,optimizer_calls=0,device='cpu',dtype='float64',
        seconds=time.perf_counter()-start,lock_sha256=prior.digest(out/'lock.json'),
        witnesses_sha256=prior.digest(out/'witnesses.pt'),metrics_sha256=prior.digest(out/'metrics.csv'))
    prior.check_files(files);prior.write(out/'report.json',report)


def preflight():
    spec=importlib.util.spec_from_file_location('test_shared_readout_feasibility',ROOT/SOURCES[-1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if result.wasSuccessful():
        p=ROOT/'results/shared_readout_feasibility_preflight.json';assert not p.exists()
        prior.write(p,dict(passed=True,methods=result.testsRun,sources={p:prior.digest(ROOT/p) for p in SOURCES},versions=prior.versions()))
    assert result.wasSuccessful()


if __name__=='__main__':
    torch.set_num_threads(1);parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    if parser.parse_args().preflight:preflight()
    else:execute(OUT,[dict(seed=c['seed'],checkpoint=c['checkpoint'],data=c['data'],pools=[f'train_{m}' for m in LOADS]) for c in prior.inputs()])
