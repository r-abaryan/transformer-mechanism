"""Full-coordinate search recovery, with proofs checked on original reads."""
import argparse
import copy
import csv
import importlib.util
import json
import math
import time
import unittest
from decimal import Decimal,localcontext
import numpy as np
import torch
from scipy.optimize import linprog
import shared_readout_feasibility as original

ROOT=original.ROOT
OUT=ROOT/'results/shared_readout_recovery'
PROTOCOL='docs/protocols/shared-readout-numerical-recovery.md'
SOURCES=('experiments/shared_readout_recovery.py','experiments/audit_shared_readout_recovery.py',
         'tests/test_shared_readout_recovery.py')
OPTIONS=dict(original.OPTIONS,disp=True)


def transform(reads):
    mu=reads.mean(0);_,singular,vh=np.linalg.svd(reads-mu,full_matrices=False)
    assert reads.shape[1]==8 and singular.shape==(8,) and np.all(singular>0) and np.isfinite(singular).all()
    exponent=np.floor(np.log2(singular/math.sqrt(len(reads)))).astype(int)
    scale=np.ldexp(np.ones(8),exponent);t=vh.T/scale
    assert np.isfinite(t).all() and np.linalg.slogdet(vh)[0]!=0
    with localcontext() as context:
        context.prec=80;D=Decimal.from_float
        mean=[D(float(v)) for v in mu];columns=[[D(float(v)) for v in t[:,j]] for j in range(8)]
        projected=[]
        for row in reads:
            centered=[D(float(v))-m for v,m in zip(row,mean)]
            projected.append([float(sum((v*a for v,a in zip(centered,col)),Decimal(0))) for col in columns])
    x=np.asarray(projected);assert np.isfinite(x).all()
    return dict(mean=torch.from_numpy(mu),matrix=torch.from_numpy(t),singular_values=singular.tolist(),
                exponents=exponent.tolist(),transformed=torch.from_numpy(x))


def map_head(parameters,coordinate):
    u=np.vstack((np.zeros(9),parameters[:-1].reshape(5,9)))
    with localcontext() as context:
        context.prec=80;D=Decimal.from_float
        t=[[D(float(v)) for v in row] for row in coordinate['matrix']];mu=[D(float(v)) for v in coordinate['mean']]
        weights=[];bias=[]
        for row in u:
            w=[sum((v*D(float(a)) for v,a in zip(line,row[:-1])),Decimal(0)) for line in t]
            weights.append([float(v) for v in w]);bias.append(float(D(float(row[-1]))-sum((v*m for v,m in zip(w,mu)),Decimal(0))))
    return dict(weight=torch.tensor(weights,dtype=torch.float64),bias=torch.tensor(bias,dtype=torch.float64))


def pool_support(certificate,blocks):
    bounds=[];offset=0
    for block in blocks:
        n=block['labels'].numel();bounds.append((offset,offset+n,block['pool']));offset+=n
    result={p:0 for _,_,p in bounds};classes={str(c):0 for c in range(6)}
    labels=torch.cat([b['labels'].flatten() for b in blocks])
    for item in certificate:
        sample=item['index']//5
        result[next(p for lo,hi,p in bounds if lo<=sample<hi)]+=1
        classes[str(int(labels[sample]))]+=1
    return dict(pools=result,labels=classes)


@torch.no_grad()
def evaluate(bundle,case):
    x=torch.cat([b['reads'].flatten(0,1) for b in bundle['blocks']]).numpy()
    y=torch.cat([b['labels'].flatten() for b in bundle['blocks']]).numpy()
    model=original.prior.historical.parent.previous.load_model(ROOT/case['checkpoint']).eval()
    values=model.value.weight.detach().numpy();_,vsv,vh=np.linalg.svd(values,full_matrices=True)
    geometry=dict(read_range=[float(x.min()),float(x.max())],coordinate_sd=x.std(0).tolist(),value_singular_values=vsv.tolist(),
                  max_outside_value_span=float(np.abs(x@vh[6:].T).max()))
    record=dict(outcome='unresolved',solver_calls=0,solver_status=None,solver_message='transform unavailable',iterations=0,
                solver_seconds=0.,solver_margin=None,positive_lower_margin=None,certificate_support=0,
                reconstruction=dict(attempted=False,reason='not_needed'),candidate_queries=0,witness_queries=0,
                native_candidate_all_correct=None,constraints=len(y)*5+1,variables=46)
    try:coordinate=transform(x)
    except (AssertionError,np.linalg.LinAlgError,OverflowError,ValueError) as error:
        record['solver_message']=type(error).__name__+': '+str(error)
        return dict(seed=bundle['seed'],geometry=geometry,coordinate=None,candidate=None,primal=None,dual=None,
                    certificate=[],record=record,cells=[])
    a,rhs,cost=original.program(coordinate['transformed'].numpy(),y);start=time.perf_counter()
    result=linprog(cost,A_ub=a,b_ub=rhs,bounds=[(None,None)]*46,method='highs-ds',options=OPTIONS)
    dual=None if result.ineqlin.marginals is None else -result.ineqlin.marginals
    record.update(solver_calls=1,solver_status=int(result.status),solver_message=str(result.message),iterations=int(result.nit),
                  solver_seconds=time.perf_counter()-start,solver_margin=None if result.x is None else float(result.x[-1]))
    candidate=None;certificate=[];cells=[]
    if result.x is not None and np.isfinite(result.x).all():
        candidate=map_head(result.x,coordinate)
        if bool(torch.isfinite(candidate['weight']).all()) and bool(torch.isfinite(candidate['bias']).all()):
            lower=original.decimal_margin(x,candidate['weight'],candidate['bias'],y);record['positive_lower_margin']=lower
            private=copy.deepcopy(model);private.output.weight.copy_(candidate['weight']);private.output.bias.copy_(candidate['bias'])
            predictions=[]
            for block in bundle['blocks']:
                q=block['queries'];n,m=q.shape;s=block['states'][:,None].expand(-1,m,-1,-1).reshape(n*m,16,8)
                z=private.read((s,),q.flatten()).reshape(n,m,6);predictions.append(z)
                cells.append(dict(seed=bundle['seed'],pool=block['pool'],arm='candidate',metrics=original.prior.metrics(z,block['labels'],block['repeat'])))
            record['candidate_queries']=len(y);record['native_candidate_all_correct']=all(bool((z.argmax(-1)==b['labels']).all()) for z,b in zip(predictions,bundle['blocks']))
            candidate['logits']=predictions
            if lower>original.MARGIN and record['native_candidate_all_correct']:
                record.update(outcome='verified_feasible',witness_queries=len(y))
    if record['outcome']!='verified_feasible' and dual is not None and np.isfinite(dual).all():
        certificate,reconstruction=original.reconstruct(x,y,dual);record['reconstruction']=reconstruction
        if certificate:record['outcome']='certified_infeasible'
    record['certificate_support']=len(certificate);record['support_groups']=pool_support(certificate,bundle['blocks'])
    return dict(seed=bundle['seed'],geometry=geometry,coordinate=coordinate,candidate=candidate,
                primal=None if result.x is None else torch.from_numpy(result.x),dual=None if dual is None else torch.from_numpy(dual),
                certificate=certificate,record=record,cells=cells)


def execute(out,origin=original.OUT,fixture=False):
    assert not out.exists(),'Preserve executed evidence'
    prior=original.prior;oldlock=json.loads((origin/'lock.json').read_text());oldreport=json.loads((origin/'report.json').read_text())
    oldaudit=json.loads((origin/'audit.json').read_text());assert oldaudit['passed'] and oldaudit['report_sha256']==prior.digest(origin/'report.json')
    prior.check_files(oldlock['files']);files=dict(oldlock['files'])
    for path in (*SOURCES,PROTOCOL,'experiments/audit_shared_readout_feasibility.py'):
        files[path]=prior.digest(ROOT/path)
    for name in ('lock.json','report.json','audit.json','witnesses.pt','metrics.csv'):
        p=origin/name;files[p.relative_to(ROOT).as_posix()]=prior.digest(p)
    if not fixture:
        p=ROOT/'results/shared_readout_recovery_preflight.json';pre=json.loads(p.read_text())
        assert pre['passed'] and pre['sources']=={p:prior.digest(ROOT/p) for p in SOURCES};files[p.relative_to(ROOT).as_posix()]=prior.digest(p)
    prior.check_files(files);out.mkdir(parents=True)
    prior.write(out/'lock.json',dict(files=files,origin=origin.relative_to(ROOT).as_posix(),cases=oldlock['cases'],fixture=fixture,
        options=OPTIONS,versions=prior.versions(),torch_threads=torch.get_num_threads(),solver_threads='HiGHS default'))
    oldbundles=torch.load(origin/'witnesses.pt',weights_only=True);bundles=[];start=time.perf_counter()
    for source,case in zip(oldbundles,oldlock['cases']):
        bundle=evaluate(source,case);bundles.append(bundle);torch.save(bundles,out/'witnesses.pt')
        print(json.dumps(dict(seed=bundle['seed'],record=bundle['record'])),flush=True)
    cells=[cell for b in bundles for cell in b['cells']];flat=[]
    for cell in cells:
        row={k:v for k,v in cell.items() if k!='metrics'}
        for group,stats in cell['metrics'].items():
            if group=='complete_maps':row[group]=stats
            else:
                for k,v in stats.items():row[group+'_'+k]=v
        flat.append(row)
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        if flat:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    report=dict(status='complete',summary=[dict(seed=b['seed'],geometry=b['geometry'],**b['record']) for b in bundles],cells=cells,
                original_queries=oldreport['native_queries'],histories=oldreport['histories'],native_reference=oldreport['cells'],
                solver_calls=sum(b['record']['solver_calls'] for b in bundles),training_updates=0,optimizer_calls=0,native_prefills=0,
                seconds=time.perf_counter()-start,lock_sha256=prior.digest(out/'lock.json'),witnesses_sha256=prior.digest(out/'witnesses.pt'),
                metrics_sha256=prior.digest(out/'metrics.csv'))
    prior.check_files(files);prior.write(out/'report.json',report)


def preflight():
    spec=importlib.util.spec_from_file_location('checks',ROOT/SOURCES[-1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if result.wasSuccessful():
        p=ROOT/'results/shared_readout_recovery_preflight.json';assert not p.exists()
        original.prior.write(p,dict(passed=True,methods=result.testsRun,sources={p:original.prior.digest(ROOT/p) for p in SOURCES}))
    assert result.wasSuccessful()


if __name__=='__main__':
    torch.set_num_threads(1);parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    if parser.parse_args().preflight:preflight()
    else:execute(OUT)
