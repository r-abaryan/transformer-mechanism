"""Fixed weight-projection decoder interventions, reusing audited source logits."""
import argparse
import itertools
import json
import math
import time

import torch
import fitting_representation as previous
from audit_shared_state_feasibility import compare

ROOT=previous.ROOT
OUT=ROOT/'results/decoder_scale_shape'
SOURCES=('experiments/decoder_scale_shape.py','tests/test_decoder_scale_shape.py')
PROTOCOL='docs/protocols/decoder-scale-shape.md'
VARIANTS=('base','scale','shape','full')
replay=previous.replay


def centered_map(state):
    b=torch.cat((state['output.weight']@state['value.weight'].T,state['output.bias'][:,None]),1)
    return b-b.mean(0,keepdim=True)


def projection(states):
    L,N=[centered_map(states[arm]) for arm in previous.ARMS]
    scale=float((L*N).sum()/N.square().sum())
    assert math.isfinite(scale) and scale>0 and float(N.norm())>0 and float(L.norm())>0,'Invalid positive projection'
    return dict(scale=scale,relative_residual=float((L-scale*N).norm()/L.norm()))


def transform(negative,lbfgs,scale):
    assert math.isfinite(scale) and scale>0
    n=negative-negative.mean(-1,keepdim=True);l=lbfgs-lbfgs.mean(-1,keepdim=True)
    return dict(base=n,scale=n*scale,shape=l/scale,full=l)


def effects(cells):
    result=[]
    keys=sorted({(c['seed'],c['routing'],c['pool']) for c in cells})
    for seed,routing,pool in keys:
        subset={c['variant']:c['metrics']['groups']['all']['ce'] for c in cells if (c['seed'],c['routing'],c['pool'])==(seed,routing,pool)}
        assert set(subset)==set(VARIANTS)
        a,b,c,d=[subset[v] for v in VARIANTS]
        first=b-a;last=d-c;shape_first=c-a;shape_last=d-b;total=d-a
        result.append(dict(seed=seed,routing=routing,pool=pool,base=a,scale=b,shape=c,full=d,total_penalty=total,
            scale_at_base_shape=first,scale_at_lbfgs_shape=last,shape_at_base_scale=shape_first,shape_at_lbfgs_scale=shape_last,
            interaction=d-b-c+a,scale_first_ratio=first/total if abs(total)>1e-8 else None,
            scale_last_ratio=last/total if abs(total)>1e-8 else None,
            scale_dominates=total>1e-8 and min(first,last)>max(abs(shape_first),abs(shape_last))))
    primary=[r for r in result if r['pool'] in ('validation_32','test_32')]
    assert len(primary)==4*len({c['seed'] for c in cells})
    return dict(contrasts=result,primary=primary,all_primary_scale_dominates=all(r['scale_dominates'] for r in primary))


def inputs():
    cases,pools=previous.inputs();source=torch.load(previous.OUT/'scores.pt',weights_only=True)
    return cases,pools,source


@torch.no_grad()
def execute(out,cases,pools,source,fixture=False):
    assert not out.exists(),'Preserve evidence'
    files={n:replay.digest(ROOT/n) for n in previous.previous.prior.dependencies()|set(SOURCES)}
    for c in cases:
        for n in c['checkpoints'].values():files[n]=replay.digest(ROOT/n)
    for n in {c['checkpoint'] for c in source}:files[n]=replay.digest(ROOT/n)
    if not fixture:
        assert [c['seed'] for c in cases]==list(previous.previous.SEEDS)
        for n in (PROTOCOL,'results/decoder_scale_shape_preflight.json','results/fitting_representation/report.json','results/fitting_representation/audit.json','results/fitting_representation/scores.pt','results/fitting_representation/pools.pt'):
            files[n]=replay.digest(ROOT/n)
        audit=json.loads((previous.OUT/'audit.json').read_text());assert audit['passed'] and audit['report_sha256']==replay.digest(previous.OUT/'report.json')
        pre=json.loads((ROOT/'results/decoder_scale_shape_preflight.json').read_text())
        assert pre['passed'] and pre['sources']=={n:files[n] for n in SOURCES}
    out.mkdir(parents=True);torch.save(pools,out/'pools.pt');torch.save(source,out/'source_scores.pt')
    replay.write(out/'lock.json',dict(files=files,cases=cases,fixture=fixture,torch=torch.__version__,pools_sha256=replay.digest(out/'pools.pt'),source_scores_sha256=replay.digest(out/'source_scores.pt')))
    cells=[];projections=[];started=time.perf_counter()
    for case in cases:
        states={a:torch.load(ROOT/p,weights_only=True)['model'] for a,p in case['checkpoints'].items()}
        record=projection(states);projections.append(dict(seed=case['seed'],**record))
        for routing,split,load in itertools.product(previous.ARMS,('train','validation','test'),previous.LOADS):
            pool=f'{split}_{load}'
            parents={c['decoder']:c for c in source if (c['seed'],c['routing'],c['pool'])==(case['seed'],routing,pool)}
            assert set(parents)==set(previous.ARMS)
            l,n=[parents[a] for a in previous.ARMS]
            assert torch.equal(l['labels'],n['labels']) and torch.equal(l['repeat'],n['repeat'])
            scores=transform(n['logits'],l['logits'],record['scale'])
            for variant,z in scores.items():
                cell=dict(seed=case['seed'],routing=routing,variant=variant,pool=pool,logits=z,labels=n['labels'],repeat=n['repeat'],
                          metrics=previous.training.reduce_scores(z,n['labels'],n['repeat']))
                cells.append(cell)
            assert torch.equal(scores['base'].argmax(-1),scores['scale'].argmax(-1)) and torch.equal(scores['shape'].argmax(-1),scores['full'].argmax(-1)),'Scale changed decisions'
        print(json.dumps(dict(seed=case['seed'],**record)),flush=True)
    torch.save(cells,out/'scores.pt');replay.check_files(files)
    replay.write(out/'report.json',dict(status='complete',projections=projections,summary=effects(cells),
        cells=[{k:v for k,v in c.items() if k not in ('logits','labels','repeat')} for c in cells],evaluation_targets=sum(c['labels'].numel() for c in cells),
        optimizer_calls=0,gradient_calls=0,solver_calls=0,producer_model_forwards=0,seconds=time.perf_counter()-started,device='cpu',dtype='float64',threads=1,
        lock_sha256=replay.digest(out/'lock.json'),scores_sha256=replay.digest(out/'scores.pt')))


def independent_projection(states):
    centered=[]
    for arm in previous.ARMS:
        state=states[arm];w=state['output.weight'].tolist();v=state['value.weight'].tolist();b=state['output.bias'].tolist()
        rows=[[math.fsum(x*y for x,y in zip(row,value)) for value in v]+[bias] for row,bias in zip(w,b)]
        means=[math.fsum(row[j] for row in rows)/len(rows) for j in range(7)]
        centered.append([x-means[j] for row in rows for j,x in enumerate(row)])
    l,n=centered;scale=math.fsum(x*y for x,y in zip(l,n))/math.fsum(x*x for x in n)
    residual=math.sqrt(math.fsum((x-scale*y)**2 for x,y in zip(l,n))/math.fsum(x*x for x in l))
    return scale,residual


@torch.no_grad()
def audit(out,persist=True):
    lock=json.loads((out/'lock.json').read_text());report=json.loads((out/'report.json').read_text());replay.check_files(lock['files'])
    assert report['status']=='complete' and lock['torch']==torch.__version__
    for key,name in [('pools_sha256','pools.pt'),('source_scores_sha256','source_scores.pt')]:assert lock[key]==replay.digest(out/name)
    assert report['lock_sha256']==replay.digest(out/'lock.json') and report['scores_sha256']==replay.digest(out/'scores.pt')
    pools=torch.load(out/'pools.pt',weights_only=True);source=torch.load(out/'source_scores.pt',weights_only=True);scores=torch.load(out/'scores.pt',weights_only=True)
    if not lock['fixture']:
        cases,old,original=inputs();assert cases==lock['cases'] and old==pools and replay.equal(source,original)
    source_ids=[(c['seed'],c['routing'],c['decoder'],c['pool']) for c in source]
    wanted={(c['seed'],r,d,f'{split}_{m}') for c,r,d,split,m in itertools.product(lock['cases'],previous.ARMS,previous.ARMS,('train','validation','test'),previous.LOADS)}
    assert len(source_ids)==len(wanted) and set(source_ids)==wanted,'Missing source group'
    donors={c['seed']:{a:torch.load(ROOT/p,weights_only=True)['model'] for a,p in c['checkpoints'].items()} for c in lock['cases']}
    maximum=0.
    for cell in source:
        actual=torch.load(ROOT/cell['checkpoint'],weights_only=True)['model']
        for name,value in actual.items():
            donor=cell['routing'] if name in previous.ROUTING else cell['decoder']
            assert torch.equal(value,donors[cell['seed']][donor][name]),'Source donor scope'
        with torch.random.fork_rng(devices=[]):check=previous.reference.audit_cell(ROOT/cell['checkpoint'],pools[cell['pool']],cell)
        maximum=max(maximum,check['max_logit_error'])
    expected={(c['seed'],r,v,f'{split}_{m}') for c,r,v,split,m in itertools.product(lock['cases'],previous.ARMS,VARIANTS,('train','validation','test'),previous.LOADS)}
    ids=[(c['seed'],c['routing'],c['variant'],c['pool']) for c in scores]
    assert len(ids)==len(expected) and set(ids)==expected,'Missing transformed group'
    assert len(report['cells'])==len(scores) and len(report['projections'])==len(lock['cases'])
    scales={}
    for case in lock['cases']:
        states={a:torch.load(ROOT/p,weights_only=True)['model'] for a,p in case['checkpoints'].items()}
        scalar,residual=independent_projection(states);saved=next(p for p in report['projections'] if p['seed']==case['seed'])
        assert abs(scalar-saved['scale'])<=1e-12 and abs(residual-saved['relative_residual'])<=1e-12,'Weight projection'
        assert scalar>0;scales[case['seed']]=saved['scale']
    targets=0;transform_error=0.
    for cell,record in zip(scores,report['cells']):
        compare(record,{k:v for k,v in cell.items() if k not in ('logits','labels','repeat')})
        donor='negative_gradient' if cell['variant'] in ('base','scale') else 'lbfgs'
        parent=next(c for c in source if (c['seed'],c['routing'],c['decoder'],c['pool'])==(cell['seed'],cell['routing'],donor,cell['pool']))
        assert torch.equal(cell['labels'],parent['labels']) and torch.equal(cell['repeat'],parent['repeat']),'Transformed labels'
        # Center relative to class zero instead of using the producer mean.
        shifted=parent['logits']-parent['logits'][...,:1];wanted=shifted-shifted.sum(-1,keepdim=True)/6
        if cell['variant']=='scale':wanted=wanted*scales[cell['seed']]
        if cell['variant']=='shape':wanted=wanted/scales[cell['seed']]
        transform_error=max(transform_error,previous.reference.independent.close(cell['logits'],wanted))
        metrics=previous.reference.metrics(cell['logits'],cell['labels'],cell['repeat']);previous.reference.check_metrics(cell['metrics'],metrics)
        if cell['variant'] in ('base','full'):
            previous.reference.check_metrics(cell['metrics'],parent['metrics'])
        assert torch.equal(cell['logits'].argmax(-1),parent['logits'].argmax(-1)),'Positive scale decisions'
        targets+=cell['labels'].numel()
    compare(report['summary'],effects(scores))
    # Reconstruct the primary conjunction independently from scalar CE metrics.
    decisions=[]
    for case,routing,split in itertools.product(lock['cases'],previous.ARMS,('validation','test')):
        vals={c['variant']:c['metrics']['groups']['all']['ce'] for c in scores if (c['seed'],c['routing'],c['pool'])==(case['seed'],routing,f'{split}_32')}
        base=vals['base'];scaled=vals['scale'];shape=vals['shape'];full=vals['full']
        passed=full-base>1e-8 and all(x>abs(y) for x in (scaled-base,full-shape) for y in (shape-base,full-scaled))
        saved=next(x for x in report['summary']['primary'] if (x['seed'],x['routing'],x['pool'])==(case['seed'],routing,f'{split}_32'))
        assert saved['scale_dominates']==passed,'Primary decision'
        decisions.append(passed)
    assert report['summary']['all_primary_scale_dominates']==all(decisions)
    assert report['evaluation_targets']==targets and (report['optimizer_calls'],report['gradient_calls'],report['solver_calls'],report['producer_model_forwards'])==(0,0,0,0)
    result=dict(passed=True,cells=len(scores),source_cells=len(source),evaluation_targets=targets,max_source_logit_error=maximum,
                max_transformation_error=transform_error,protected_files=len(lock['files']),independent_primary_passes=decisions,report_sha256=replay.digest(out/'report.json'))
    if persist:
        assert not (out/'audit.json').exists();replay.write(out/'audit.json',result)
    return result


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    if args.audit:print(json.dumps(audit(OUT)),flush=True)
    else:execute(OUT,*inputs())
