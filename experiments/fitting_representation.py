"""Saved endpoint routing versus paired value/classifier donors; no training."""
import argparse
import copy
import itertools
import json
import time

import torch
from torch.nn import functional as F
import fitting_direction as previous
import audit_query_feedback_training as reference
from audit_shared_state_feasibility import compare

ROOT=previous.ROOT
OUT=ROOT/'results/fitting_representation'
SOURCES=('experiments/fitting_representation.py','tests/test_fitting_representation.py')
PROTOCOL='docs/protocols/fitting-representation.md'
ARMS=previous.ARMS
LOADS=previous.prior.LOADS
replay=previous.replay
training=previous.prior.prior.training
ROUTING={'write_logit','key.weight'}
DECODER={'value.weight','output.weight','output.bias'}


def model(path):
    with torch.random.fork_rng(devices=[]):return training.previous.load_model(ROOT/path).eval()


def hybrid(donors,routing,decoder):
    result=copy.deepcopy(donors[decoder]);state={}
    assert set(result.state_dict())==ROUTING|DECODER
    for name in result.state_dict():state[name]=donors[routing if name in ROUTING else decoder].state_dict()[name].clone()
    result.load_state_dict(state)
    return result


@torch.no_grad()
def factorized(reader,events,queries):
    """Local exact factorization S=C V, with one basis vector per value ID."""
    keys=reader.keys(events[...,0]);q=reader.keys(queries);beta=reader.write_logit.sigmoid()
    basis=F.one_hot(events[...,1],6).to(torch.float64);c=keys.new_zeros(len(events),16,6)
    for t in range(events.shape[1]):
        k=keys[:,t];old=torch.einsum('bk,bkj->bj',k,c)
        c=c+beta*k[:,:,None]*(basis[:,t]-old)[:,None,:]
    coefficients=torch.einsum('bmk,bkj->bmj',q,c)
    effective=reader.output.weight@reader.value.weight.T
    return coefficients@effective.T+reader.output.bias


def parameter_stats(state,start):
    k=F.normalize(state['key.weight'],dim=-1,eps=1e-12);gram=k@k.T
    # Class centering removes logits common to all classes. Include bias.
    def effective(s):
        a=torch.cat((s['output.weight']@s['value.weight'].T,s['output.bias'][:,None]),1)
        return a-a.mean(0,keepdim=True)
    a=effective(state);base=effective(start);scale=float((a*base).sum()/base.square().sum())
    residual=float((a-scale*base).norm()/a.norm())
    return dict(beta=float(state['write_logit'].sigmoid()),key_overlap_rms=float((gram-torch.eye(len(k),dtype=k.dtype)).square().sum().sqrt()/(len(k)*(len(k)-1))**.5),
                decoder_norm=float((state['output.weight']@state['value.weight'].T).norm()),
                centered_decoder_scale=scale,centered_decoder_relative_residual=residual)


def summarize(cells,seeds):
    rows=[];contrasts=[]
    for seed in seeds:
        stats={}
        for routing,decoder in itertools.product(ARMS,repeat=2):
            selected=[c for c in cells if c['seed']==seed and c['routing']==routing and c['decoder']==decoder and c['pool'].startswith('train_')]
            assert {c['pool'] for c in selected}=={f'train_{m}' for m in LOADS}
            stats[routing,decoder]=dict(objective=sum(c['metrics']['groups']['all']['ce'] for c in selected)/4,
                errors=sum(round(c['metrics']['groups']['all']['count']*(1-c['metrics']['groups']['all']['accuracy'])) for c in selected))
        ll=stats[ARMS[0],ARMS[0]];ln=stats[ARMS[0],ARMS[1]];nl=stats[ARMS[1],ARMS[0]];nn=stats[ARMS[1],ARMS[1]]
        rows.append(dict(seed=seed,native_lbfgs=ll,native_negative_gradient=nn,routing_only=ln,decoder_only=nl,
                         routing_sufficient=ln['errors']<=ll['errors'],decoder_sufficient=nl['errors']<=ll['errors']))
        for split in ('train','validation','test'):
            for load in LOADS:
                values={(c['routing'],c['decoder']):c['metrics']['groups']['all'] for c in cells if c['seed']==seed and c['pool']==f'{split}_{load}'}
                assert len(values)==4
                record=dict(seed=seed,pool=f'{split}_{load}')
                for field in ('accuracy','ce'):
                    a,b,c,d=[values[r,dec][field] for r,dec in ((ARMS[1],ARMS[1]),(ARMS[0],ARMS[1]),(ARMS[1],ARMS[0]),(ARMS[0],ARMS[0]))]
                    record[field]=dict(routing_negative_decoder=b-a,routing_lbfgs_decoder=d-c,
                                      decoder_negative_routing=c-a,decoder_lbfgs_routing=d-b,interaction=d-b-c+a)
                contrasts.append(record)
    return dict(seeds=rows,all_seed_routing_sufficient=all(r['routing_sufficient'] for r in rows),
                all_seed_decoder_sufficient=all(r['decoder_sufficient'] for r in rows),contrasts=contrasts)


def inputs():
    report=json.loads((previous.OUT/'report.json').read_text())
    cases=[dict(seed=s,checkpoints={arm:next(r['final_checkpoint'] for r in report['runs'] if r['seed']==s and r['arm']==arm) for arm in ARMS}) for s in previous.SEEDS]
    return cases,torch.load(previous.OUT/'pools.pt',weights_only=True)


@torch.no_grad()
def execute(out,cases,pools,fixture=False):
    assert not out.exists(),'Preserve evidence'
    files={name:replay.digest(ROOT/name) for name in previous.prior.dependencies()|set(SOURCES)}
    for case in cases:
        for path in case['checkpoints'].values():files[path]=replay.digest(ROOT/path)
    if not fixture:
        assert [c['seed'] for c in cases]==list(previous.SEEDS)
        for name in (PROTOCOL,'results/fitting_representation_preflight.json','results/fitting_direction/report.json','results/fitting_direction/audit.json','results/fitting_direction/pools.pt','results/fitting_direction/scores.pt'):
            files[name]=replay.digest(ROOT/name)
        prior_report=json.loads((previous.OUT/'report.json').read_text());pre=json.loads((ROOT/'results/fitting_representation_preflight.json').read_text())
        assert pre['passed'] and pre['sources']=={name:files[name] for name in SOURCES}
        for r in prior_report['runs']:
            for snap in r['snapshots']:files[snap['checkpoint']]=replay.digest(ROOT/snap['checkpoint'])
    replay.check_files(files);out.mkdir(parents=True);torch.save(pools,out/'pools.pt')
    replay.write(out/'lock.json',dict(files=files,cases=cases,fixture=fixture,pools_sha256=replay.digest(out/'pools.pt'),torch=torch.__version__))
    cells=[];trajectories=[];started=time.perf_counter()
    for case in cases:
        donors={arm:model(path) for arm,path in case['checkpoints'].items()}
        for routing,decoder in itertools.product(ARMS,repeat=2):
            reader=hybrid(donors,routing,decoder);path=out/f"{case['seed']}_{routing}_{decoder}.pt"
            torch.save(dict(model=reader.state_dict()),path)
            for split,load in itertools.product(('train','validation','test'),LOADS):
                pool=f'{split}_{load}';cell=training.evaluate(reader,pools[pool])
                cell.update(seed=case['seed'],routing=routing,decoder=decoder,pool=pool,checkpoint=path.relative_to(ROOT).as_posix())
                cells.append(cell)
            print(json.dumps(dict(seed=case['seed'],routing=routing,decoder=decoder)),flush=True)
        if not fixture:
            for run in [r for r in prior_report['runs'] if r['seed']==case['seed']]:
                base=torch.load(ROOT/run['snapshots'][0]['checkpoint'],weights_only=True)['model']
                for snap in run['snapshots']:
                    state=torch.load(ROOT/snap['checkpoint'],weights_only=True)['model']
                    trajectories.append(dict(seed=run['seed'],arm=run['arm'],closures=snap['step'],checkpoint=snap['checkpoint'],**parameter_stats(state,base)))
    torch.save(cells,out/'scores.pt');summary=summarize(cells,[c['seed'] for c in cases]);replay.check_files(files)
    replay.write(out/'report.json',dict(status='complete',cells=[{k:v for k,v in c.items() if k not in ('logits','labels','repeat')} for c in cells],
                 summary=summary,trajectories=trajectories,evaluation_targets=sum(c['labels'].numel() for c in cells),hybrids=4*len(cases),
                 optimizer_calls=0,gradient_calls=0,solver_calls=0,seconds=time.perf_counter()-started,
                 device='cpu',dtype='float64',threads=1,lock_sha256=replay.digest(out/'lock.json'),scores_sha256=replay.digest(out/'scores.pt')))


@torch.no_grad()
def audit(out,persist=True):
    lock=json.loads((out/'lock.json').read_text());report=json.loads((out/'report.json').read_text());replay.check_files(lock['files'])
    assert report['status']=='complete' and lock['torch']==torch.__version__
    assert lock['pools_sha256']==replay.digest(out/'pools.pt')
    assert (report['lock_sha256'],report['scores_sha256'])==(replay.digest(out/'lock.json'),replay.digest(out/'scores.pt'))
    pools=torch.load(out/'pools.pt',weights_only=True);scores=torch.load(out/'scores.pt',weights_only=True)
    if not lock['fixture']:
        cases,original=inputs();assert cases==lock['cases'] and pools==original
        old=torch.load(previous.OUT/'scores.pt',weights_only=True);old_report=json.loads((previous.OUT/'report.json').read_text())
    expected={(c['seed'],r,d,f'{split}_{m}') for c,r,d,split,m in itertools.product(lock['cases'],ARMS,ARMS,('train','validation','test'),LOADS)}
    ids=[(c['seed'],c['routing'],c['decoder'],c['pool']) for c in scores]
    assert len(ids)==len(expected) and set(ids)==expected,'Missing cell'
    assert len(report['cells'])==len(scores);maximum=0.;targets=0
    for case in lock['cases']:
        donors={arm:model(path) for arm,path in case['checkpoints'].items()}
        for cell,saved in [(c,x) for c,x in zip(scores,report['cells']) if c['seed']==case['seed']]:
            compare(saved,{k:v for k,v in cell.items() if k not in ('logits','labels','repeat')})
            actual=model(cell['checkpoint']);state=actual.state_dict()
            for name in state:
                donor=cell['routing'] if name in ROUTING else cell['decoder']
                assert torch.equal(state[name],donors[donor].state_dict()[name]),'Wrong parameter donor'
            with torch.random.fork_rng(devices=[]):check=reference.audit_cell(ROOT/cell['checkpoint'],pools[cell['pool']],cell)
            maximum=max(maximum,check['max_logit_error']);targets+=check['targets']
            if not lock['fixture'] and cell['routing']==cell['decoder']:
                run=next(r for r in old_report['runs'] if (r['seed'],r['arm'])==(case['seed'],cell['routing']))
                native=next(c for c in old if (c['seed'],c['arm'],c['step'],c['pool'])==(case['seed'],cell['routing'],run['snapshots'][-1]['step'],cell['pool']))
                assert torch.equal(cell['logits'],native['logits']),'Native self-control'
                compare(cell['metrics'],native['metrics'])
    # Independent training-error count from logits, not rounded report accuracy.
    decisions=[]
    for case in lock['cases']:
        errors={}
        for r,d in itertools.product(ARMS,repeat=2):
            errors[r,d]=sum(int((c['logits'].argmax(-1)!=c['labels']).sum()) for c in scores if c['seed']==case['seed'] and c['routing']==r and c['decoder']==d and c['pool'].startswith('train_'))
        decisions.append(dict(seed=case['seed'],routing_sufficient=errors[ARMS[0],ARMS[1]]<=errors[ARMS[0],ARMS[0]],decoder_sufficient=errors[ARMS[1],ARMS[0]]<=errors[ARMS[0],ARMS[0]]))
    for actual,want in zip(report['summary']['seeds'],decisions):
        for key in want:assert actual[key]==want[key],'Sufficiency decision'
    assert report['summary']['all_seed_routing_sufficient']==all(d['routing_sufficient'] for d in decisions)
    assert report['summary']['all_seed_decoder_sufficient']==all(d['decoder_sufficient'] for d in decisions)
    compare(report['summary'],summarize(scores,[c['seed'] for c in lock['cases']]))
    if not lock['fixture']:
        expected_rows=[]
        for run in old_report['runs']:
            base=torch.load(ROOT/run['snapshots'][0]['checkpoint'],weights_only=True)['model']
            for snap in run['snapshots']:
                state=torch.load(ROOT/snap['checkpoint'],weights_only=True)['model']
                k=state['key.weight'];unit=k/k.square().sum(-1,keepdim=True).sqrt();gram=unit@unit.T
                actual=next(t for t in report['trajectories'] if (t['seed'],t['arm'],t['closures'])==(run['seed'],run['arm'],snap['step']))
                assert abs(actual['key_overlap_rms']-float(gram[~torch.eye(32,dtype=torch.bool)].square().mean().sqrt()))<1e-12
                expected_rows.append(dict(seed=run['seed'],arm=run['arm'],closures=snap['step'],checkpoint=snap['checkpoint'],**parameter_stats(state,base)))
        compare(report['trajectories'],expected_rows)
    assert report['evaluation_targets']==targets and report['hybrids']==len(lock['cases'])*4
    assert (report['optimizer_calls'],report['gradient_calls'],report['solver_calls'])==(0,0,0)
    result=dict(passed=True,cells=len(scores),hybrids=report['hybrids'],evaluation_targets=targets,max_logit_error=maximum,
                protected_files=len(lock['files']),report_sha256=replay.digest(out/'report.json'),independent_decisions=decisions)
    if persist:
        assert not (out/'audit.json').exists();replay.write(out/'audit.json',result)
    return result


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    if args.audit:print(json.dumps(audit(OUT)),flush=True)
    else:execute(OUT,*inputs())
