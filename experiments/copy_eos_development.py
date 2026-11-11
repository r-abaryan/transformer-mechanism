"""One bounded restored-state intervention on terminal training feedback."""
import hashlib
import json
import time
import uuid
import torch
from torch.nn import functional as F
from copy_learning_trajectory import ROOT,PARENT,MODELS,equal,summarize,audit_logits
from copy_position_sites import make_model,forward,inputs,sample,positions,evaluate_run,specs,digest,write
from audit_copy_position_sites import audit_cell
from copy_intact_development import trace,SNAP

OUT=ROOT/'results/copy_eos_development'


def objective(z,y,policy):
    if policy=='native':return F.cross_entropy(z.flatten(0,1),y.flatten())
    assert policy=='no_terminal'
    return F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='none').reshape_as(y)[:,:-1].sum()/y.numel()


def continuation(start,arm,data,policy,stop,dest):
    cp=torch.load(start,weights_only=True);model=make_model('rope');model.load_state_dict(cp['model'])
    opt=torch.optim.Adam(model.parameters(),lr=.001);opt.load_state_dict(cp['optimizer'])
    gen=torch.Generator();gen.set_state(cp['generator']);pg=torch.Generator();pg.set_state(cp['position_generator'])
    pools={name:torch.tensor(rows) for name,rows in data['pools'].items() if name.startswith('train_')}
    stream=hashlib.sha256();pstream=hashlib.sha256();examples=0;targets=0;tick=time.perf_counter()
    for step in range(cp['step']+1,stop+1):
        n,idx,_=sample(pools,gen,'mixed',step);pos=positions(n,pg);stream.update(json.dumps([n,idx.tolist()]).encode());pstream.update(json.dumps(pos.tolist()).encode())
        x=pools[f'train_{n}'][idx];p,y=inputs(x);opt.zero_grad(set_to_none=True)
        z=forward(model,x,p,pos if arm=='both' else None,pos)[0];loss=objective(z,y,policy);assert torch.isfinite(loss);loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters());opt.step();examples+=len(x);targets+=y.numel()
    assert not dest.exists();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=stop,generator=gen.get_state(),position_generator=pg.get_state()),dest)
    return dict(checkpoint=str(dest.relative_to(ROOT)),stream=stream.hexdigest(),position_stream=pstream.hexdigest(),updates=stop-cp['step'],examples=examples,computed_targets=targets,active_targets=targets if policy=='native' else targets-examples,seconds=time.perf_counter()-tick,optimizer_bytes=sum(v.numel()*v.element_size() for state in opt.state.values() for v in state.values() if torch.is_tensor(v)))


def evaluate_saved(path,data,arm,directory):
    m=make_model('rope').eval();m.load_state_dict(torch.load(path,weights_only=True)['model'])
    metrics,saved=evaluate_run(m,data,arm,True);maxerror=0.
    for name,spec in specs(data,True).items():
        pos=torch.tensor(spec['positions']) if spec['positions'] is not None else None
        error,_=audit_cell(m,spec['rows'],pos,arm,saved[name],metrics[name]);maxerror=max(maxerror,error)
    paired={};paired_saved={};rows=data['pools']['test_10']
    for n in (6,10):
        ev,error=trace(m,[r[:n] for r in rows]);maxerror=max(maxerror,error);met=summarize(ev['logits'],ev['labels'],rows,True);audit_logits(ev['logits'],ev['labels'],rows,met,True)
        for group,g in met.items():
            ids=[i for i,r in enumerate(rows) if group=='all' or (len(set(r[:6]))==6)==(group=='distinct')]
            if g is not None:g['head0_marker_mass']=float(ev['attention'][ids,0,-1].mean())
        paired[str(n)]=met;paired_saved[str(n)]=ev
    artifact=directory/(path.stem+'_scores.pt');torch.save(dict(ordinary=saved,paired=paired_saved),artifact)
    return dict(metrics=metrics,paired=paired,max_reference_error=maxerror,scores=str(artifact.relative_to(ROOT)),scores_sha256=digest(artifact))


def preflight(data):
    folder=ROOT/'results/test_runs'/('eos_window_'+uuid.uuid4().hex);folder.mkdir(parents=True);torch.manual_seed(93);m=make_model('rope');opt=torch.optim.Adam(m.parameters(),lr=.001)
    start=folder/'initialized.pt';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),step=0,generator=torch.Generator().manual_seed(1093).get_state(),position_generator=torch.Generator().manual_seed(2093).get_state()),start)
    evaluate_saved(start,data,'cross_only',folder)
    z=torch.randn(2,4,16,dtype=torch.float64,requires_grad=True);y=torch.tensor([[3,4,5,2],[5,4,3,2]])
    native=objective(z,y,'native');content=objective(z,y,'no_terminal');terminal=F.cross_entropy(z[:,-1],y[:,-1],reduction='sum')/y.numel()
    gn=torch.autograd.grad(native,z,retain_graph=True)[0];gc=torch.autograd.grad(content,z,retain_graph=True)[0];gt=torch.autograd.grad(terminal,z)[0]
    assert torch.allclose(gn,gc+gt,atol=1e-12,rtol=0) and torch.equal(gc[:,-1],torch.zeros_like(gc[:,-1]))
    assert torch.equal(gn[:,:-1],gc[:,:-1])
    assert not torch.allclose(content,content*y.numel()/y[:,:-1].numel())
    met=summarize(z.detach()[:,0],y[:,0],[[3]*10,[3,4,5,6,7,8,3,3,3,3]],True);bad=dict(met);bad.pop('distinct')
    try:audit_logits(z.detach()[:,0],y[:,0],[[3]*10,[3,4,5,6,7,8,3,3,3,3]],bad,True)
    except AssertionError:pass
    else:raise AssertionError('missing group accepted')
    profiles=[]
    for policy in ('native','no_terminal'):
        dest=folder/f'{policy}.pt';profiles.append(continuation(start,'cross_only',data,policy,1,dest));evaluate_saved(dest,data,'cross_only',folder)
    return dict(passed=True,initialized_saved_continuation_evaluator_audit=True,gradient_decomposition=True,unchanged_content_coefficients=True,wrong_denominator_rejected=True,missing_group_rejected=True,fixture_training_updates=2,complete_step_profiles=profiles)


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists();data=json.loads((PARENT/'data.json').read_text());write(OUT/'preflight.json',preflight(data))
    protected=json.loads((ROOT/'results/copy_head_pattern_transfer/plan.json').read_text())['protected']
    for name in ('experiments/copy_eos_development.py','docs/protocols/copy-eos-development.md','results/copy_eos_development/preflight.json','experiments/audit_copy_position_sites.py','results/copy_position_sites/profile.json'):
        protected[name]=digest(ROOT/name)
    assert all(digest(ROOT/p)==v for p,v in protected.items())
    write(OUT/'plan.json',dict(models=MODELS,policies=['native','no_terminal'],start=500,stop=1000,protected=protected))
    report=dict(status='running',runs=[],plan_sha256=digest(OUT/'plan.json'));write(OUT/'report.json',report)
    for seed,arm in MODELS:
        pair=[]
        for policy in ('native','no_terminal'):
            dest=OUT/f'{seed}_{arm}_{policy}.pt';run=continuation(SNAP/f'{seed}_{arm}_500.pt',arm,data,policy,1000,dest)
            if policy=='native':assert equal(torch.load(dest,weights_only=True),torch.load(SNAP/f'{seed}_{arm}_1000.pt',weights_only=True))
            run.update(seed=seed,arm=arm,policy=policy,checkpoint_sha256=digest(dest),evaluation=evaluate_saved(dest,data,arm,OUT));pair.append(run);report['runs'].append(run)
            write(OUT/'report.json',report);print(seed,arm,policy,run['evaluation']['paired']['10']['all'],flush=True)
        assert pair[0]['stream']==pair[1]['stream'] and pair[0]['position_stream']==pair[1]['position_stream']
    a,b=[x['evaluation'] for x in report['runs'][:2]]
    outcome=b['paired']['10']['all']['exact_accuracy']-a['paired']['10']['all']['exact_accuracy']>=.5 and (a['paired']['10']['all']['eos']-b['paired']['10']['all']['eos'])/128>=.5
    gap=lambda ev:abs(ev['paired']['6']['all']['head0_marker_mass']-ev['paired']['10']['all']['head0_marker_mass'])
    pattern=gap(b)<=.5*gap(a)
    content=all(a['metrics'][name][g]['content_accuracy']-b['metrics'][name][g]['content_accuracy']<=.05 for name in a['metrics'] if name.startswith(('train_','validation_','test_')) and int(name.split('_')[-1])<=6 for g in ('all','distinct','repeated') if a['metrics'][name][g] is not None)
    assert all(digest(ROOT/p)==v for p,v in protected.items())
    report.update(status='complete',weak_signatures=dict(answer_improvement=outcome,marker_gap_reduction=pattern,short_content_preservation=content,joint=outcome and pattern and content),audit=dict(passed=True,native_checkpoints_exact=True,paired_streams_exact=True,protected_files=len(protected),max_reference_error=max(x['evaluation']['max_reference_error'] for x in report['runs'])))
    write(OUT/'report.json',report)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()
