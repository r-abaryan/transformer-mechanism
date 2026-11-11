"""Keep terminal CE but route its new gradient only to the final classifier."""
import hashlib
import json
import time
import uuid
import torch
from torch.nn import functional as F
from copy_eos_development import evaluate_saved,preflight as prior_preflight
from copy_learning_trajectory import ROOT,PARENT,MODELS,equal
from copy_position_sites import make_model,forward,inputs,sample,positions,digest,write
from copy_intact_development import SNAP

OUT=ROOT/'results/copy_terminal_gradient_route'


def detach_terminal(module,args):
    h=args[0];return (torch.cat((h[:,:-1],h[:,-1:].detach()),dim=1),)


def gradient_check(model,detach=True):
    x=torch.tensor([[3,4,5],[5,4,3]]);prefix,y=inputs(x);params=dict(model.named_parameters());grads={};zs={}
    for mode in ('native','content','routed'):
        hook=model.base.output.register_forward_pre_hook(detach_terminal) if mode=='routed' and detach else None
        try:z=forward(model,x,prefix)[0]
        finally:
            if hook:hook.remove()
        zs[mode]=z.detach()
        loss=F.cross_entropy(z.flatten(0,1),y.flatten()) if mode!='content' else F.cross_entropy(z[:,:-1].flatten(0,1),y[:,:-1].flatten(),reduction='sum')/y.numel()
        grads[mode]=dict(zip(params,torch.autograd.grad(loss,tuple(params.values()))))
    assert torch.equal(zs['native'],zs['routed'])
    assert 'base.output.weight' in params and model.base.output.bias is None
    differences=[]
    for name in params:
        expected=grads['native'][name] if name=='base.output.weight' else grads['content'][name]
        assert torch.allclose(grads['routed'][name],expected,atol=1e-11,rtol=1e-11),name
        if name!='base.output.weight':differences.append(float((grads['native'][name]-grads['content'][name]).abs().max()))
    assert max(differences)>1e-6


def train(start,arm,data,policy,stop,dest):
    cp=torch.load(start,weights_only=True);m=make_model('rope');m.load_state_dict(cp['model']);opt=torch.optim.Adam(m.parameters(),lr=.001);opt.load_state_dict(cp['optimizer'])
    gen=torch.Generator();gen.set_state(cp['generator']);pg=torch.Generator();pg.set_state(cp['position_generator'])
    pools={k:torch.tensor(v) for k,v in data['pools'].items() if k.startswith('train_')};stream=hashlib.sha256();pstream=hashlib.sha256();examples=0;targets=0;tick=time.perf_counter()
    assert policy in ('native','classifier_only')
    hook=m.base.output.register_forward_pre_hook(detach_terminal) if policy=='classifier_only' else None
    try:
        for step in range(cp['step']+1,stop+1):
            n,idx,_=sample(pools,gen,'mixed',step);pos=positions(n,pg);stream.update(json.dumps([n,idx.tolist()]).encode());pstream.update(json.dumps(pos.tolist()).encode())
            x=pools[f'train_{n}'][idx];p,y=inputs(x);opt.zero_grad(set_to_none=True)
            z=forward(m,x,p,pos if arm=='both' else None,pos)[0];loss=F.cross_entropy(z.flatten(0,1),y.flatten());assert torch.isfinite(loss);loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters());opt.step();examples+=len(x);targets+=y.numel()
    finally:
        if hook:hook.remove()
    assert not dest.exists();torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),step=stop,generator=gen.get_state(),position_generator=pg.get_state()),dest)
    return dict(checkpoint=str(dest.relative_to(ROOT)),checkpoint_sha256=digest(dest),stream=stream.hexdigest(),position_stream=pstream.hexdigest(),updates=stop-cp['step'],examples=examples,targets=targets,seconds=time.perf_counter()-tick,optimizer_bytes=sum(v.numel()*v.element_size() for state in opt.state.values() for v in state.values() if torch.is_tensor(v)))


def preflight(data):
    prior=prior_preflight(data)
    folder=ROOT/'results/test_runs'/('terminal_route_'+uuid.uuid4().hex);folder.mkdir(parents=True);torch.manual_seed(107);m=make_model('rope');gradient_check(m)
    try:gradient_check(m,False)
    except AssertionError:pass
    else:raise AssertionError('Missing detach was accepted')
    opt=torch.optim.Adam(m.parameters(),lr=.001);start=folder/'initialized.pt';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),step=0,generator=torch.Generator().manual_seed(1107).get_state(),position_generator=torch.Generator().manual_seed(2107).get_state()),start)
    profiles=[]
    for policy in ('native','classifier_only'):
        dest=folder/f'{policy}.pt';profiles.append(train(start,'cross_only',data,policy,1,dest));evaluate_saved(dest,data,'cross_only',folder)
    return dict(passed=True,prior_evaluator_checks=prior,parameter_gradient_scope=True,missing_detach_rejected=True,unchanged_logits=True,saved_routed_evaluator=True,complete_step_profiles=profiles,total_fixture_updates=4)


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists();data=json.loads((PARENT/'data.json').read_text());write(OUT/'preflight.json',preflight(data))
    protected=json.loads((ROOT/'results/copy_eos_development/plan.json').read_text())['protected']
    for p in ('experiments/copy_terminal_gradient_route.py','docs/protocols/copy-terminal-gradient-route.md','results/copy_terminal_gradient_route/preflight.json'):
        protected[p]=digest(ROOT/p)
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    write(OUT/'plan.json',dict(models=MODELS,policies=['native','classifier_only'],start=500,stop=1000,protected=protected))
    report=dict(status='running',runs=[],plan_sha256=digest(OUT/'plan.json'));write(OUT/'report.json',report)
    for seed,arm in MODELS:
        pair=[]
        for policy in ('native','classifier_only'):
            dest=OUT/f'{seed}_{arm}_{policy}.pt';run=train(SNAP/f'{seed}_{arm}_500.pt',arm,data,policy,1000,dest)
            if policy=='native':assert equal(torch.load(dest,weights_only=True),torch.load(SNAP/f'{seed}_{arm}_1000.pt',weights_only=True))
            run.update(seed=seed,arm=arm,policy=policy,evaluation=evaluate_saved(dest,data,arm,OUT));pair.append(run);report['runs'].append(run);write(OUT/'report.json',report)
            print(seed,arm,policy,run['evaluation']['metrics']['test_10']['all'],flush=True)
        assert pair[0]['stream']==pair[1]['stream'] and pair[0]['position_stream']==pair[1]['position_stream']
    weak=report['runs'][1]['evaluation']['metrics']['test_10']['all'];primary=weak['exact_accuracy']>=.95 and weak['on_time_eos']>=.95;checks=[]
    for i in range(0,len(report['runs']),2):
        a,b=[r['evaluation']['metrics'] for r in report['runs'][i:i+2]]
        for name in a:
            eligible=(name.startswith(('train_','validation_','test_')) and int(name.split('_')[-1])<=6) or (i>0 and name in ('test_8','test_10'))
            if eligible:
                for g,v in a[name].items():
                    if v is not None:checks.append(dict(seed=report['runs'][i]['seed'],arm=report['runs'][i]['arm'],pool=name,group=g,change=b[name][g]['exact_accuracy']-v['exact_accuracy']))
    preservation=all(c['change']>=-.05 for c in checks)
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    report.update(status='complete',criteria=dict(primary=primary,preservation=preservation,supported=primary and preservation),preservation_checks=checks,audit=dict(passed=True,native_checkpoints_exact=True,streams_exact=True,protected_files=len(protected),max_reference_error=max(r['evaluation']['max_reference_error'] for r in report['runs'])))
    write(OUT/'report.json',report)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()
