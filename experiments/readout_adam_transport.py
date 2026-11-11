"""Replay 16 original-budget updates and decompose their classifier movement."""
import argparse
import copy
import csv
import importlib.util
import json
import unittest

import torch
from torch.nn import functional as F
import readout_scale_pressure as pressure

training=pressure.training
continuation=pressure.scale.previous.parent
ROOT=pressure.ROOT
OUT=ROOT/'results/readout_adam_transport'
SEEDS=(3461,3463)
STARTS=(3000,11000)
SOURCES=('experiments/readout_adam_transport.py','experiments/audit_readout_adam_transport.py',
         'tests/test_readout_adam_transport.py')
PROTOCOL='docs/protocols/readout-adam-transport.md'


def center(state):
    a=torch.cat((state['output.weight'],state['output.bias'][:,None]),1)
    return a-a.mean(0,keepdim=True)


def transport(before,after,before_opt,after_opt,gradients,names):
    c=center(before);d=center(after)-c;norm2=float(c.square().sum());assert norm2>0
    head_grad=torch.cat((gradients['output.weight'],gradients['output.bias'][:,None]),1)
    raw=float((head_grad*c).sum());group=before_opt['param_groups'][0];lr=group['lr'];b1,b2=group['betas'];eps=group['eps']
    prior_parts={};current_parts={}
    for name,pid in zip(names,group['params']):
        if not name.startswith('output.'):continue
        previous=before_opt['state'].get(pid,{});updated=after_opt['state'][pid]
        old_m=previous.get('exp_avg',torch.zeros_like(gradients[name]))
        t=int(updated['step']);den=(updated['exp_avg_sq']/(1-b2**t)).sqrt()+eps
        prior_parts[name]=-lr*b1*old_m/((1-b1**t)*den)
        current_parts[name]=-lr*(1-b1)*gradients[name]/((1-b1**t)*den)
    def project(parts):
        a=torch.cat((parts['output.weight'],parts['output.bias'][:,None]),1)
        return float((a*c).sum())/norm2
    rho=float((d*c).sum())/norm2;old_norm=norm2**.5;new_norm=float(center(after).norm())
    stored=project(prior_parts);current=project(current_parts)
    assert abs(rho-stored-current)<=1e-11,'Adam projection terms do not reconcile'
    identity=new_norm**2-norm2-(2*float((c*d).sum())+float(d.square().sum()))
    assert abs(identity)<=1e-9,'Head norm identity failed'
    return dict(raw_g=raw,sgd_rho=-lr*raw/norm2,adam_rho=rho,stored_rho=stored,current_rho=current,
        before_norm=old_norm,after_norm=new_norm,relative_norm_change=new_norm/old_norm-1,
        perpendicular_ratio=float((d-rho*c).norm())/old_norm,norm_identity_error=identity)


def beta_record(trace):
    before,after=trace['before_model'],trace['after_model'];pid=trace['before_optimizer']['param_groups'][0]['params'][0]
    state=trace['after_optimizer']['state'][pid]
    return dict(step=trace['step'],load=trace['load'],alpha_before=float(before['write_logit']),alpha_after=float(after['write_logit']),
        beta_before=float(before['write_logit'].sigmoid()),beta_after=float(after['write_logit'].sigmoid()),
        gradient=float(trace['gradients']['write_logit']),exp_avg=float(state['exp_avg']),exp_avg_sq=float(state['exp_avg_sq']))


def replay_cycle(path,pools,seed,start):
    saved=torch.load(path,weights_only=True);assert saved['step']==start
    with torch.random.fork_rng():
        model,opt,restored=continuation.restore(path);assert restored==start
        hr=torch.Generator();hr.set_state(saved['history_rng']);qr=torch.Generator();qr.set_state(saved['query_rng'])
        traces=[]
        for step in range(start+1,start+5):
            load=(2,8,16,32)[(step-1)%4];rows=pools[f'train_{load}']
            events,_,_=training.parent.study.parent.tensors(rows);queries,labels=training.parent.query_tables(rows)
            history_before=hr.get_state().clone();query_before=qr.get_state().clone()
            indices=torch.randint(len(rows),(64,),generator=hr);draws=torch.randint(load,(64,load),generator=qr)
            q,y,_=training.choose(queries,labels,indices,draws,'exhaustive')
            chosen=[rows[int(i)] for i in indices];_,_,_,repeat=training.replay.pressure.prepare(chosen,'all_queries')
            before=copy.deepcopy(model.state_dict());before_opt=copy.deepcopy(opt.state_dict())
            opt.zero_grad(set_to_none=True);z=training.forward_queries(model,events[indices],q)
            loss=F.cross_entropy(z.reshape(-1,6),y.flatten());loss.backward()
            gradients={n:p.grad.detach().clone() for n,p in model.named_parameters()}
            opt.step();after=copy.deepcopy(model.state_dict());after_opt=copy.deepcopy(opt.state_dict())
            trace=dict(seed=seed,start=start,step=step,load=load,indices=indices,draws=draws,queries=q,labels=y,repeat=repeat,
                logits=z.detach(),before_model=before,after_model=after,before_optimizer=before_opt,after_optimizer=after_opt,
                gradients=gradients,names=list(dict(model.named_parameters())),history_before=history_before,query_before=query_before,
                history_after=hr.get_state().clone(),query_after=qr.get_state().clone())
            trace['record']=dict(seed=seed,start=start,step=step,load=load,loss=float(loss.detach()),
                groups=pressure.summarize(dict(logits=z.detach(),labels=y,repeat=repeat)),
                **transport(before,after,before_opt,after_opt,gradients,trace['names']))
            assert abs(trace['record']['raw_g']-trace['record']['groups']['all']['mean_g'])<=1e-9
            traces.append(trace)
    assert training.replay.equal(torch.load(path,weights_only=True),saved),'Source checkpoint changed'
    return traces


def inputs():
    report=json.loads((continuation.OUT/'report.json').read_text())
    pools=json.loads((training.parent.OUT/'development_data.json').read_text())['pools'];cycles=[]
    for seed in SEEDS:
        run=next(r for r in report['runs'] if r['seed']==seed)
        for start in STARTS:
            snap=next(r for r in run['snapshots'] if r['step']==start)
            path=ROOT/snap['checkpoint'];assert training.replay.digest(path)==snap['sha256']
            original=[next(r for r in run['steps'] if r['step']==t) for t in range(start+1,start+5)]
            cycles.append((seed,start,path,original))
    return pools,cycles


def preflight():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    spec=importlib.util.spec_from_file_location('test_readout_adam_transport',ROOT/SOURCES[-1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    record=dict(passed=result.wasSuccessful(),methods=result.testsRun,sources={n:training.replay.digest(ROOT/n) for n in SOURCES})
    path=ROOT/'results'/('readout_adam_transport_preflight.json' if record['passed'] else 'readout_adam_transport_preflight_failed.json')
    assert not path.exists();training.replay.write(path,record);assert record['passed']


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert not OUT.exists(),'Preserve evidence'
    ready_path=ROOT/'results/readout_adam_transport_preflight.json';ready=json.loads(ready_path.read_text())
    assert ready['passed'] and ready['sources']=={n:training.replay.digest(ROOT/n) for n in SOURCES}
    files=dict(json.loads((pressure.OUT/'lock.json').read_text())['files'])
    previous=json.loads((pressure.OUT/'audit.json').read_text())
    assert previous['passed'] and previous['report_sha256']==training.replay.digest(pressure.OUT/'report.json')
    for name in ('report.json','audit.json','metrics.csv','lock.json'):
        p=pressure.OUT/name;files[p.relative_to(ROOT).as_posix()]=training.replay.digest(p)
    for name in (*SOURCES,PROTOCOL,ready_path.relative_to(ROOT).as_posix()):files[name]=training.replay.digest(ROOT/name)
    training.replay.check_files(files);OUT.mkdir()
    training.replay.write(OUT/'lock.json',dict(files=files,seeds=list(SEEDS),starts=list(STARTS),cycle_updates=4,
        raw_sign_resolution=1e-10,rho_sign_resolution=1e-12))
    pools,cycles=inputs();traces=[];bindings=[]
    for seed,start,path,original in cycles:
        cycle=replay_cycle(path,pools,seed,start)
        assert [beta_record(t) for t in cycle]==original,'Historical beta log mismatch'
        traces.extend(cycle);bindings.append(dict(seed=seed,start=start,checkpoint=path.relative_to(ROOT).as_posix()))
    assert len(traces)==16 and sum(t['labels'].numel() for t in traces)==2*2*64*sum((2,8,16,32))==14848
    torch.save(traces,OUT/'traces.pt')
    report=dict(status='complete',new_training_budget_updates=0,replayed_optimizer_calls=16,histories=1024,input_events=32768,
        answer_targets=14848,device='cpu',dtype='float64',cycles=bindings,cells=[t['record'] for t in traces],
        traces_sha256=training.replay.digest(OUT/'traces.pt'),lock_sha256=training.replay.digest(OUT/'lock.json'))
    training.replay.check_files(files);training.replay.write(OUT/'report.json',report)
    with (OUT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        fields=[k for k in report['cells'][0] if k!='groups'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for cell in report['cells']:writer.writerow({k:cell[k] for k in fields})
    print(json.dumps(dict(replayed_optimizer_calls=16,new_training_budget_updates=0,answer_targets=14848,
        cells=[{k:c[k] for k in ('seed','step','load','raw_g','sgd_rho','adam_rho','stored_rho','current_rho','relative_norm_change')} for c in report['cells']])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    if parser.parse_args().preflight:preflight()
    else:run()
