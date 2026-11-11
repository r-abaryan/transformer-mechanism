"""Paired terminal-EOS objective removal with identical content weighting."""
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from encoder_decoder_contract import ROOT, Model, independent
from encoder_decoder_copy import inputs, logits, evaluate as ordinary_evaluate
from copy_length_exposure import sample, groups, audit_data
from mqa_training import hash_state
from position_distractors import digest
from recurrent_retrieval import write

DIRECTORY=ROOT/'results/copy_eos_training'
PROTOCOL=ROOT/'docs/protocols/copy-eos-training.md'
DATA=ROOT/'results/copy_length_exposure/data.json'
SEEDS=(3083,3089,3109)
ARMS={'eos':1.,'content_only':0.}
STEPS=6000


def objective(z,source,alpha):
    _,labels=inputs(source)
    losses=F.cross_entropy(z.flatten(0,1),labels.flatten(),reduction='none').reshape_as(labels)
    return (losses[:,:-1].sum()+alpha*losses[:,-1].sum())/labels.numel()


def oracle(next_logits,source):
    prefix=torch.ones(len(source),1,dtype=torch.long);outputs=[]
    for _ in range(source.shape[1]):
        z=next_logits(source,prefix).clone();z[:,2]=-torch.inf
        chosen=z.argmax(-1);outputs.append(chosen)
        prefix=torch.cat((prefix,chosen[:,None]),1)
    return torch.stack(outputs,1).tolist()


def controller_checks():
    x=torch.tensor([[3,4,5],[5,4,3]])
    def scripted(source,prefix):
        j=prefix.shape[1]-1;z=torch.full((len(source),16),-100.,dtype=torch.float64)
        z[:,2]=100.;z.scatter_(1,source[:,j:j+1],50.)
        # A wrong prefix makes the next symbol wrong, checking the feedback path.
        if j and not torch.equal(prefix[:,1:],source[:,:j]):z[:,11]=75.
        return z
    actual=oracle(scripted,x);assert actual==x.tolist()
    assert scripted(x,torch.ones(2,1,dtype=torch.long)).argmax(-1).tolist()==[2,2]
    wrong=scripted(x,torch.tensor([[1,9],[1,9]])).clone();wrong[:,2]=-torch.inf
    assert wrong.argmax(-1).tolist()==[11,11]
    # Continuing for n-1 positions is an intentionally incorrect boundary.
    assert [row[:-1] for row in actual]!=x.tolist()
    return dict(passed=True,eos_selection_error_detected=True,wrong_prefix_detected=True,off_by_one_detected=True)


def objective_controls():
    x=torch.tensor([[3,4]])
    z=torch.arange(48,dtype=torch.float64).reshape(1,3,16).div(10).requires_grad_()
    a=objective(z,x,0);b=objective(z,x,1)
    ga=torch.autograd.grad(a,z,retain_graph=True)[0];gb=torch.autograd.grad(b,z,retain_graph=True)[0]
    assert torch.equal(ga[:,:2],gb[:,:2]) and bool((ga[:,-1]==0).all()) and float(gb[:,-1].abs().max())>0
    wrong=F.cross_entropy(z[:,:2].flatten(0,1),x.flatten())
    gw=torch.autograd.grad(wrong,z)[0]
    assert float((gw-ga).abs().max())>1e-3
    return dict(passed=True,unremoved_eos_target_detected=True,content_rescaling_error_detected=True,
                content_row_gradients_identical=True,content_only_terminal_gradient_zero=True)


def measures(z,rows,outputs,alpha):
    source=torch.tensor(rows);_,labels=inputs(source);n=source.shape[1]
    losses=F.cross_entropy(z.flatten(0,1),labels.flatten(),reduction='none').reshape_as(labels)
    restricted=z[:,:n].clone();restricted[:,:,2]=-torch.inf
    generated=torch.tensor(outputs);out={}
    for group,mask in groups(rows).items():
        correct=z[mask,:n].argmax(-1)==source[mask]
        conditional=restricted[mask].argmax(-1)==source[mask]
        prediction=generated[mask]==source[mask]
        out[group]=dict(sequences=int(mask.sum()),content_targets=int(mask.sum())*n,
            oracle_content_exact=float(prediction.all(1).double().mean()),oracle_token_accuracy=float(prediction.double().mean()),
            teacher_content_accuracy=float(correct.double().mean()),teacher_no_eos_accuracy=float(conditional.double().mean()),
            teacher_no_eos_exact=float(conditional.all(1).double().mean()),
            content_loss=float(losses[mask,:n].mean()),terminal_eos_loss=float(losses[mask,-1].mean()),
            weighted_objective=float((losses[mask,:n].sum()+alpha*losses[mask,-1].sum())/(int(mask.sum())*(n+1))),
            teacher_content_eos_rate=float((z[mask,:n].argmax(-1)==2).double().mean()),
            terminal_eos_accuracy=float((z[mask,-1].argmax(-1)==2).double().mean()))
    return out


def evaluate(model,rows,alpha):
    zs=[];outputs=[]
    with torch.no_grad():
        for x in torch.tensor(rows).split(64):
            shifted,_=inputs(x);zs.append(logits(model,x,shifted))
            outputs.extend(oracle(lambda a,b:logits(model,a,b)[:,-1],x))
    z=torch.cat(zs)
    return measures(z,rows,outputs,alpha),dict(logits=z,oracle_outputs=outputs)


def decision(runs):
    assert len(runs)==6
    fit={arm:all(r['status']=='trained' and all(r['metrics'][f'train_{n}'][g]['oracle_content_exact']>=.99
        and r['metrics'][f'train_{n}'][g]['teacher_content_accuracy']>=.99
        for n in (3,4,5,6) for g in ('distinct','repeated')) for r in runs if r['arm']==arm) for arm in ARMS}
    held={arm:fit[arm] and all(r['metrics'][f'test_{n}'][g]['oracle_content_exact']>=.95
        for r in runs if r['arm']==arm for n in (3,4,5,6) for g in ('distinct','repeated')) for arm in ARMS}
    pairs=[]
    for seed in SEEDS:
        a,b=[next(r for r in runs if r['seed']==seed and r['arm']==arm) for arm in ARMS]
        gain={str(n):b['metrics'][f'test_{n}']['all']['oracle_content_exact']-a['metrics'][f'test_{n}']['all']['oracle_content_exact'] for n in (8,10)}
        preserve=all(b['metrics'][f'test_{n}'][g]['oracle_content_exact']-a['metrics'][f'test_{n}'][g]['oracle_content_exact']>=-.05
            for n in (3,4,5,6) for g in ('distinct','repeated'))
        pairs.append(dict(seed=seed,long_gains=gain,seen_length_preserved=preserve))
    long={str(n):held['content_only'] and all(r['metrics'][f'test_{n}'][g]['oracle_content_exact']>=.9
        for r in runs if r['arm']=='content_only' for g in (('distinct','repeated') if n==8 else ('repeated',))) for n in (8,10)}
    return dict(training_fit=fit,seen_length_reliability=held,paired=pairs,
        removing_eos_benefit=all(fit.values()) and all(p['seen_length_preserved'] and all(v>=.1 for v in p['long_gains'].values()) for p in pairs),
        content_only_oracle_transfer=long)


def profile():
    assert not DIRECTORY.exists();cases=[]
    for arm,alpha in ARMS.items():
        torch.manual_seed(SEEDS[0]);model=Model();opt=torch.optim.Adam(model.parameters(),lr=.001)
        x=torch.randint(3,11,(120,6));prefix,_=inputs(x);times=[]
        for _ in range(3):
            tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            loss=objective(logits(model,x,prefix),x,alpha);loss.backward();opt.step();times.append(time.perf_counter()-tick)
        cases.append(dict(arm=arm,length=6,batch=120,complete_step_seconds=times,
            optimizer_state_bytes=sum(v.numel()*v.element_size() for state in opt.state.values() for v in state.values() if torch.is_tensor(v))))
    DIRECTORY.mkdir();write(DIRECTORY/'profile.json',dict(device='cpu',dtype='float64',torch=torch.__version__,
        threads=1,gpu_used=False,peak_gpu_allocated_bytes=None,peak_gpu_reserved_bytes=None,cases=cases,
        other_lengths_profile='results/copy_length_exposure/profile.json',profile_updates_discarded=True))
    print(json.dumps(cases),flush=True)


def main():
    assert PROTOCOL.exists() and (DIRECTORY/'profile.json').exists() and not (DIRECTORY/'report.json').exists()
    payload=json.loads(DATA.read_text(encoding='utf-8'));parent=json.loads((DATA.parent/'report.json').read_text(encoding='utf-8'))
    assert digest(DATA)==parent['data_sha256'];checks=audit_data(payload)
    paths=[Path(__file__),ROOT/'experiments/audit_copy_eos_training.py',PROTOCOL,DATA,DATA.parent/'report.json',
        DIRECTORY/'profile.json',DATA.parent/'profile.json',ROOT/'results/encoder_decoder_copy_data.json',
        *[ROOT/'experiments'/p for p in ('copy_length_exposure.py','encoder_decoder_copy.py','encoder_decoder_contract.py',
            'audit_copy_length_exposure.py','audit_encoder_decoder_copy.py','audit_kv_cache_equivalence.py','position_distractors.py','mqa_training.py','recurrent_retrieval.py')]]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths};pools={k:torch.tensor(v) for k,v in payload['pools'].items()}
    report=dict(status='running',device='cpu',dtype='float64',torch=torch.__version__,parameters=1888,seeds=list(SEEDS),
        arms=ARMS,updates_per_run=STEPS,reused_open_data=True,data_path=DATA.relative_to(ROOT).as_posix(),
        source_dependencies=hashes,input_checks=checks,controller_checks=controller_checks(),objective_controls=objective_controls(),runs=[])
    write(DIRECTORY/'report.json',report)
    for seed in SEEDS:
        torch.manual_seed(seed);initial=Model().state_dict();initial_hash=hash_state(initial)
        for arm,alpha in ARMS.items():
            model=Model();model.load_state_dict(copy.deepcopy(initial));opt=torch.optim.Adam(model.parameters(),lr=.001)
            gen=torch.Generator().manual_seed(seed+1000);stream=hashlib.sha256();ranks=hashlib.sha256()
            run=dict(seed=seed,arm=arm,status='running',initial_sha256=initial_hash,curve=[],metrics={},
                sequences=0,content_targets=0,terminal_eos_targets=0,computed_target_rows=0,attention_pairs=0)
            report['runs'].append(run);tick=time.perf_counter()
            for step in range(1,STEPS+1):
                n,idx,raw=sample(pools,gen,'mixed',step);ranks.update(raw)
                stream.update(json.dumps([n,idx.tolist()],separators=(',',':')).encode())
                x=pools[f'train_{n}'][idx];prefix,labels=inputs(x);opt.zero_grad(set_to_none=True)
                z=logits(model,x,prefix);loss=objective(z,x,alpha)
                if step==1:
                    s,t=model.features(x,prefix);rz=independent(model,s,t,torch.zeros_like(x,dtype=torch.bool),torch.zeros_like(prefix,dtype=torch.bool))
                    rl=objective(rz,x,alpha);grads=torch.autograd.grad(rl,tuple(model.parameters()))
                    le=float((z-rz).detach().abs().max());assert le<=1e-9
                    no_eos=objective(z,x,0);with_eos=objective(z,x,1)
                    terminal=F.cross_entropy(z[:,-1],torch.full((len(x),),2,dtype=torch.long))/(n+1)
                    assert float((with_eos-no_eos-terminal).detach().abs())<1e-12
                    gradient_gap=torch.autograd.grad(with_eos-no_eos,tuple(model.parameters()),retain_graph=True)
                    terminal_grad=torch.autograd.grad(terminal,tuple(model.parameters()),retain_graph=True)
                    gap=max(float((a-b).abs().max()) for a,b in zip(gradient_gap,terminal_grad));assert gap<=1e-9
                if not bool(torch.isfinite(loss)):
                    run.update(status='failed',failure_step=step);report['status']='failed';write(DIRECTORY/'report.json',report)
                    raise RuntimeError('Nonfinite objective; declared comparison stops.')
                loss.backward()
                if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    run.update(status='failed',failure_step=step);report['status']='failed';write(DIRECTORY/'report.json',report)
                    raise RuntimeError('Nonfinite gradient; declared comparison stops.')
                if step==1:
                    ge=max(float((p.grad-g).abs().max()) for p,g in zip(model.parameters(),grads));assert ge<=1e-9
                    run['first_gradient_checks']=dict(logit_error=le,independent_gradient_error=ge,terminal_gradient_decomposition_error=gap)
                opt.step();run['sequences']+=len(x);run['content_targets']+=x.numel()
                run['terminal_eos_targets']+=int(alpha)*len(x);run['computed_target_rows']+=labels.numel()
                run['attention_pairs']+=len(x)*(n*n+(n+1)*(n+2)//2+(n+1)*n)
                if step%1000==0:
                    row=dict(step=step,minibatch_objective=float(loss.detach()))
                    if step in (1000,3000,6000):
                        row['pools']={p:evaluate(model,rows,alpha)[0] for p,rows in payload['pools'].items() if not p.startswith('test_')}
                    run['curve'].append(row);write(DIRECTORY/'report.json',report)
                    print(json.dumps(dict(seed=seed,arm=arm,step=step,objective=row['minibatch_objective'],elapsed_seconds=time.perf_counter()-tick)),flush=True)
            cp=DIRECTORY/f'{seed}_{arm}.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=STEPS,generator=gen.get_state()),cp)
            run.update(status='trained',step=STEPS,metrics=run['curve'][-1]['pools'],elapsed_seconds=time.perf_counter()-tick,
                sample_stream_sha256=stream.hexdigest(),rank_stream_sha256=ranks.hexdigest(),
                checkpoint=cp.relative_to(ROOT).as_posix(),checkpoint_sha256=digest(cp))
            write(DIRECTORY/'report.json',report)
        a,b=report['runs'][-2:]
        assert a['sample_stream_sha256']==b['sample_stream_sha256'] and a['rank_stream_sha256']==b['rank_stream_sha256']
    report['status']='all_training_finished_before_test';write(DIRECTORY/'pretest.json',report)
    for run in report['runs']:
        model=Model();model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model']);saved={}
        for pool,rows in payload['pools'].items():
            met,details=evaluate(model,rows,ARMS[run['arm']])
            if pool in run['metrics']:assert met==run['metrics'][pool]
            run['metrics'][pool]=met;saved[pool]=details
        # Ordinary stopping is a separate descriptive control, not the shared comparison metric.
        run['ordinary_stopping']={p:ordinary_evaluate(model,torch.tensor(payload['pools'][p]))[0]
                                 for p in ('train_6','test_6','test_8','test_10')}
        scores=DIRECTORY/f"{run['seed']}_{run['arm']}_scores.pt";torch.save(saved,scores)
        run.update(scores=scores.relative_to(ROOT).as_posix(),scores_sha256=digest(scores));write(DIRECTORY/'report.json',report)
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    report.update(status='complete',decision=decision(report['runs']),prior_hashes_unchanged=True)
    with (DIRECTORY/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        rows=[dict(seed=r['seed'],arm=r['arm'],pool=p,group=g,**m) for r in report['runs'] for p,gs in r['metrics'].items() for g,m in gs.items()]
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report['metrics_sha256']=digest(DIRECTORY/'metrics.csv');write(DIRECTORY/'report.json',report);print(json.dumps(report['decision']),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if sys.argv[1:]==['profile']:profile()
    else:main()
