"""Matched positional-representation comparison, fixed budget and opened copy pools."""
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from encoder_decoder_contract import ROOT
from encoder_decoder_copy import inputs, logits, generate
from copy_length_exposure import sample, audit_data, metrics as ordinary_metrics
from copy_eos_training import oracle, measures as oracle_metrics, controller_checks
from copy_rope_model import make_model, reference
from mqa_training import hash_state
from position_distractors import digest
from recurrent_retrieval import write

DIRECTORY = ROOT/'results/copy_rope_training'
DATA = ROOT/'results/copy_length_exposure/data.json'
PROTOCOL = ROOT/'docs/protocols/copy-rope-training.md'
SEEDS = (3119, 3121, 3137)
ARMS = ('sinusoidal', 'rope')
STEPS = 6000


def evaluate(model, rows):
    zs, generated, oracles = [], [], []
    with torch.no_grad():
        for x in torch.tensor(rows).split(64):
            prefix, _ = inputs(x)
            zs.append(logits(model, x, prefix))
            generated.extend(generate(lambda a,b: logits(model,a,b)[:,-1], x, x.shape[1]+3))
            oracles.extend(oracle(lambda a,b: logits(model,a,b)[:,-1], x))
    z = torch.cat(zs)
    normal = ordinary_metrics(z, rows, generated)
    content = oracle_metrics(z, rows, oracles, 1.)
    return {g: dict(normal[g], **content[g]) for g in normal}, dict(logits=z, generated=generated, oracle_outputs=oracles)


def decision(runs):
    assert len(runs) == 6
    fit = {a: all(r['status']=='trained' and all(r['metrics'][f'train_{n}'][g]['exact_accuracy']>=.99
               for n in (3,4,5,6) for g in ('distinct','repeated')) for r in runs if r['arm']==a) for a in ARMS}
    seen = {a: fit[a] and all(r['metrics'][f'test_{n}'][g]['exact_accuracy']>=.95
               for r in runs if r['arm']==a for n in (3,4,5,6) for g in ('distinct','repeated')) for a in ARMS}
    pairs = []
    for seed in SEEDS:
        s, r = [next(x for x in runs if x['seed']==seed and x['arm']==a) for a in ARMS]
        gain = {str(n): r['metrics'][f'test_{n}']['all']['exact_accuracy']-s['metrics'][f'test_{n}']['all']['exact_accuracy'] for n in (8,10)}
        preserve = all(r['metrics'][f'test_{n}'][g]['exact_accuracy']-s['metrics'][f'test_{n}'][g]['exact_accuracy']>=-.05
                       for n in (3,4,5,6) for g in ('distinct','repeated'))
        pairs.append(dict(seed=seed, long_gains=gain, seen_length_preserved=preserve))
    transfer = {str(n): seen['rope'] and all(r['metrics'][f'test_{n}'][g]['exact_accuracy']>=.9
                for r in runs if r['arm']=='rope' for g in (('distinct','repeated') if n==8 else ('repeated',))) for n in (8,10)}
    return dict(training_fit=fit, seen_length_reliability=seen, paired=pairs,
                rope_benefit=all(fit.values()) and all(p['seen_length_preserved'] and min(p['long_gains'].values())>=.1 for p in pairs),
                rope_extrapolation=transfer)


def profile():
    assert not DIRECTORY.exists()
    cases = []
    for arm in ARMS:
        for n in (3,4,5,6):
            torch.manual_seed(SEEDS[0]);model=make_model(arm);opt=torch.optim.Adam(model.parameters(),lr=.001)
            x=torch.randint(3,11,(840//(n+1),n));prefix, labels=inputs(x);times=[]
            for _ in range(3):
                tick=time.perf_counter();opt.zero_grad(set_to_none=True)
                loss=F.cross_entropy(logits(model,x,prefix).flatten(0,1),labels.flatten());loss.backward();opt.step()
                times.append(time.perf_counter()-tick)
            cases.append(dict(arm=arm,length=n,batch=len(x),complete_step_seconds=times,
                optimizer_state_bytes=sum(v.numel()*v.element_size() for state in opt.state.values() for v in state.values() if torch.is_tensor(v))))
    DIRECTORY.mkdir()
    write(DIRECTORY/'profile.json',dict(device='cpu',dtype='float64',threads=1,torch=torch.__version__,gpu_used=False,
        peak_gpu_allocated_bytes=None,peak_gpu_reserved_bytes=None,profile_updates_discarded=True,cases=cases))
    print(json.dumps(cases),flush=True)


def main():
    assert PROTOCOL.exists() and (DIRECTORY/'preflight.json').exists() and not (DIRECTORY/'report.json').exists()
    payload=json.loads(DATA.read_text(encoding='utf-8'));checks=audit_data(payload)
    assert digest(DATA)==json.loads((DATA.parent/'report.json').read_text(encoding='utf-8'))['data_sha256']
    paths=[Path(__file__),ROOT/'experiments/audit_copy_rope_training.py',ROOT/'experiments/copy_rope_model.py',
        ROOT/'tests/test_copy_rope.py',PROTOCOL,DATA,DIRECTORY/'profile.json',DIRECTORY/'preflight.json',
        *[ROOT/'experiments'/p for p in ('encoder_decoder_contract.py','encoder_decoder_copy.py','copy_length_exposure.py',
          'copy_eos_training.py','audit_copy_length_exposure.py','audit_copy_eos_training.py','audit_kv_cache_equivalence.py',
          'mqa_training.py','position_distractors.py','recurrent_retrieval.py')]]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths};pools={p:torch.tensor(v) for p,v in payload['pools'].items()}
    report=dict(status='running',device='cpu',dtype='float64',torch=torch.__version__,parameters=1888,seeds=list(SEEDS),arms=list(ARMS),
        updates_per_run=STEPS,reused_open_data=True,source_dependencies=hashes,input_checks=checks,controller_checks=controller_checks(),runs=[])
    write(DIRECTORY/'report.json',report)
    for seed in SEEDS:
        torch.manual_seed(seed);initial=make_model('sinusoidal').state_dict();initial_hash=hash_state(initial)
        for arm in ARMS:
            model=make_model(arm);model.load_state_dict(copy.deepcopy(initial));opt=torch.optim.Adam(model.parameters(),lr=.001)
            gen=torch.Generator().manual_seed(seed+1000);stream=hashlib.sha256();ranks=hashlib.sha256()
            run=dict(seed=seed,arm=arm,status='running',initial_sha256=initial_hash,curve=[],metrics={},
                     sequences=0,supervised_targets=0,content_targets=0,attention_pairs=0)
            report['runs'].append(run);tick=time.perf_counter()
            for step in range(1,STEPS+1):
                n,idx,raw=sample(pools,gen,'mixed',step);ranks.update(raw)
                stream.update(json.dumps([n,idx.tolist()],separators=(',',':')).encode())
                x=pools[f'train_{n}'][idx];prefix,labels=inputs(x);opt.zero_grad(set_to_none=True)
                z=logits(model,x,prefix);loss=F.cross_entropy(z.flatten(0,1),labels.flatten())
                if step==1:
                    rz=reference(model,x,prefix,arm);rl=F.cross_entropy(rz.flatten(0,1),labels.flatten())
                    grads=torch.autograd.grad(rl,tuple(model.parameters()))
                    le=float((z-rz).detach().abs().max());assert le<=1e-9
                if not bool(torch.isfinite(loss)):
                    run.update(status='failed',failure_step=step);report['status']='failed';write(DIRECTORY/'report.json',report)
                    raise RuntimeError('Nonfinite objective; no restart')
                loss.backward()
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                if step==1:
                    ge=max(float((p.grad-g).abs().max()) for p,g in zip(model.parameters(),grads));assert ge<=1e-9
                    run['first_gradient_checks']=dict(logit_error=le,parameter_gradient_error=ge)
                opt.step();run['sequences']+=len(x);run['supervised_targets']+=labels.numel();run['content_targets']+=x.numel()
                run['attention_pairs']+=len(x)*(n*n+(n+1)*(n+2)//2+n*(n+1))
                if step%1000==0:
                    row=dict(step=step,minibatch_loss=float(loss.detach()))
                    if step in (1000,3000,6000):
                        row['pools']={p:evaluate(model,rows)[0] for p,rows in payload['pools'].items() if not p.startswith('test_')}
                    run['curve'].append(row);write(DIRECTORY/'report.json',report)
                    print(json.dumps(dict(seed=seed,arm=arm,step=step,loss=row['minibatch_loss'],elapsed_seconds=time.perf_counter()-tick)),flush=True)
            cp=DIRECTORY/f'{seed}_{arm}.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=STEPS,generator=gen.get_state()),cp)
            run.update(status='trained',step=STEPS,metrics=run['curve'][-1]['pools'],elapsed_seconds=time.perf_counter()-tick,
                sample_stream_sha256=stream.hexdigest(),rank_stream_sha256=ranks.hexdigest(),
                checkpoint=cp.relative_to(ROOT).as_posix(),checkpoint_sha256=digest(cp))
            write(DIRECTORY/'report.json',report)
        assert report['runs'][-2]['sample_stream_sha256']==report['runs'][-1]['sample_stream_sha256']
    report['status']='all_training_finished_before_test';write(DIRECTORY/'pretest.json',report)
    for run in report['runs']:
        model=make_model(run['arm']);model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model']);saved={}
        for pool,rows in payload['pools'].items():
            met,details=evaluate(model,rows)
            if pool in run['metrics']:assert met==run['metrics'][pool]
            run['metrics'][pool]=met;saved[pool]=details
        path=DIRECTORY/f"{run['seed']}_{run['arm']}_scores.pt";torch.save(saved,path)
        run.update(scores=path.relative_to(ROOT).as_posix(),scores_sha256=digest(path));write(DIRECTORY/'report.json',report)
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    report.update(status='complete',decision=decision(report['runs']),prior_hashes_unchanged=True)
    rows=[dict(seed=r['seed'],arm=r['arm'],pool=p,group=g,**m) for r in report['runs'] for p,gs in r['metrics'].items() for g,m in gs.items()]
    with (DIRECTORY/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report['metrics_sha256']=digest(DIRECTORY/'metrics.csv');write(DIRECTORY/'report.json',report)
    print(json.dumps(report['decision']),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if sys.argv[1:]==['profile']:profile()
    else:main()
