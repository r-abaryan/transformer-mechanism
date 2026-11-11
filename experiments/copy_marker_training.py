"""One fixed learning diagnostic: both positional arms train with source EOS."""
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
from encoder_decoder_copy import inputs, logits
from copy_length_exposure import sample, audit_data
from copy_rope_training import evaluate
from copy_eos_training import controller_checks
from copy_marker_model import make_model, reference
from mqa_training import hash_state
from position_distractors import digest
from recurrent_retrieval import write

DIRECTORY=ROOT/'results/copy_marker_training'
DATA=ROOT/'results/copy_length_exposure/data.json'
PROTOCOL=ROOT/'docs/protocols/copy-marker-training.md'
SEEDS=(3163,3167,3169)
ARMS=('sinusoidal','rope')
STEPS=6000
LENGTHS=(3,4,5,6,8,10)


def pools_and_checks(payload):
    checks=audit_data(payload);pools=dict(payload['pools']);overlap={}
    for n in LENGTHS:
        rows=[[symbol]*n for symbol in range(3,11)]
        pools[f'uniform_{n}']=rows
        overlap[str(n)]={split:sum(row in payload['pools'].get(f'{split}_{n}',[]) for row in rows)
                        for split in ('train','validation','test')}
    return pools,dict(original_pools=checks,uniform_overlap=overlap,
        uniform_cases=sum(len(pools[f'uniform_{n}']) for n in LENGTHS),
        evaluation_pool_count=len(pools),uniform_added_to_training=False)


def decision(runs):
    assert len(runs)==len(SEEDS)*len(ARMS)
    fit={a:all(r['status']=='trained' and all(r['metrics'][f'train_{n}'][g]['exact_accuracy']>=.99
         for n in range(3,7) for g in ('distinct','repeated')) for r in runs if r['arm']==a) for a in ARMS}
    seen={a:fit[a] and all(r['metrics'][f'test_{n}'][g]['exact_accuracy']>=.95
          for r in runs if r['arm']==a for n in range(3,7) for g in ('distinct','repeated')) for a in ARMS}
    uniform_short={a:all(r['metrics'][f'uniform_{n}']['all']['exact_accuracy']>=.9
          for r in runs if r['arm']==a for n in range(3,7)) for a in ARMS}
    transfer={a:{str(n):seen[a] and all(r['metrics'][f'test_{n}'][g]['exact_accuracy']>=.9
          for r in runs if r['arm']==a for g in (('distinct','repeated') if n==8 else ('repeated',)))
          for n in (8,10)} for a in ARMS}
    pairs=[]
    for seed in SEEDS:
        s,r=[next(x for x in runs if x['seed']==seed and x['arm']==a) for a in ARMS]
        gains={str(n):r['metrics'][f'test_{n}']['all']['exact_accuracy']-s['metrics'][f'test_{n}']['all']['exact_accuracy'] for n in (8,10)}
        preserve=all(r['metrics'][f'test_{n}'][g]['exact_accuracy']-s['metrics'][f'test_{n}'][g]['exact_accuracy']>=-.05
                     for n in range(3,7) for g in ('distinct','repeated'))
        pairs.append(dict(seed=seed,long_gains=gains,seen_length_preserved=preserve))
    return dict(training_fit=fit,seen_length_reliability=seen,uniform_short_copy=uniform_short,
        extrapolation=transfer,paired=pairs,
        rope_benefit=all(fit.values()) and all(p['seen_length_preserved'] and min(p['long_gains'].values())>=.1 for p in pairs))


def profile():
    assert not DIRECTORY.exists();cases=[]
    for arm in ARMS:
        for n in range(3,7):
            torch.manual_seed(SEEDS[0]);model=make_model(arm);opt=torch.optim.Adam(model.parameters(),lr=.001)
            content=torch.randint(3,11,(840//(n+1),n));prefix,labels=inputs(content);times=[]
            for _ in range(3):
                tick=time.perf_counter();opt.zero_grad(set_to_none=True)
                loss=F.cross_entropy(logits(model,content,prefix).flatten(0,1),labels.flatten());loss.backward();opt.step()
                times.append(time.perf_counter()-tick)
            cases.append(dict(arm=arm,content_length=n,source_length=n+1,batch=len(content),complete_step_seconds=times,
                optimizer_state_bytes=sum(v.numel()*v.element_size() for state in opt.state.values() for v in state.values() if torch.is_tensor(v))))
    DIRECTORY.mkdir();write(DIRECTORY/'profile.json',dict(device='cpu',dtype='float64',threads=1,torch=torch.__version__,
        gpu_used=False,peak_gpu_allocated_bytes=None,peak_gpu_reserved_bytes=None,profile_updates_discarded=True,cases=cases))
    print(json.dumps(cases),flush=True)


def main():
    assert PROTOCOL.exists() and (DIRECTORY/'preflight.json').exists() and not (DIRECTORY/'report.json').exists()
    payload=json.loads(DATA.read_text(encoding='utf-8'));eval_pools,checks=pools_and_checks(payload)
    assert digest(DATA)==json.loads((DATA.parent/'report.json').read_text(encoding='utf-8'))['data_sha256']
    paths=[Path(__file__),ROOT/'experiments/audit_copy_marker_training.py',ROOT/'experiments/copy_marker_model.py',
        ROOT/'tests/test_copy_marker.py',PROTOCOL,DATA,DIRECTORY/'profile.json',DIRECTORY/'preflight.json',
        ROOT/'results/copy_rope_training/report.json',ROOT/'results/copy_source_end_marker/count_recovery/report.json',
        *[ROOT/'experiments'/n for n in ('copy_rope_model.py','copy_rope_training.py','encoder_decoder_copy.py',
          'encoder_decoder_contract.py','copy_length_exposure.py','copy_eos_training.py','audit_copy_rope_training.py',
          'audit_copy_length_exposure.py','audit_copy_eos_training.py','audit_kv_cache_equivalence.py','mqa_training.py',
          'position_distractors.py','recurrent_retrieval.py')]]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    pools={p:torch.tensor(rows) for p,rows in payload['pools'].items()}
    report=dict(status='running',device='cpu',dtype='float64',torch=torch.__version__,parameters=1888,seeds=list(SEEDS),
        arms=list(ARMS),updates_per_run=STEPS,reused_open_data=True,source_dependencies=hashes,input_checks=checks,
        evaluation_pools=eval_pools,controller_checks=controller_checks(),runs=[])
    write(DIRECTORY/'report.json',report)
    for seed in SEEDS:
        torch.manual_seed(seed);initial=make_model('sinusoidal').state_dict();initial_hash=hash_state(initial)
        for arm in ARMS:
            model=make_model(arm);model.load_state_dict(copy.deepcopy(initial));opt=torch.optim.Adam(model.parameters(),lr=.001)
            gen=torch.Generator().manual_seed(seed+1000);stream=hashlib.sha256();ranks=hashlib.sha256()
            run=dict(seed=seed,arm=arm,status='running',initial_sha256=initial_hash,curve=[],metrics={},
                sequences=0,content_targets=0,supervised_targets=0,source_tokens=0,source_end_tokens=0,attention_pairs=0)
            report['runs'].append(run);tick=time.perf_counter()
            for step in range(1,STEPS+1):
                n,idx,raw=sample(pools,gen,'mixed',step);ranks.update(raw)
                stream.update(json.dumps([n,idx.tolist()],separators=(',',':')).encode())
                content=pools[f'train_{n}'][idx];prefix,labels=inputs(content);opt.zero_grad(set_to_none=True)
                z=logits(model,content,prefix);loss=F.cross_entropy(z.flatten(0,1),labels.flatten())
                if step==1:
                    rz=reference(model,content,prefix,arm);rl=F.cross_entropy(rz.flatten(0,1),labels.flatten())
                    grads=torch.autograd.grad(rl,tuple(model.parameters()));le=float((z-rz).detach().abs().max());assert le<=1e-9
                if not bool(torch.isfinite(loss)):
                    run.update(status='failed',failure_step=step);report['status']='failed';write(DIRECTORY/'report.json',report)
                    raise RuntimeError('Nonfinite objective; no restart')
                loss.backward()
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                if step==1:
                    ge=max(float((p.grad-g).abs().max()) for p,g in zip(model.parameters(),grads));assert ge<=1e-9
                    run['first_gradient_checks']=dict(logit_error=le,parameter_gradient_error=ge)
                opt.step();batch=len(content);run['sequences']+=batch;run['content_targets']+=content.numel()
                run['supervised_targets']+=labels.numel();run['source_tokens']+=batch*(n+1);run['source_end_tokens']+=batch
                run['attention_pairs']+=batch*(2*(n+1)**2+(n+1)*(n+2)//2)
                if step%1000==0:
                    row=dict(step=step,minibatch_loss=float(loss.detach()))
                    if step in (1000,3000,6000):
                        row['pools']={p:evaluate(model,rows)[0] for p,rows in eval_pools.items()
                                      if not p.startswith('test_') and not (p.startswith('uniform_') and int(p.split('_')[1])>6)}
                    run['curve'].append(row);write(DIRECTORY/'report.json',report)
                    print(json.dumps(dict(seed=seed,arm=arm,step=step,loss=row['minibatch_loss'],elapsed_seconds=time.perf_counter()-tick)),flush=True)
            cp=DIRECTORY/f'{seed}_{arm}.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=STEPS,generator=gen.get_state()),cp)
            run.update(status='trained',step=STEPS,metrics=run['curve'][-1]['pools'],elapsed_seconds=time.perf_counter()-tick,
                sample_stream_sha256=stream.hexdigest(),rank_stream_sha256=ranks.hexdigest(),
                checkpoint=cp.relative_to(ROOT).as_posix(),checkpoint_sha256=digest(cp));write(DIRECTORY/'report.json',report)
        assert report['runs'][-2]['sample_stream_sha256']==report['runs'][-1]['sample_stream_sha256']
    report['status']='all_training_finished_before_test';write(DIRECTORY/'pretest.json',report)
    for run in report['runs']:
        model=make_model(run['arm']);model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model']);saved={}
        for pool,rows in eval_pools.items():
            met,details=evaluate(model,rows)
            if pool in run['metrics']:assert met==run['metrics'][pool]
            run['metrics'][pool]=met;saved[pool]=details
        path=DIRECTORY/f"{run['seed']}_{run['arm']}_scores.pt";torch.save(saved,path)
        run.update(scores=path.relative_to(ROOT).as_posix(),scores_sha256=digest(path));write(DIRECTORY/'report.json',report)
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    rows=[dict(seed=r['seed'],arm=r['arm'],pool=p,group=g,**m) for r in report['runs'] for p,gs in r['metrics'].items() for g,m in gs.items()]
    with (DIRECTORY/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report.update(status='complete',decision=decision(report['runs']),prior_hashes_unchanged=True,metrics_sha256=digest(DIRECTORY/'metrics.csv'))
    write(DIRECTORY/'report.json',report);print(json.dumps(report['decision']),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if sys.argv[1:]==['profile']:profile()
    else:main()
