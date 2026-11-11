"""One paired copy-length exposure comparison on the existing audited model."""
import copy
import csv
import hashlib
import json
import random
import struct
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from encoder_decoder_contract import ROOT, Model, independent
from encoder_decoder_copy import inputs, logits, generate
from audit_encoder_decoder_copy import controller_checks
from mqa_training import hash_state
from position_distractors import digest
from recurrent_retrieval import write

DIRECTORY = ROOT/'results/copy_length_exposure'
PROTOCOL = ROOT/'docs/protocols/copy-length-exposure.md'
SEEDS = (3061, 3067, 3079)
ARMS = {'short':(3,4), 'mixed':(3,4,5,6)}
STEPS = 6000
TOKENS = 840
COUNTS = {**{f'train_{n}':64 if n==3 else 256 for n in range(3,7)},
          **{f'validation_{n}':16 if n==3 else 64 for n in range(3,7)},
          **{f'test_{n}':16 if n==3 else 128 for n in (3,4,5,6,8,10)}}


def groups(rows):
    n=len(rows[0])
    masks={'all':torch.ones(len(rows),dtype=torch.bool)}
    distinct=torch.tensor([len(set(row))==n for row in rows])
    for name,mask in (('distinct',distinct),('repeated',~distinct)):
        if bool(mask.any()):masks[name]=mask
    return masks


def make_data():
    old=json.loads((ROOT/'results/encoder_decoder_copy_data.json').read_text(encoding='utf-8'))
    historical={tuple(row) for rows in old['pools'].values() for row in rows}
    seen=set(historical); rng=random.Random(2026100306); pools={}
    for name,count in COUNTS.items():
        n=int(name.rsplit('_',1)[1]); rows=[]
        for distinct in ((True,False) if n<=8 else (False,)):
            needed=count//2 if n<=8 else count; found=0
            for attempt in range(100000):
                row=tuple(rng.sample(range(3,11),n)) if distinct else tuple(rng.choices(range(3,11),k=n))
                if (len(set(row))==n)!=distinct or row in seen:continue
                seen.add(row);rows.append(list(row));found+=1
                if found==needed:break
            assert found==needed, 'Fixed proposal limit exhausted.'
        pools[name]=rows
    assert len(seen)==len(historical)+sum(COUNTS.values())
    return dict(generator_seed=2026100306,excludes_historical_sequences=True,pools=pools)


def audit_data(payload):
    old=json.loads((ROOT/'results/encoder_decoder_copy_data.json').read_text(encoding='utf-8'))
    historical={tuple(x) for rows in old['pools'].values() for x in rows}
    seen=set(); total=0
    assert payload['pools'].keys()==COUNTS.keys()
    for name,rows in payload['pools'].items():
        n=int(name.rsplit('_',1)[1]); assert len(rows)==COUNTS[name]
        for i,row in enumerate(rows):
            assert len(row)==n and all(3<=x<=10 for x in row)
            assert tuple(row) not in seen|historical;seen.add(tuple(row))
            assert (len(set(row))==n)==(n<=8 and i<len(rows)//2)
        source=torch.tensor(rows); shifted,labels=inputs(source)
        assert labels.tolist()==[row+[2] for row in rows]
        assert shifted.tolist()==[[1]+row for row in rows]
        total+=labels.numel()
    wrong=[5,4,3,2]; assert wrong!=[3,4,5,2]
    b={}
    # The original baseline helper requires nonempty distinct groups.
    for name,rows in payload['pools'].items():
        b[name]={}
        for kind,operation in (('oracle',lambda r:r),('repeat_first',lambda r:[r[0]]*len(r)),
                               ('reversed',lambda r:list(reversed(r))),('sorted',lambda r:sorted(r))):
            b[name][kind]={g:sum(operation(rows[i])+[2]==rows[i]+[2] for i in torch.where(mask)[0].tolist())/int(mask.sum())
                           for g,mask in groups(rows).items()}
        assert b[name]['oracle']['all']==1
    return dict(passed=True,sequences=len(seen),teacher_targets=total,wrong_reversal_detected=True,
                historical_overlap=0,baselines=b)


def metrics(z,rows,generated):
    source=torch.tensor(rows); _,labels=inputs(source); n=source.shape[1]
    losses=F.cross_entropy(z.flatten(0,1),labels.flatten(),reduction='none').reshape_as(labels)
    correct=z.argmax(-1)==labels; out={}
    for group,mask in groups(rows).items():
        ids=torch.where(mask)[0].tolist(); count=len(ids)
        stopping={'early_eos':0,'on_time_eos':0,'late_eos':0,'missing_eos':0}
        for i in ids:
            output=generated[i]
            if 2 not in output:key='missing_eos'
            else:
                at=output.index(2);key='early_eos' if at<n else 'late_eos' if at>n else 'on_time_eos'
            stopping[key]+=1
        out[group]=dict(sequences=count,targets=count*(n+1),
            exact_accuracy=sum(generated[i]==rows[i]+[2] for i in ids)/count,
            generated_content_exact=sum(generated[i][:n]==rows[i] for i in ids)/count,
            token_accuracy=float(correct[mask].double().mean()),
            content_accuracy=float(correct[mask,:n].double().mean()),loss=float(losses[mask].mean()),
            **{k:v/count for k,v in stopping.items()})
    return out


def evaluate(model,rows):
    source=torch.tensor(rows); scores=[]; outputs=[]
    with torch.no_grad():
        for x in source.split(64):
            shifted,_=inputs(x);scores.append(logits(model,x,shifted))
            outputs.extend(generate(lambda a,b:logits(model,a,b)[:,-1],x,x.shape[1]+3))
    z=torch.cat(scores)
    return metrics(z,rows,outputs),dict(logits=z,generated=outputs)


def sample(pools,gen,arm,step):
    lengths=ARMS[arm]; n=lengths[(step-1)%len(lengths)]; batch=TOKENS//(n+1)
    assert batch*(n+1)==TOKENS and batch%2==0
    ranks=torch.rand(2,105,generator=gen,dtype=torch.float64)
    pool=pools[f'train_{n}'];half=len(pool)//2
    idx=(ranks[:,:batch//2]*half).long();idx[1]+=half
    return n,idx.flatten(),struct.pack('210d',*ranks.flatten().tolist())


def decision(runs):
    assert len(runs)==6
    selected={arm:[r for r in runs if r['arm']==arm] for arm in ARMS}
    fit={arm:all(r['status']=='trained' and all(r['metrics'][f'train_{n}'][g]['exact_accuracy']>=.99
            for n in ARMS[arm] for g in ('distinct','repeated')) for r in selected[arm]) for arm in ARMS}
    held={arm:fit[arm] and all(r['metrics'][f'test_{n}'][g]['exact_accuracy']>=.95
            for r in selected[arm] for n in ARMS[arm] for g in ('distinct','repeated')) for arm in ARMS}
    pairs=[]
    for seed in SEEDS:
        a,b=[next(r for r in runs if r['seed']==seed and r['arm']==arm) for arm in ARMS]
        gain=sum(b['metrics']['test_6'][g]['exact_accuracy']-a['metrics']['test_6'][g]['exact_accuracy'] for g in ('distinct','repeated'))/2
        preserve=all(b['metrics'][f'test_{n}'][g]['exact_accuracy']-a['metrics'][f'test_{n}'][g]['exact_accuracy']>=-.05
            for n in (3,4) for g in ('distinct','repeated'))
        pairs.append(dict(seed=seed,six_symbol_gain=gain,short_preserved=preserve))
    transfer={str(n):held['mixed'] and all(r['metrics'][f'test_{n}'][g]['exact_accuracy']>=.9
            for r in runs if r['arm']=='mixed' for g in (('distinct','repeated') if n==8 else ('repeated',))) for n in (8,10)}
    return dict(training_fit=fit,seen_length_reliability=held,paired=pairs,
        broader_support_benefit=all(fit.values()) and all(p['six_symbol_gain']>=.1 and p['short_preserved'] for p in pairs),
        mixed_extrapolation=transfer)


def profile():
    assert not DIRECTORY.exists();torch.manual_seed(3061)
    cases=[]
    for n in range(3,7):
        model=Model();opt=torch.optim.Adam(model.parameters(),lr=.001)
        x=torch.randint(3,11,(TOKENS//(n+1),n)); y,labels=inputs(x);elapsed=[]
        for _ in range(3):
            tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(logits(model,x,y).flatten(0,1),labels.flatten());loss.backward();opt.step()
            elapsed.append(time.perf_counter()-tick)
        cases.append(dict(length=n,batch=len(x),target_tokens=labels.numel(),complete_step_seconds=elapsed,
            optimizer_state_bytes=sum(v.numel()*v.element_size() for state in opt.state.values() for v in state.values() if torch.is_tensor(v))))
    DIRECTORY.mkdir()
    write(DIRECTORY/'profile.json',dict(device='cpu',dtype='float64',torch=torch.__version__,threads=1,
        gpu_used=False,peak_gpu_allocated_bytes=None,peak_gpu_reserved_bytes=None,cases=cases,
        note='Includes backward and allocated Adam states; timing is a feasibility check, not a benchmark. Profile updates discarded.'))
    print(json.dumps(cases),flush=True)


def main():
    assert PROTOCOL.exists() and (DIRECTORY/'profile.json').exists()
    assert not (DIRECTORY/'report.json').exists() and not (DIRECTORY/'data.json').exists()
    dependencies=[Path(__file__),ROOT/'experiments/audit_copy_length_exposure.py',PROTOCOL,DIRECTORY/'profile.json',ROOT/'results/encoder_decoder_copy_data.json',
        ROOT/'results/encoder_decoder_copy.json',ROOT/'results/encoder_decoder_copy_audit.json',
        *[ROOT/'experiments'/p for p in ('encoder_decoder_contract.py','encoder_decoder_copy.py','audit_encoder_decoder_copy.py',
            'mqa_training.py','position_distractors.py','audit_kv_cache_equivalence.py','recurrent_retrieval.py')]]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in dependencies}
    payload=make_data();checks=audit_data(payload);control=controller_checks()
    write(DIRECTORY/'data.json',payload);pools={k:torch.tensor(v) for k,v in payload['pools'].items()}
    report=dict(status='running',seeds=list(SEEDS),arms={k:list(v) for k,v in ARMS.items()},parameters=1888,
        device='cpu',dtype='float64',torch=torch.__version__,updates_per_run=STEPS,target_tokens_per_update=TOKENS,
        supervised_tokens_per_run=STEPS*TOKENS,source_dependencies=hashes,data_sha256=digest(DIRECTORY/'data.json'),
        input_checks=checks,controller_checks=control,runs=[])
    write(DIRECTORY/'report.json',report)
    for seed in SEEDS:
        torch.manual_seed(seed);initial=Model().state_dict();initial_hash=hash_state(initial)
        for arm,lengths in ARMS.items():
            model=Model();model.load_state_dict(copy.deepcopy(initial));opt=torch.optim.Adam(model.parameters(),lr=.001)
            gen=torch.Generator().manual_seed(seed+1000);rank_hash=hashlib.sha256();stream=hashlib.sha256()
            run=dict(seed=seed,arm=arm,status='running',initial_sha256=initial_hash,curve=[],metrics={},
                sequences=0,supervised_tokens=0,source_tokens=0,attention_pairs=0)
            report['runs'].append(run);tick=time.perf_counter()
            for step in range(1,STEPS+1):
                n,idx,ranks=sample(pools,gen,arm,step);rank_hash.update(ranks)
                stream.update(json.dumps([n,idx.tolist()],separators=(',',':')).encode())
                x=pools[f'train_{n}'][idx];prefix,labels=inputs(x);opt.zero_grad(set_to_none=True)
                z=logits(model,x,prefix);loss=F.cross_entropy(z.flatten(0,1),labels.flatten())
                if step==1:
                    s,t=model.features(x,prefix);rz=independent(model,s,t,torch.zeros_like(x,dtype=torch.bool),torch.zeros_like(prefix,dtype=torch.bool))
                    rl=F.cross_entropy(rz.flatten(0,1),labels.flatten())
                    reference_grad=torch.autograd.grad(rl,tuple(model.parameters()))
                    first_logit_error=float((z-rz).detach().abs().max());assert first_logit_error<=1e-9
                if not bool(torch.isfinite(loss)):
                    run.update(status='failed',failure_step=step,failure='nonfinite loss');report['status']='failed'
                    write(DIRECTORY/'report.json',report);raise RuntimeError(f'Nonfinite loss {seed}/{arm}/{step}')
                loss.backward()
                if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    run.update(status='failed',failure_step=step,failure='nonfinite gradients');report['status']='failed'
                    write(DIRECTORY/'report.json',report)
                    raise RuntimeError(f'Nonfinite gradients {seed}/{arm}/{step}')
                if step==1:
                    grad_error=max(float((p.grad-g).abs().max()) for p,g in zip(model.parameters(),reference_grad));assert grad_error<=1e-9
                    run['first_update_check']=dict(logit_error=first_logit_error,gradient_error=grad_error)
                opt.step()
                run['sequences']+=len(x);run['supervised_tokens']+=labels.numel();run['source_tokens']+=x.numel()
                run['attention_pairs']+=len(x)*(n*n+(n+1)*(n+2)//2+(n+1)*n)
                if step%1000==0:
                    row=dict(step=step,minibatch_loss=float(loss.detach()))
                    if step in (1000,3000,6000):
                        row['pools']={key:evaluate(model,rows)[0] for key,rows in payload['pools'].items()
                            if key.startswith('validation_') or key in [f'train_{n}' for n in lengths]}
                    run['curve'].append(row)
                    print(json.dumps(dict(seed=seed,arm=arm,step=step,loss=row['minibatch_loss'],
                        elapsed_seconds=time.perf_counter()-tick)),flush=True)
                    write(DIRECTORY/'report.json',report)
            run.update(status='trained',step=STEPS,metrics=run['curve'][-1]['pools'],
                elapsed_seconds=time.perf_counter()-tick,rank_stream_sha256=rank_hash.hexdigest(),sample_stream_sha256=stream.hexdigest())
            cp=DIRECTORY/f'{seed}_{arm}.pt'
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=STEPS,generator=gen.get_state()),cp)
            run.update(checkpoint=cp.relative_to(ROOT).as_posix(),checkpoint_sha256=digest(cp))
            assert run['supervised_tokens']==STEPS*TOKENS
            write(DIRECTORY/'report.json',report)
        pair=report['runs'][-2:];assert pair[0]['rank_stream_sha256']==pair[1]['rank_stream_sha256']
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    report['status']='all_training_finished_before_test';write(DIRECTORY/'pretest.json',report)
    for run in report['runs']:
        model=Model();model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model']);saved={}
        for name,rows in payload['pools'].items():
            if name.startswith('train_') and int(name.split('_')[1]) not in ARMS[run['arm']]:continue
            met,ev=evaluate(model,rows)
            if name in run['metrics']:assert met==run['metrics'][name]
            run['metrics'][name]=met;saved[name]=ev
        scores=DIRECTORY/f"{run['seed']}_{run['arm']}_scores.pt";torch.save(saved,scores)
        run.update(scores=scores.relative_to(ROOT).as_posix(),scores_sha256=digest(scores))
        write(DIRECTORY/'report.json',report)
    report.update(status='complete',decision=decision(report['runs']),prior_hashes_unchanged=True)
    with (DIRECTORY/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        rows=[dict(seed=r['seed'],arm=r['arm'],pool=pool,group=g,**m) for r in report['runs'] for pool,gm in r['metrics'].items() for g,m in gm.items()]
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report['metrics_sha256']=digest(DIRECTORY/'metrics.csv');write(DIRECTORY/'report.json',report)
    print(json.dumps(report['decision']),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if sys.argv[1:] == ['profile']:profile()
    else:main()
