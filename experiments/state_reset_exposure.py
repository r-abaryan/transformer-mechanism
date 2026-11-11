"""Paired reset-cue exposure comparison with a fixed diagnostic recipe."""
import copy
import csv
import hashlib
import itertools
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from state_tracking import ROOT, STRATA, model_new, forward, tensors, answer, save_json, provenance
from audit_state_tracking import independent, oracle
from state_recipe_development import rate, evaluate, metric
from mqa_training import hash_state
from position_distractors import digest

DIRECTORY = ROOT/'results/state_reset_exposure'
PROTOCOL = ROOT/'docs/protocols/state-reset-exposure.md'
SEEDS = (3011, 3019, 3023)
STEPS = 6000
BATCH = 64
ARMS = ('correlated', 'independent')


def make_pools():
    original = json.loads((ROOT/'results/state_tracking_data.json').read_text())
    base = {k:v for k,v in original.items() if k.startswith('train/')}
    pools = {k:v for k,v in original.items() if k.split('/')[0] in ('train','validation','test')
             and int(k.split('/')[1]) in (8,12) and not k.endswith('long_tail')}
    rng = random.Random(2026100301)
    families = {}
    for n in (8,12,24):
        trained = {tuple(ids[2:-1]) for ids,y in base.get(f'train/{n}/long_tail', [])}
        if n < 24:
            choices = [v for v in itertools.product((0,1), repeat=n-1) if v not in trained]
            rng.shuffle(choices)
            nv,nt = (16,32) if n==8 else (64,128)
            families[f'validation/{n}'] = choices[:nv]
            families[f'test/{n}'] = choices[nv:nv+nt]
            families[f'train/{n}'] = sorted(trained)
        else:
            choices=set()
            while len(choices)<128: choices.add(tuple(rng.randrange(2) for _ in range(n-1)))
            families[f'test/{n}'] = sorted(choices)
    for prefix,suffixes in families.items():
        agree=[];disagree=[]
        for suffix in suffixes:
            for reset in (0,1):
                for start in (0,1):
                    ids=[4+start,2+reset]+list(suffix)+[6]
                    row=[ids,oracle(ids)]
                    (agree if start==reset else disagree).append(row)
        pools[prefix+'/long_tail_agree']=agree
        pools[prefix+'/long_tail_disagree']=disagree
    return base,pools,families


def prepare():
    torch.set_num_threads(1);torch.manual_seed(3001)
    assert not DIRECTORY.exists()
    base,pools,families=make_pools()
    sources=provenance()
    sources.update({'experiments/state_reset_exposure.py':digest(Path(__file__)),
                    'experiments/state_recipe_development.py':digest(ROOT/'experiments/state_recipe_development.py')})
    # Profile the complete step, including Adam's lazily allocated state.
    x,y=tensors(base['train/12/long_tail'][:BATCH]);x=x.clone();x[::2,0]=9-x[::2,0]
    m=model_new();opt=torch.optim.Adam(m.parameters(),lr=.001)
    tick=time.perf_counter();loss=F.cross_entropy(forward(m,x)[:,-1],y);loss.backward();opt.step()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters())
    profile=dict(python=platform.python_version(),torch=torch.__version__,device='cpu',dtype='float64',threads=1,
                 parameters=sum(p.numel() for p in m.parameters()),batch=BATCH,operation_length=12,
                 optimizer_entries=len(opt.state),optimizer_tensor_bytes=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v)),
                 full_step_seconds=time.perf_counter()-tick,gpu_peak_allocated_bytes=0,gpu_peak_reserved_bytes=0,cpu_peak_measured=False,
                 source_dependencies=sources,input_sha256=digest(ROOT/'results/state_tracking_data.json'))
    DIRECTORY.mkdir();save_json(DIRECTORY/'profile.json',profile)
    save_json(DIRECTORY/'data.json',dict(training_base=base,evaluation=pools,suffix_families=families))
    print(json.dumps(profile),flush=True)


def check_data(data):
    base,pools,families=data['training_base'],data['evaluation'],data['suffix_families']
    labels=wrong=0
    for key,rows in pools.items():
        assert sum(y for ids,y in rows)==len(rows)//2
        for ids,y in rows:
            assert y==oracle(ids)==answer(ids[0]-4,ids[1:-1]);labels+=1
            if '/long_tail_' in key:
                assert all(v in (0,1) for v in ids[2:-1]) and ids[1] in (2,3)
                assert ((ids[0]-4)==(ids[1]-2))==key.endswith('_agree')
                flipped=[9-ids[0]]+ids[1:];assert oracle(flipped)==y
                wrong+=((ids[0]-4) ^ (ids[2:-1].count(1)%2))!=y
    for n in (8,12):
        sets=[set(map(tuple,families[f'{split}/{n}'])) for split in ('train','validation','test')]
        assert not any(a & b for i,a in enumerate(sets) for b in sets[i+1:])
        assert sets[0]=={tuple(ids[2:-1]) for ids,y in base[f'train/{n}/long_tail']}
    assert wrong>0
    records=[]
    for seed in SEEDS:
        torch.manual_seed(seed);m=model_new()
        x,y=tensors(pools['train/12/long_tail_disagree'][:8])
        a,b=forward(m,x)[:,-1],independent(m,x)[:,-1]
        ga=torch.autograd.grad(F.cross_entropy(a,y),list(m.parameters()),retain_graph=True)
        gb=torch.autograd.grad(F.cross_entropy(b,y),list(m.parameters()))
        bad=torch.autograd.grad(F.cross_entropy(a,1-y),list(m.parameters()))
        oe=float((a-b).abs().max().detach());ge=max(float((u-v).abs().max()) for u,v in zip(ga,gb))
        ne=max(float((u-v).abs().max()) for u,v in zip(ga,bad))
        assert max(oe,ge)<=1e-10 and ne>1e-6
        records.append(dict(seed=seed,logit_error=oe,gradient_error=ge,wrong_label_gradient_effect=ne))
    return dict(checked_labels=labels,wrong_initial_base_label_errors=wrong,suffix_split_disjoint=True,initial_gradient_cases=records)


def encode_base(base):
    result={}
    for n in (8,12):
        rows=[row for g in STRATA for row in base[f'train/{n}/{g}']]
        x,y=tensors(rows)
        mask=torch.tensor([g=='long_tail' for g in STRATA for _ in base[f'train/{n}/{g}']])
        result[n]=(x,y,mask)
    return result


def training_metrics(model,pools):
    return evaluate(model,{k:v for k,v in pools.items() if k.startswith('train/')})[0]


def paired_reset_metrics(scores,pools):
    result={}
    prefixes=sorted({k.rsplit('/',1)[0] for k in pools if '/long_tail_' in k})
    for prefix in prefixes:
        a=scores[prefix+'/long_tail_agree'];b=scores[prefix+'/long_tail_disagree']
        y=tensors(pools[prefix+'/long_tail_agree'])[1]
        assert torch.equal(y,tensors(pools[prefix+'/long_tail_disagree'])[1])
        pa,pb=a.argmax(-1),b.argmax(-1)
        result[prefix]=dict(pairs=len(y),prediction_changes=float((pa!=pb).double().mean()),
                            both_correct=float(((pa==y)&(pb==y)).double().mean()),
                            mean_probability_change=float(((a[:,1]-a[:,0]).sigmoid()-(b[:,1]-b[:,0]).sigmoid()).abs().mean()))
    return result


def decide(runs):
    comparisons=[]
    for seed in SEEDS:
        pair={r['arm']:r for r in runs if r['seed']==seed}
        if len(pair)!=2 or any(r['status']!='trained' for r in pair.values()):
            comparisons.append(dict(seed=seed,complete=False,eligible=False));continue
        a,b=pair['correlated'],pair['independent']
        def support(r):
            return {k:v for k,v in r['metrics'].items() if k.startswith('train/') and
                    (r['arm']=='independent' or not k.endswith('long_tail_disagree'))}
        afit=min(v['accuracy'] for v in support(a).values());bfit=min(v['accuracy'] for v in support(b).values())
        gain=b['metrics']['test/12/long_tail_disagree']['accuracy']-a['metrics']['test/12/long_tail_disagree']['accuracy']
        robust=b['reset_pairs']['test/12']['both_correct']
        comparisons.append(dict(seed=seed,complete=True,correlated_worst_support_fit=afit,independent_worst_support_fit=bfit,
                                disagree_gain=gain,independent_both_correct=robust,
                                behavioral_rule_pass=gain>=.20 and robust>=.90,eligible=afit>=.99 and bfit>=.99))
    return dict(per_seed=comparisons,adoption_pass=all(v.get('eligible') and v.get('behavioral_rule_pass') for v in comparisons),
                rule='each seed: both arms >=99% on training support; independent test12 disagree gain >=20 points and both-correct >=90%; no selection or retuning')


def train():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert PROTOCOL.exists() and (DIRECTORY/'profile.json').exists()
    assert not (DIRECTORY/'report.json').exists() and not (DIRECTORY/'checkpoints').exists()
    profile=json.loads((DIRECTORY/'profile.json').read_text())
    assert all(digest(ROOT/p)==h for p,h in profile['source_dependencies'].items())
    data=json.loads((DIRECTORY/'data.json').read_text());checks=check_data(data)
    base,pools=data['training_base'],data['evaluation'];encoded=encode_base(base)
    preserved={p:digest(ROOT/p) for p in ['results/state_recipe_development/report.json','results/state_recipe_errors/report.json']}
    historical=json.loads((ROOT/'results/state_recipe_development/report.json').read_text())
    preserved.update({r['checkpoint']:digest(ROOT/r['checkpoint']) for r in historical['runs']})
    report=dict(phase='running',profile_sha256=digest(DIRECTORY/'profile.json'),data_sha256=digest(DIRECTORY/'data.json'),
                protocol_sha256=digest(PROTOCOL),source_dependencies=profile['source_dependencies'],preserved=preserved,
                preflight=checks,seeds=list(SEEDS),steps=STEPS,batch=BATCH,runs=[],heldout_used_for_selection=False)
    path=DIRECTORY/'report.json';save_json(path,report);(DIRECTORY/'checkpoints').mkdir()
    for si,seed in enumerate(SEEDS):
        torch.manual_seed(seed);base_model=model_new();initial=hash_state(base_model.state_dict())
        order=ARMS if si%2==0 else ARMS[::-1]
        for arm in order:
            m=copy.deepcopy(base_model);opt=torch.optim.Adam(m.parameters(),lr=.001)
            gen=torch.Generator().manual_seed(seed+1000);aug=torch.Generator().manual_seed(seed+2000)
            stream=hashlib.sha256();augmentation=hashlib.sha256();tick=time.perf_counter();counts={}
            run=dict(seed=seed,arm=arm,status='running',step=0,initial_sha256=initial,curve=[])
            report['runs'].append(run);save_json(path,report)
            try:
                for step in range(1,STEPS+1):
                    n=8 if step%2 else 12;x,y,mask=encoded[n]
                    ix=torch.randint(len(y),(BATCH,),generator=gen)
                    flip=torch.randint(2,(BATCH,),generator=aug).bool()
                    stream.update(json.dumps(ix.tolist()).encode());augmentation.update(json.dumps(flip.tolist()).encode())
                    batch=x[ix].clone();long=mask[ix]
                    if arm=='independent':batch[long & flip,0]=9-batch[long & flip,0]
                    for initial_bit,reset in batch[long][:,:2].tolist():
                        key=f'{n}/{initial_bit-4}/{reset-2}';counts[key]=counts.get(key,0)+1
                    for g in opt.param_groups:g['lr']=rate(step,.001,True)
                    opt.zero_grad();loss=F.cross_entropy(forward(m,batch)[:,-1],y[ix])
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
                    loss.backward()
                    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters()):raise FloatingPointError('Nonfinite gradient')
                    opt.step();run['step']=step
                    if not all(bool(torch.isfinite(p).all()) for p in m.parameters()):raise FloatingPointError('Nonfinite parameter')
                    if step%1000==0:
                        metrics=training_metrics(m,pools);run['curve'].append(dict(step=step,training=metrics))
                        save_json(path,report)
                        print(json.dumps(dict(seed=seed,arm=arm,step=step,train12_disagree=metrics['train/12/long_tail_disagree']['accuracy'],train12_agree=metrics['train/12/long_tail_agree']['accuracy'])),flush=True)
                run['status']='trained'
            except Exception as exc:
                run.update(status='failed',failure=repr(exc))
            cp=DIRECTORY/'checkpoints'/f'{seed}_{arm}.pt'
            torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),generator=gen.get_state(),augmentation_generator=aug.get_state(),steps=run['step']),cp)
            run.update(checkpoint=cp.relative_to(ROOT).as_posix(),checkpoint_sha256=digest(cp),stream_sha256=stream.hexdigest(),
                       augmentation_stream_sha256=augmentation.hexdigest(),exposure_counts=counts,seconds=time.perf_counter()-tick,
                       examples=run['step']*BATCH,supervised_targets=run['step']*BATCH,
                       input_tokens=sum(BATCH*(10 if s%2 else 14) for s in range(1,run['step']+1)))
            if run['status']=='trained':
                run['metrics'],scores=evaluate(m,pools);run['reset_pairs']=paired_reset_metrics(scores,pools)
                sp=DIRECTORY/'checkpoints'/f'{seed}_{arm}_scores.pt';torch.save(scores,sp)
                run.update(scores=sp.relative_to(ROOT).as_posix(),scores_sha256=digest(sp))
            save_json(path,report)
    report.update(phase='finished',decision=decide(report['runs']))
    assert all(digest(ROOT/p)==h for p,h in preserved.items());save_json(path,report)
    print(json.dumps(report['decision']),flush=True)


def audit():
    torch.set_num_threads(1)
    r=json.loads((DIRECTORY/'report.json').read_text());data=json.loads((DIRECTORY/'data.json').read_text())
    assert digest(PROTOCOL)==r['protocol_sha256'] and digest(DIRECTORY/'data.json')==r['data_sha256']
    assert all(digest(ROOT/p)==h for p,h in {**r['source_dependencies'],**r['preserved']}.items())
    pools=data['evaluation'];encoded=encode_base(data['training_base']);cells=[];pair_checks=[];records=[]
    for seed in SEEDS:
        gen=torch.Generator().manual_seed(seed+1000);aug=torch.Generator().manual_seed(seed+2000)
        stream=hashlib.sha256();augmentation=hashlib.sha256();counts={a:{} for a in ARMS}
        for step in range(1,STEPS+1):
            n=8 if step%2 else 12;x,y,mask=encoded[n]
            ix=torch.randint(len(y),(BATCH,),generator=gen);flip=torch.randint(2,(BATCH,),generator=aug).bool()
            stream.update(json.dumps(ix.tolist()).encode());augmentation.update(json.dumps(flip.tolist()).encode())
            for arm in ARMS:
                batch=x[ix].clone();long=mask[ix]
                if arm=='independent':batch[long & flip,0]=9-batch[long & flip,0]
                for start,reset in batch[long][:,:2].tolist():
                    key=f'{n}/{start-4}/{reset-2}';counts[arm][key]=counts[arm].get(key,0)+1
        for run in [v for v in r['runs'] if v['seed']==seed]:
            assert run['status']=='trained' and run['step']==STEPS
            assert run['stream_sha256']==stream.hexdigest() and run['augmentation_stream_sha256']==augmentation.hexdigest()
            assert run['exposure_counts']==counts[run['arm']]
            saved=torch.load(ROOT/run['checkpoint'],weights_only=True)
            assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256']
            assert saved['steps']==STEPS and torch.equal(saved['generator'],gen.get_state()) and torch.equal(saved['augmentation_generator'],aug.get_state())
            assert all(int(s['step'])==STEPS for s in saved['optimizer']['state'].values())
            assert abs(saved['optimizer']['param_groups'][0]['lr']-.0001)<1e-15
            torch.manual_seed(seed);m=model_new();assert hash_state(m.state_dict())==run['initial_sha256'];m.load_state_dict(saved['model'])
            _,reference=evaluate(m,pools,independent)
            assert digest(ROOT/run['scores'])==run['scores_sha256'];native=torch.load(ROOT/run['scores'],weights_only=True)
            for key,z in reference.items():
                labels=[oracle(ids) for ids,y in pools[key]];assert labels==[y for ids,y in pools[key]]
                oe=float((z-native[key]).abs().max());changes=int((z.argmax(-1)!=native[key].argmax(-1)).sum())
                correct=0;losses=[]
                for row,y in zip(z.tolist(),labels):
                    correct+=max(range(2),key=lambda j:row[j])==y;mx=max(row)
                    losses.append(mx+math.log(sum(math.exp(v-mx) for v in row))-row[y])
                me=max(abs(correct/len(labels)-run['metrics'][key]['accuracy']),abs(sum(losses)/len(labels)-run['metrics'][key]['loss']))
                assert max(oe,me)<=1e-10 and changes==0
                cells.append(dict(seed=seed,arm=run['arm'],pool=key,count=len(labels),logit_error=oe,metric_error=me,argmax_changes=changes))
                records.append(dict(seed=seed,arm=run['arm'],pool=key,**run['metrics'][key]))
            for prefix,val in run['reset_pairs'].items():
                a=reference[prefix+'/long_tail_agree'].tolist();b=reference[prefix+'/long_tail_disagree'].tolist()
                ys=[y for ids,y in pools[prefix+'/long_tail_agree']]
                changed=good=0;prob=[]
                def sig(v):return 1/(1+math.exp(-v)) if v>=0 else math.exp(v)/(1+math.exp(v))
                for aa,bb,y in zip(a,b,ys):
                    pa=int(aa[1]>aa[0]);pb=int(bb[1]>bb[0]);changed+=pa!=pb;good+=pa==y and pb==y
                    prob.append(abs(sig(aa[1]-aa[0])-sig(bb[1]-bb[0])))
                pe=max(abs(changed/len(ys)-val['prediction_changes']),abs(good/len(ys)-val['both_correct']),abs(sum(prob)/len(ys)-val['mean_probability_change']))
                assert pe<=1e-10;pair_checks.append(dict(seed=seed,arm=run['arm'],prefix=prefix,error=pe))
    assert decide(r['runs'])==r['decision']
    out=dict(passed=True,cells=len(cells),predictions=sum(c['count'] for c in cells),max_logit_error=max(c['logit_error'] for c in cells),
             max_metric_error=max(c['metric_error'] for c in cells),max_reset_pair_error=max(v['error'] for v in pair_checks),argmax_changes=0,
             stream_replays=6,report_sha256=digest(DIRECTORY/'report.json'),evaluation_cells=cells,reset_pair_checks=pair_checks)
    assert not (DIRECTORY/'audit.json').exists();save_json(DIRECTORY/'audit.json',out)
    with (DIRECTORY/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps({k:v for k,v in out.items() if k not in ('evaluation_cells','reset_pair_checks')}),flush=True)


if __name__=='__main__':
    {'prepare':prepare,'train':train,'audit':audit}[sys.argv[1]]()
