"""Fixed three-only versus three/four exposure; reserved six-pair evaluation."""
import hashlib
import itertools
import json
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from reserved_cycle import model_new
from position_training import shifted, weight_hash
from position_distractors import digest, logits

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(761,787,809,839,863)
EXCLUSIONS=('reserved_cycle_splits.json','position_training_splits.json','joint_routing_confirmation_inputs.json',
            'canonical_positions_inputs.json','final_length_inputs.json','norm_placement_splits.json',
            'norm_extension_inputs.json','key_content_inputs.json')


def identity(x): return tuple(sorted(zip(x[:-2:2],x[1:-2:2])))


def excluded():
    maps=set(); hashes={}
    def walk(obj):
        if isinstance(obj,dict):
            for v in obj.values(): walk(v)
        elif isinstance(obj,list):
            if len(obj) in (8,10,14) and all(isinstance(t,int) for t in obj) and obj[-2]==18:
                assert all(0<=k<12 and 12<=v<18 for k,v in identity(obj))
                maps.add(identity(obj))
            else:
                for v in obj: walk(v)
    for name in EXCLUSIONS:
        p=ROOT/'results'/name; hashes[name]=digest(p); walk(json.loads(p.read_text()))
    return maps,hashes


def make_data(path):
    old,hashes=excluded(); rng=random.Random(20270101); datasets={}
    for n,counts in ((3,{'train':80,'validation':10,'test':20}),(4,{'train':107,'validation':14,'test':28}),(6,{'test':420})):
        groups={k:[] for k in counts}; used=set(old)
        for values in itertools.combinations(range(12,18),n):
            for split,count in counts.items():
                for _ in range(count):
                    while True:
                        m=tuple(zip(sorted(rng.sample(range(12),n)),rng.sample(values,n)))
                        if m not in used: break
                    used.add(m); groups[split].append(m)
        for split,maps in groups.items():
            rows=[]
            for m in maps:
                for order in rng.sample(list(itertools.permutations(m)),6):
                    for q,v in m: rows.append(([t for pair in order for t in pair]+[18,q],v))
            datasets[f'{split}_{n}']=rows
    payload=dict(exclusion_hashes=hashes,excluded_mappings=[list(m) for m in sorted(old)],datasets=datasets)
    path.write_text(json.dumps(payload,separators=(',',':'))+'\n')
    return payload


def measure(s,y,x):
    n=(x.shape[1]-2)//2; correct=s.argmax(-1)==y
    rank=(x[:,:-2:2]==x[:,-1:]).long().argmax(-1)
    return dict(accuracy=correct.double().mean().item(),loss=F.cross_entropy(s,y).item(),
                mapping_accuracy=correct.reshape(-1,6*n).all(1).double().mean().item(),
                rank_accuracy=[correct[rank==i].double().mean().item() for i in range(n)])


def evaluate(model,data):
    x,y=data; model.eval(); s=logits(model,x,torch.arange(x.shape[1]).expand_as(x)+4); model.train()
    return measure(s,y,x),s


def decision(runs):
    pairs=[(next(r for r in runs if r['seed']==seed and r['group']=='baseline'),next(r for r in runs if r['seed']==seed and r['group']=='mixed')) for seed in SEEDS]
    if any(r['status']!='trained' for r in runs): return dict(keep=False,reason='training failure')
    six=[m['test']['6']['accuracy']-b['test']['6']['accuracy'] for b,m in pairs]
    three=[m['test']['3']['accuracy']-b['test']['3']['accuracy'] for b,m in pairs]
    checks=dict(six_all_90=all(m['test']['6']['accuracy']>=.9 for b,m in pairs),six_gain_5=sum(six)/5>=.05,
                six_no_seed_worse=min(six)>=0,three_all_99=all(m['test']['3']['accuracy']>=.99 for b,m in pairs),three_preserved=sum(three)/5>=-.005)
    return dict(keep=all(checks.values()),checks=checks,paired_six_gains=six,paired_three_gains=three)


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    ip=ROOT/'results/length_exposure_inputs.json'; rp=ROOT/'results/length_exposure.json'; directory=ROOT/'results/length_exposure_checkpoints'; directory.mkdir(exist_ok=True)
    provenance={n:digest(ROOT/'experiments'/n) for n in ('length_exposure.py','reserved_cycle.py','compare_depth.py','position_training.py','position_distractors.py')}
    provenance['protocol']=digest(ROOT/'docs/length-exposure-protocol.md')
    if rp.exists():
        report=json.loads(rp.read_text()); assert report['provenance']==provenance and report['input_sha256']==digest(ip)
        if report['phase']=='complete': print('Already complete.'); return
        payload=json.loads(ip.read_text())
    else:
        assert not ip.exists(); payload=make_data(ip)
        report=dict(phase='training',provenance=provenance,input_sha256=digest(ip),torch=torch.__version__,runs=[])
    data={k:(torch.tensor([x for x,y in rows]),torch.tensor([y for x,y in rows])) for k,rows in payload['datasets'].items()}
    def save():
        tmp=rp.with_suffix('.tmp'); tmp.write_text(json.dumps(report,indent=2)+'\n'); tmp.replace(rp)
    save()
    for seed in SEEDS:
        for group in ('baseline','mixed'):
            run=next((r for r in report['runs'] if r['seed']==seed and r['group']==group),None)
            path=directory/f'{group}_{seed}.pt'
            if run and run['status'] in ('trained','failed'):
                assert digest(path)==run['checkpoint_sha256']; continue
            torch.manual_seed(seed); model=model_new(); assert sum(p.numel() for p in model.parameters())==5088
            initial=weight_hash(model); opt=torch.optim.Adam(model.parameters(),lr=.001)
            g3=torch.Generator().manual_seed(seed+1000); g4=torch.Generator().manual_seed(seed+2000); start=0
            if path.exists():
                state=torch.load(path,weights_only=True); assert state['provenance']==provenance
                model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); g3.set_state(state['g3']); g4.set_state(state['g4']); torch.set_rng_state(state['rng'])
                run=state['run']; start=run['step']; report['runs']=[r for r in report['runs'] if (r['seed'],r['group'])!=(seed,group)]; report['runs'].append(run)
            elif run is None:
                run=dict(seed=seed,group=group,initial_sha256=initial,status='running',step=0,curve=[],tokens=0,attention_pairs=0,elapsed_seconds=0.)
                report['runs'].append(run)
                run['curve'].append(dict(step=0,validation={str(n):evaluate(model,data[f'validation_{n}'])[0] for n in (3,4)}))
            h3=hashlib.sha256(); h4=hashlib.sha256()
            rg3=torch.Generator().manual_seed(seed+1000); rg4=torch.Generator().manual_seed(seed+2000)
            for _ in range(start):
                h3.update(json.dumps(torch.randint(len(data['train_3'][0]),(64,),generator=rg3).tolist()).encode())
                h4.update(json.dumps(torch.randint(len(data['train_4'][0]),(64,),generator=rg4).tolist()).encode())
            tick=time.perf_counter()
            for step in range(start+1,6001):
                i3=torch.randint(len(data['train_3'][0]),(64,),generator=g3); i4=torch.randint(len(data['train_4'][0]),(64,),generator=g4)
                h3.update(json.dumps(i3.tolist()).encode()); h4.update(json.dumps(i4.tolist()).encode())
                n=4 if group=='mixed' and step%2==0 else 3; idx=i4 if n==4 else i3; x,y=data[f'train_{n}']
                opt.zero_grad(); loss=F.cross_entropy(shifted(model,x[idx],4),y[idx]); finite=bool(torch.isfinite(loss))
                if finite:
                    loss.backward(); finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                if finite: opt.step()
                else: run.update(status='failed',failure_step=step)
                run['tokens']+=64*(2*n+2); run['attention_pairs']+=64*(2*n+2)**2
                if step%1000==0 or not finite:
                    run['curve'].append(dict(step=step,validation={str(n):evaluate(model,data[f'validation_{n}'])[0] for n in (3,4)}))
                    run.update(step=step,stream3_sha256=h3.hexdigest(),stream4_sha256=h4.hexdigest())
                    if step==6000 and finite: run['status']='trained'
                    run['elapsed_seconds']+=time.perf_counter()-tick
                    state=dict(model=model.state_dict(),optimizer=opt.state_dict(),g3=g3.get_state(),g4=g4.get_state(),rng=torch.get_rng_state(),run=dict(run),provenance=provenance)
                    tmp=path.with_suffix('.tmp'); torch.save(state,tmp); tmp.replace(path)
                    run.update(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=digest(path)); save(); tick=time.perf_counter()
                    print(json.dumps(dict(seed=seed,group=group,step=step,validation=run['curve'][-1]['validation'])),flush=True)
                if not finite: break
    assert len(report['runs'])==10; report['phase']='all_attempts_finished_before_test'; save(); details=[]
    for run in report['runs']:
        if run['status']!='trained': continue
        cp=ROOT/run['checkpoint']; assert digest(cp)==run['checkpoint_sha256']
        model=model_new(); model.load_state_dict(torch.load(cp,weights_only=True)['model']); run['test']={}; scores={}
        for n in (3,4,6): run['test'][str(n)],scores[str(n)]=evaluate(model,data[f'test_{n}'])
        details.append(dict(seed=run['seed'],group=run['group'],scores=scores)); save()
    dp=ROOT/'results/length_exposure_scores.pt'; torch.save(details,dp)
    report.update(phase='complete',scores_sha256=digest(dp),decision=decision(report['runs'])); save(); print(json.dumps(report['decision']))


if __name__=='__main__': main()
