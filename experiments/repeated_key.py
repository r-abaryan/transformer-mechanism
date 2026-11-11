"""Latest-write recall: frozen matched positional comparison.

Reuse positional forward math. The training loop is archived separately so the
previous executed source and its provenance remain unchanged.
"""
import hashlib
import itertools
import json
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from positional_comparison import ROOT, forward
from reserved_cycle import model_new
from position_training import weight_hash
from position_distractors import digest

SEEDS=(1013,1031,1051,1069,1091)


def family(pairs):
    return min(tuple(sorted((k,12+(v-12+s)%6) for k,v in pairs)) for s in range(6))


def make_data(path):
    rng=random.Random(20270104); datasets={}; identities={}; seen=set()
    for n in (3,4,6):
        splits={'train':800,'validation':100,'test':200} if n<6 else {'test':200}
        for split,count in splits.items():
            rows=[]; ids=[]
            for _ in range(count):
                while True:
                    keys=rng.sample(range(12),n-1); query=keys[0]
                    pairs=list(zip(keys+[query],rng.sample(range(12,18),n)))
                    signature=family(pairs)
                    if signature not in seen: break
                seen.add(signature); ids.append(signature)
                orders=[]; used=set()
                while len(orders)<6:
                    order=tuple(rng.sample(pairs,n)); reverse=order[::-1]
                    if order in used: continue
                    orders.extend((order,reverse)); used.update((order,reverse))
                for shift in range(6):
                    for order in orders:
                        renamed=[(k,12+(v-12+shift)%6) for k,v in order]
                        label=next(v for k,v in reversed(renamed) if k==query)
                        rows.append(([t for pair in renamed for t in pair]+[18,query],label))
            datasets[f'{split}_{n}']=rows; identities[f'{split}_{n}']=ids
    payload=dict(generator_seed=20270104,datasets=datasets,families=identities)
    path.write_text(json.dumps(payload,separators=(',',':'))+'\n')
    return payload


def measure(s,y,x):
    n=(x.shape[1]-2)//2; prediction=s.argmax(-1); correct=prediction==y
    matches=x[:,:-2:2]==x[:,-1:]
    latest=n-1-matches.flip(1).long().argmax(1)
    first=matches.long().argmax(1); old=x[torch.arange(len(x)),2*first+1]
    return dict(accuracy=correct.double().mean().item(),loss=F.cross_entropy(s,y).item(),
                family_accuracy=correct.reshape(-1,36).all(1).double().mean().item(),
                reversal_accuracy=correct.reshape(-1,2).all(1).double().mean().item(),
                obsolete_rate=(prediction==old).double().mean().item(),
                last_pair_heuristic=(x[:,-3]==y).double().mean().item(),
                latest_position_accuracy=[correct[latest==i].double().mean().item() for i in range(1,n)])


def evaluate(model,data,group):
    x,y=data
    with torch.no_grad(): scores=torch.cat([forward(model,c,group)[:,-1] for c in x.split(256)])
    return measure(scores,y,x),scores


def decision(runs):
    if any(r['status']!='trained' for r in runs): return dict(keep=False,reason='training failure')
    pairs=[tuple(next(r for r in runs if r['seed']==seed and r['group']==g) for g in ('baseline','rope')) for seed in SEEDS]
    gains={str(n):[c['test'][str(n)]['accuracy']-b['test'][str(n)]['accuracy'] for b,c in pairs] for n in (3,4,6)}
    checks=dict(six_all_90=all(c['test']['6']['accuracy']>=.9 for b,c in pairs),six_gain_5=sum(gains['6'])/5>=.05,six_no_seed_worse=min(gains['6'])>=0)
    for n in (3,4):
        checks[f'{n}_all_99']=all(c['test'][str(n)]['accuracy']>=.99 for b,c in pairs)
        checks[f'{n}_preserved']=sum(gains[str(n)])/5>=-.005
    return dict(keep=all(checks.values()),checks=checks,paired_gains=gains)


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    ip=ROOT/'results/repeated_key_data.json'; rp=ROOT/'results/repeated_key.json'; directory=ROOT/'results/repeated_key_checkpoints'; directory.mkdir(exist_ok=True)
    provenance={n:digest(ROOT/'experiments'/n) for n in ('repeated_key.py','audit_repeated_key.py','audit_positional_comparison.py','activation_paths.py','positional_comparison.py','reserved_cycle.py','compare_depth.py','position_training.py','position_distractors.py')}
    provenance['protocol']=digest(ROOT/'docs/repeated-key-protocol.md')
    if rp.exists():
        report=json.loads(rp.read_text()); assert report['provenance']==provenance and report['input_sha256']==digest(ip)
        if report['phase']=='complete': print('Already complete.'); return
        payload=json.loads(ip.read_text())
    else:
        assert not ip.exists(); payload=make_data(ip)
        report=dict(phase='training',provenance=provenance,input_sha256=digest(ip),torch=torch.__version__,runs=[])
    from audit_repeated_key import audit_inputs
    verification=json.loads((ROOT/'results/positional_comparison_verification.json').read_text())
    assert verification['passed'] and verification['source_sha256']==digest(ROOT/'experiments/positional_comparison.py')
    report['input_checks']=audit_inputs(payload)
    data={k:(torch.tensor([x for x,y in rows]),torch.tensor([y for x,y in rows])) for k,rows in payload['datasets'].items()}
    def save():
        tmp=rp.with_suffix('.tmp'); tmp.write_text(json.dumps(report,indent=2)+'\n'); tmp.replace(rp)
    save()
    for seed in SEEDS:
        for group in ('baseline','rope'):
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
                run['curve'].append(dict(step=0,validation={str(n):evaluate(model,data[f'validation_{n}'],group)[0] for n in (3,4)}))
            if start==0 and not path.exists():
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),g3=g3.get_state(),g4=g4.get_state(),rng=torch.get_rng_state(),run=dict(run),provenance=provenance),path)
            h3=hashlib.sha256(); h4=hashlib.sha256()
            rg3=torch.Generator().manual_seed(seed+1000); rg4=torch.Generator().manual_seed(seed+2000)
            for _ in range(start):
                h3.update(json.dumps(torch.randint(len(data['train_3'][0]),(64,),generator=rg3).tolist()).encode())
                h4.update(json.dumps(torch.randint(len(data['train_4'][0]),(64,),generator=rg4).tolist()).encode())
            tick=time.perf_counter()
            for step in range(start+1,6001):
                i3=torch.randint(len(data['train_3'][0]),(64,),generator=g3); i4=torch.randint(len(data['train_4'][0]),(64,),generator=g4)
                h3.update(json.dumps(i3.tolist()).encode()); h4.update(json.dumps(i4.tolist()).encode())
                n=4 if step%2==0 else 3; idx=i4 if n==4 else i3; x,y=data[f'train_{n}']
                opt.zero_grad(); loss=F.cross_entropy(forward(model,x[idx],group)[:,-1],y[idx]); finite=bool(torch.isfinite(loss))
                if finite:
                    loss.backward(); finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                if finite: opt.step()
                else: run.update(status='failed',failure_step=step)
                run['tokens']+=64*(2*n+2); run['attention_pairs']+=64*(2*n+2)**2
                if step%1000==0 or not finite:
                    run['curve'].append(dict(step=step,validation={str(n):evaluate(model,data[f'validation_{n}'],group)[0] for n in (3,4)}))
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
        for n in (3,4,6): run['test'][str(n)],scores[str(n)]=evaluate(model,data[f'test_{n}'],run['group'])
        details.append(dict(seed=run['seed'],group=run['group'],scores=scores)); save()
    dp=ROOT/'results/repeated_key_scores.pt'; torch.save(details,dp)
    report.update(phase='complete',scores_sha256=digest(dp),decision=decision(report['runs'])); save(); print(json.dumps(report['decision']))


if __name__=='__main__': main()
