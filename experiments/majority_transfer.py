"""One bounded aggregation task; reuse the frozen, verified MQA architecture."""
import hashlib
import itertools
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from mqa_training import ROOT, model_new as recall_model, forward, hash_state
from position_distractors import digest

SEEDS = (1613, 1627, 1657, 1669, 1693)
LENGTHS = (9, 11, 15, 21)


def model_new():
    model = recall_model('mqa')
    model.embedding = nn.Embedding(3, 16).double()
    model.output = nn.Linear(16, 2, bias=False).double()
    return model


def orbit(bits):
    x = tuple(bits); c = tuple(1-b for b in x)
    return sorted({x, x[::-1], c, c[::-1]})


def make_data(path):
    rng = random.Random(20270111); families = {}; datasets = {}
    for n, counts in ((9, (80, 20, 36)), (11, (350, 50, 128))):
        candidates = sorted({orbit(x)[0] for x in itertools.product((0, 1), repeat=n)})
        assert len(candidates) == sum(counts) == (2**n + 2**((n+1)//2))//4
        rng.shuffle(candidates); cursor = 0
        for split, count in zip(('train', 'validation', 'test'), counts):
            families[f'{split}_{n}'] = candidates[cursor:cursor+count]; cursor += count
    for n in (15, 21):
        chosen = []; used = set()
        while len(chosen) < 200:
            key = orbit([rng.randrange(2) for _ in range(n)])[0]
            if key not in used: chosen.append(key); used.add(key)
        families[f'test_{n}'] = chosen
    for key, representatives in families.items():
        datasets[key] = [[list(x)+[2], int(2*sum(x)>len(x))]
                         for representative in representatives for x in orbit(representative)]
    payload = dict(generator_seed=20270111, families=families, datasets=datasets)
    path.write_text(json.dumps(payload, separators=(',', ':'))+'\n'); return payload


def tensors(rows):
    return torch.tensor([x for x,y in rows]), torch.tensor([y for x,y in rows])


def measure(scores, x, y, representatives):
    correct = scores.argmax(-1) == y; n = x.shape[1]-1
    margins = (2*x[:, :-1].sum(-1)-n).abs()
    strata = {str(int(m)): dict(count=int((margins==m).sum()),
              accuracy=correct[margins==m].double().mean().item()) for m in margins.unique()}
    sizes = [len(orbit(r)) for r in representatives]; cursor = 0; family_correct = 0
    for size in sizes:
        family_correct += int(correct[cursor:cursor+size].all()); cursor += size
    assert cursor == len(y)
    return dict(count=len(y), correct=int(correct.sum()), accuracy=correct.double().mean().item(),
                loss=F.cross_entropy(scores,y).item(), boundary_accuracy=strata['1']['accuracy'],
                class_accuracy={str(c):correct[y==c].double().mean().item() for c in (0,1)},
                family_count=len(sizes), family_all_correct=family_correct/len(sizes), strata=strata,
                baselines=dict(constant=0.5, first=(x[:,0]==y).double().mean().item(),
                               last=(x[:,-2]==y).double().mean().item(), exact_count=1.0))


def evaluate(model, data, representatives):
    x,y = data
    with torch.no_grad(): scores = torch.cat([forward(model,c)[:,-1] for c in x.split(256)])
    return measure(scores,x,y,representatives), scores


def decision(runs):
    checks = dict(completed=len(runs)==5 and all(r['status']=='trained' and r['step']==6000 for r in runs))
    for n in LENGTHS:
        checks[f'{n}_overall'] = len(runs)==5 and all(r.get('test',{}).get(str(n),{}).get('accuracy',0)>=(.99 if n<15 else .90) for r in runs)
        if n>=15:
            checks[f'{n}_boundary'] = len(runs)==5 and all(r.get('test',{}).get(str(n),{}).get('boundary_accuracy',0)>=.90 for r in runs)
    return dict(pass_transfer=all(checks.values()), checks=checks)


def provenance():
    prior = json.loads((ROOT/'results/mqa_training.json').read_text())
    sources = {name:digest(ROOT/'experiments'/name) for name in prior['provenance'] if name!='protocol'}
    assert all(sources[n]==h for n,h in prior['provenance'].items() if n!='protocol')
    for name in ('majority_transfer.py','verify_majority_transfer.py','audit_majority_transfer.py'):
        sources[name] = digest(ROOT/'experiments'/name)
    sources['protocol'] = digest(ROOT/'docs/majority-transfer-protocol.md')
    return sources


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    ip=ROOT/'results/majority_transfer_data.json'; rp=ROOT/'results/majority_transfer.json'
    directory=ROOT/'results/majority_transfer_checkpoints'; directory.mkdir(exist_ok=True)
    prov=provenance(); verification=json.loads((ROOT/'results/majority_transfer_verification.json').read_text())
    assert verification['passed'] and verification['provenance']==prov
    if rp.exists():
        report=json.loads(rp.read_text()); assert report['provenance']==prov and report['input_sha256']==digest(ip)
        if report['phase']=='complete': print('Already complete.'); return
        payload=json.loads(ip.read_text())
    else:
        assert not ip.exists(); payload=make_data(ip)
        report=dict(phase='training', provenance=prov, input_sha256=digest(ip), torch=torch.__version__, runs=[])
    from audit_majority_transfer import audit_inputs
    report['input_checks']=audit_inputs(payload)
    # Keep test tensors out of the training/validation path.
    data={k:tensors(rows) for k,rows in payload['datasets'].items() if not k.startswith('test_')}
    def save():
        tmp=rp.with_suffix('.tmp'); tmp.write_text(json.dumps(report,indent=2)+'\n'); tmp.replace(rp)
    save()
    for seed in SEEDS:
        run=next((r for r in report['runs'] if r['seed']==seed),None); path=directory/f'{seed}.pt'
        if run and run['status'] in ('trained','failed'):
            assert digest(path)==run['checkpoint_sha256']; continue
        torch.manual_seed(seed); model=model_new(); initial=hash_state(model.state_dict())
        assert sum(p.numel() for p in model.parameters())==4016
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        gs={n:torch.Generator().manual_seed(seed+offset) for n,offset in ((9,1000),(11,2000))}; start=0
        if path.exists():
            state=torch.load(path,weights_only=True); assert state['provenance']==prov
            model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
            for n in (9,11): gs[n].set_state(state[f'g{n}'])
            run=state['run']; start=run['step']; torch.set_rng_state(state['rng'])
            report['runs']=[r for r in report['runs'] if r['seed']!=seed]; report['runs'].append(run)
        else:
            run=dict(seed=seed,initial_sha256=initial,parameters=4016,status='running',step=0,curve=[],
                     tokens=0,attention_entries=0,elapsed_seconds=0.)
            run['curve'].append(dict(step=0,validation={str(n):evaluate(model,data[f'validation_{n}'],payload['families'][f'validation_{n}'])[0] for n in (9,11)}))
            report['runs'].append(run)
        hashes={n:hashlib.sha256() for n in (9,11)}
        replay={n:torch.Generator().manual_seed(seed+offset) for n,offset in ((9,1000),(11,2000))}
        for _ in range(start):
            for n in (9,11): hashes[n].update(json.dumps(torch.randint(len(data[f'train_{n}'][0]),(64,),generator=replay[n]).tolist()).encode())
        tick=time.perf_counter(); losses=[]
        for step in range(start+1,6001):
            draws={n:torch.randint(len(data[f'train_{n}'][0]),(64,),generator=gs[n]) for n in (9,11)}
            for n in (9,11): hashes[n].update(json.dumps(draws[n].tolist()).encode())
            n=9 if step%2 else 11; x,y=data[f'train_{n}']; bx,by=x[draws[n]],y[draws[n]]
            opt.zero_grad(); loss=F.cross_entropy(forward(model,bx)[:,-1],by); finite=bool(torch.isfinite(loss))
            if finite:
                loss.backward(); finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            if finite:
                opt.step(); finite=all(bool(torch.isfinite(p).all()) for p in model.parameters()); losses.append(loss.item())
            if not finite: run.update(status='failed',failure_step=step)
            run['tokens']+=64*(n+1); run['attention_entries']+=64*(n+1)**2
            if step%1000==0 or not finite:
                point=dict(step=step,training_window_mean_loss=sum(losses)/max(1,len(losses)),finite_training=finite,
                           validation={str(n):evaluate(model,data[f'validation_{n}'],payload['families'][f'validation_{n}'])[0] for n in (9,11)})
                run['curve'].append(point); losses=[]
                run.update(step=step,stream9_sha256=hashes[9].hexdigest(),stream11_sha256=hashes[11].hexdigest())
                if step==6000 and finite: run['status']='trained'
                run['elapsed_seconds']+=time.perf_counter()-tick
                state=dict(model=model.state_dict(),optimizer=opt.state_dict(),g9=gs[9].get_state(),g11=gs[11].get_state(),
                           rng=torch.get_rng_state(),run=dict(run),provenance=prov)
                tmp=path.with_suffix('.tmp'); torch.save(state,tmp); tmp.replace(path)
                run.update(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=digest(path)); save(); tick=time.perf_counter()
                print(json.dumps(dict(seed=seed,step=step,validation={k:v['accuracy'] for k,v in point['validation'].items()})),flush=True)
            if not finite: break
    assert len(report['runs'])==5; report['phase']='all_attempts_finished_before_test'; save(); details=[]
    for run in report['runs']:
        if run['status']!='trained': continue
        path=ROOT/run['checkpoint']; assert digest(path)==run['checkpoint_sha256']
        model=model_new().eval(); model.load_state_dict(torch.load(path,weights_only=True)['model'])
        run['test']={}; scores={}
        for n in LENGTHS:
            key=f'test_{n}'; run['test'][str(n)],scores[str(n)]=evaluate(model,tensors(payload['datasets'][key]),payload['families'][key])
        details.append(dict(seed=run['seed'],scores=scores)); save()
    dp=ROOT/'results/majority_transfer_scores.pt'; assert not dp.exists(); torch.save(details,dp)
    report.update(phase='complete',scores_sha256=digest(dp),decision=decision(report['runs'])); save()
    print(json.dumps(report['decision']))


if __name__=='__main__': main()
