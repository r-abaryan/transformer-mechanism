"""Fixed state-tracking study; preserve and reuse the verified MQA forward."""
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from mqa_training import ROOT, model_new as template, forward, hash_state
from position_distractors import digest

SEEDS = (2003, 2011, 2027)
STRATA = ('reset_free', 'long_tail', 'mixed')
STEPS = 3000


def model_new():
    model = template('mqa')
    model.embedding = nn.Embedding(7, 16).double()
    model.output = nn.Linear(16, 2, bias=False).double()
    return model


def answer(start, ops):
    state = start
    for op in ops:
        if op == 1:
            state = 1 - state
        elif op >= 2:
            state = op - 2
    return state


def complement(ops):
    return [5-op if op >= 2 else op for op in ops]


def make_data():
    rng = random.Random(20261001)
    cells = {}
    for n in (8, 12, 16, 24):
        counts = {'train':64, 'validation':32, 'test':32} if n == 8 else (
            {'train':256, 'validation':64, 'test':128} if n == 12 else {'test':256})
        for kind in STRATA:
            seen = set()
            for split, count in counts.items():
                rows = []
                while len(rows) < 2*count:
                    if kind == 'reset_free':
                        ops = [rng.randrange(2) for _ in range(n)]
                    elif kind == 'long_tail':
                        ops = [2]+[rng.randrange(2) for _ in range(n-1)]
                    else:
                        ops = [rng.randrange(4) for _ in range(n)]
                        if sum(o >= 2 for o in ops) < 2 or not any(o >= 2 for o in ops[n//2:]):
                            continue
                    if tuple(ops) in seen:
                        continue
                    seen.add(tuple(ops))
                    for start, sequence in ((0, ops), (1, complement(ops))):
                        rows.append([[4+start]+sequence+[6], answer(start, sequence)])
                cells[f'{split}/{n}/{kind}'] = rows
    return cells


def tensors(rows):
    return torch.tensor([x for x,y in rows]), torch.tensor([y for x,y in rows])


def metrics(logits, y):
    good = logits.argmax(-1) == y
    return dict(count=len(y), correct=int(good.sum()), accuracy=good.double().mean().item(),
                loss=F.cross_entropy(logits,y).item(),
                class_accuracy={str(c):good[y==c].double().mean().item() for c in (0,1)})


def evaluate(model, rows):
    x,y = tensors(rows)
    with torch.no_grad():
        logits = torch.cat([forward(model,b)[:,-1] for b in x.split(128)])
    return metrics(logits,y), logits


def provenance():
    # Transitive producer dependencies were already frozen by the MQA study.
    old = json.loads((ROOT/'results/mqa_training.json').read_text())['provenance']
    paths = ['experiments/'+n for n in old if n != 'protocol']
    for n,h in old.items():
        if n != 'protocol':
            assert digest(ROOT/'experiments'/n) == h
    paths += ['experiments/state_tracking.py','experiments/audit_state_tracking.py',
              'experiments/audit_mqa_training.py','experiments/audit_kv_cache_equivalence.py',
              'docs/protocols/state-tracking.md']
    return {p:digest(ROOT/p) for p in paths}


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rp = ROOT/'results/state_tracking.json'
    assert not rp.exists(), 'Preserve existing attempts; do not silently rerun.'
    from audit_state_tracking import preflight, audit_data
    prov = provenance()
    verification = preflight()
    cells = make_data()
    checks = audit_data(cells)
    ip = ROOT/'results/state_tracking_data.json'
    assert not ip.exists()
    save_json(ip,cells)
    report = dict(phase='training', provenance=prov, input_sha256=digest(ip),
                  torch=torch.__version__, device='cpu', dtype='float64',
                  preflight=verification, data_checks=checks, runs=[])
    save_json(rp,report)
    train = {n:tensors([row for kind in STRATA for row in cells[f'train/{n}/{kind}']]) for n in (8,12)}
    directory = ROOT/'results/state_tracking_checkpoints'
    directory.mkdir(exist_ok=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = model_new()
        opt = torch.optim.Adam(model.parameters(),lr=.001)
        gen = torch.Generator().manual_seed(seed+1000)
        stream = hashlib.sha256()
        run = dict(seed=seed, initial_sha256=hash_state(model.state_dict()),
                   parameters=sum(p.numel() for p in model.parameters()), status='running', step=0, curve=[])
        report['runs'].append(run)
        tick = time.perf_counter()
        for step in range(1,STEPS+1):
            n = 8 if step%2 else 12
            x,y = train[n]
            ix = torch.randint(len(y),(64,),generator=gen)
            stream.update(json.dumps(ix.tolist()).encode())
            opt.zero_grad()
            loss = F.cross_entropy(forward(model,x[ix])[:,-1],y[ix])
            finite = bool(torch.isfinite(loss))
            if finite:
                loss.backward()
                finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            if finite:
                opt.step()
                finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
            run['step'] = step
            if not finite:
                run.update(status='failed',failure_step=step)
                break
            if step%500 == 0:
                val = {k:evaluate(model,rows)[0] for k,rows in cells.items() if k.startswith('validation/')}
                run['curve'].append(dict(step=step,validation=val))
                save_json(rp,report)
                print(json.dumps(dict(seed=seed,step=step,validation={k:v['accuracy'] for k,v in val.items()})),flush=True)
        if run['status'] != 'failed':
            run['status'] = 'trained'
        run.update(elapsed_seconds=time.perf_counter()-tick,stream_sha256=stream.hexdigest())
        cp = directory/f'{seed}.pt'
        assert not cp.exists()
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),generator=gen.get_state(),
                        rng=torch.get_rng_state(),provenance=prov),cp)
        run.update(checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=digest(cp))
        save_json(rp,report)
    report['phase'] = 'attempts_finished_before_test'
    save_json(rp,report)
    scores = {}
    for run in report['runs']:
        if run['status'] != 'trained':
            continue
        model = model_new().eval()
        model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        run['evaluation'] = {}
        scores[run['seed']] = {}
        for key,rows in cells.items():
            if key.startswith('train/'):
                continue
            run['evaluation'][key],scores[run['seed']][key] = evaluate(model,rows)
    checks = {'completed':all(r['status']=='trained' and r['step']==STEPS for r in report['runs'])}
    for n in (8,12,16,24):
        for kind in STRATA:
            key = f'test/{n}/{kind}'
            checks[key] = all(r.get('evaluation',{}).get(key,{}).get('accuracy',0) >= (.99 if n<=12 else .90) for r in report['runs'])
    sp = ROOT/'results/state_tracking_scores.pt'
    assert not sp.exists()
    torch.save(scores,sp)
    report.update(phase='complete',scores_sha256=digest(sp),decision=dict(passed=all(checks.values()),checks=checks))
    save_json(rp,report)
    print(json.dumps(report['decision']),flush=True)


if __name__ == '__main__':
    main()
