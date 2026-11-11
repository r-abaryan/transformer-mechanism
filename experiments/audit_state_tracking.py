"""Independent state rule, split audit and SDPA forward verification."""
import hashlib
import itertools
import json

import torch
from torch.nn import functional as F
from audit_mqa_training import independent
from state_tracking import ROOT, SEEDS, STRATA, STEPS, model_new, forward, answer, tensors, provenance, save_json
from mqa_training import hash_state
from position_distractors import digest


def oracle(ids):
    ops = ids[1:-1]
    resets = [i for i,o in enumerate(ops) if o in (2,3)]
    last = resets[-1] if resets else -1
    base = ops[last]-2 if resets else ids[0]-4
    return base ^ (ops[last+1:].count(1)%2)


def preflight():
    checked = omitted_flip_errors = 0
    for n in range(8):
        for ops in itertools.product(range(4),repeat=n):
            for initial in (0,1):
                ids = [4+initial]+list(ops)+[6]
                truth = oracle(ids)
                assert answer(initial,ops) == truth
                a,b = 1,0
                for op in ops:
                    c,d = ((1,0),(1,1),(0,0),(0,1))[op]
                    a,b = c*a,(c*b)^d
                assert ((a*initial)^b) == truth
                broken = answer(initial,[o for o in ops if o != 1])
                omitted_flip_errors += int(broken != truth)
                checked += 1
    assert omitted_flip_errors > 0
    ids = torch.tensor([[4,2,0,1,1,3,0,1,0,6],[5,3,0,1,1,2,0,1,0,6]])
    labels = torch.tensor([oracle(row) for row in ids.tolist()])
    records = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = model_new()
        a,b = forward(model,ids),independent(model,ids)
        error = (a-b).abs().max().item()
        assert error <= 1e-10
        ga = torch.autograd.grad(F.cross_entropy(a[:,-1],labels),list(model.parameters()))
        gb = torch.autograd.grad(F.cross_entropy(b[:,-1],labels),list(model.parameters()))
        grad_error = max((x-y).abs().max().item() for x,y in zip(ga,gb))
        assert grad_error <= 1e-10 and all(torch.isfinite(g).all() for g in ga)
        assert max(g.abs().max().item() for g in ga) > 1e-7
        changed = ids.clone()
        changed[:,7] = 0
        with torch.no_grad():
            c = forward(model,changed)
        prefix = (a[:,:7]-c[:,:7]).abs().max().item()
        suffix = (a[:,7:]-c[:,7:]).abs().max().item()
        assert prefix <= 1e-10 and suffix > 1e-6
        records.append(dict(seed=seed,logit_error=error,gradient_error=grad_error,causal_prefix_error=prefix))
    return dict(passed=True,exhaustive_cases=checked,omitted_flip_errors=omitted_flip_errors,models=records)


def audit_data(cells):
    expected = {}
    for n in (8,12,16,24):
        sizes = {'train':128,'validation':64,'test':64} if n == 8 else (
            {'train':512,'validation':128,'test':256} if n == 12 else {'test':512})
        for split,size in sizes.items():
            for kind in STRATA:
                expected[f'{split}/{n}/{kind}'] = size
    assert set(cells) == set(expected)
    seen = set()
    stats = {}
    for key,rows in cells.items():
        split,length,kind = key.split('/')
        n = int(length)
        assert len(rows) == expected[key]
        ignore_correct = 0
        for i,(ids,y) in enumerate(rows):
            assert len(ids)==n+2 and ids[0] in (4,5) and ids[-1]==6
            ops = ids[1:-1]
            assert set(ops) <= {0,1,2,3} and y == oracle(ids)
            assert tuple(ids) not in seen
            seen.add(tuple(ids))
            if kind == 'reset_free':
                assert all(o<2 for o in ops)
            elif kind == 'long_tail':
                assert ops[0] in (2,3) and all(o<2 for o in ops[1:])
            else:
                assert sum(o>=2 for o in ops)>=2 and any(o>=2 for o in ops[n//2:])
            if i%2:
                prev,py = rows[i-1]
                assert ids[0] == 5 and prev[0] == 4 and y == 1-py
                assert ops == [{2:3,3:2}.get(o,o) for o in prev[1:-1]]
            resets = [o-2 for o in ops if o>=2]
            guess = resets[-1] if resets else ids[0]-4
            ignore_correct += int(guess==y)
        assert sum(y for _,y in rows)*2 == len(rows)
        stats[key] = dict(rows=len(rows),constant_accuracy=.5,ignore_flip_accuracy=ignore_correct/len(rows))
    return stats


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rp = ROOT/'results/state_tracking.json'
    report = json.loads(rp.read_text())
    assert report['phase']=='complete' and report['provenance']==provenance()
    ip = ROOT/'results/state_tracking_data.json'
    assert digest(ip)==report['input_sha256']
    cells = json.loads(ip.read_text())
    assert audit_data(cells)==report['data_checks']
    sp = ROOT/'results/state_tracking_scores.pt'
    assert digest(sp)==report['scores_sha256']
    scores = torch.load(sp,weights_only=True)
    assert [r['seed'] for r in report['runs']]==list(SEEDS)
    checks = {'completed':True}
    records = []
    max_error = max_loss_error = 0.
    for r in report['runs']:
        torch.manual_seed(r['seed'])
        model = model_new()
        assert hash_state(model.state_dict())==r['initial_sha256']
        assert sum(p.numel() for p in model.parameters())==r['parameters']
        cp = ROOT/r['checkpoint']
        assert digest(cp)==r['checkpoint_sha256']
        state = torch.load(cp,weights_only=True)
        assert state['provenance']==report['provenance']
        gen = torch.Generator().manual_seed(r['seed']+1000)
        stream = hashlib.sha256()
        for step in range(1,r['step']+1):
            n = 8 if step%2 else 12
            count = sum(len(cells[f'train/{n}/{kind}']) for kind in STRATA)
            ix = torch.randint(count,(64,),generator=gen)
            stream.update(json.dumps(ix.tolist()).encode())
        assert stream.hexdigest()==r['stream_sha256'] and torch.equal(gen.get_state(),state['generator'])
        if r['status']!='trained':
            checks['completed'] = False
            continue
        assert r['step']==STEPS and all(s['step'].item()==STEPS for s in state['optimizer']['state'].values())
        model.load_state_dict(state['model'])
        for key,rows in cells.items():
            if key.startswith('train/'):
                continue
            x,_ = tensors(rows)
            y = torch.tensor([oracle(row) for row,_ in rows])
            with torch.no_grad():
                logits = torch.cat([independent(model,b)[:,-1] for b in x.split(128)])
            error = (logits-scores[r['seed']][key]).abs().max().item()
            assert error<=1e-10
            max_error = max(max_error,error)
            assert torch.equal(logits.argmax(-1),scores[r['seed']][key].argmax(-1))
            good = logits.argmax(-1)==y
            m = r['evaluation'][key]
            assert len(y)==m['count'] and int(good.sum())==m['correct']
            acc = int(good.sum())/len(y)
            assert acc==m['accuracy']
            for c in (0,1):
                assert int(good[y==c].sum())/int((y==c).sum())==m['class_accuracy'][str(c)]
            loss = (logits.logsumexp(-1)-logits[torch.arange(len(y)),y]).mean().item()
            loss_error = abs(loss-m['loss'])
            assert loss_error<=1e-10
            max_loss_error=max(max_loss_error,loss_error)
            if key.startswith('test/'):
                n = int(key.split('/')[1])
                checks[key] = checks.get(key,True) and acc >= (.99 if n<=12 else .90)
            records.append(dict(seed=r['seed'],cell=key,count=len(y),accuracy=acc))
    for n in (8,12,16,24):
        for kind in STRATA:
            key = f'test/{n}/{kind}'
            if not checks['completed']:
                checks[key]=False
    assert checks==report['decision']['checks'] and all(checks.values())==report['decision']['passed']
    result = dict(passed=True,report_sha256=digest(rp),evaluations=len(records),
                  predictions_checked=sum(c['count'] for c in records),max_logit_error=max_error,
                  max_loss_error=max_loss_error,argmax_differences=0,cells=records)
    ap = ROOT/'results/state_tracking_audit.json'
    assert not ap.exists()
    save_json(ap,result)
    print(json.dumps({k:v for k,v in result.items() if k!='cells'}))


if __name__=='__main__':
    main()
