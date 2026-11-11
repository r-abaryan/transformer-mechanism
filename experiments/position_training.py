"""Matched fixed versus random-offset training; no attention intervention."""
import hashlib
import itertools
import json
import random
import statistics
from pathlib import Path
import torch
from torch.nn import functional as F
from reserved_cycle import model_new, evaluate
from position_distractors import pe, digest, logits
from activation_paths import forward

ROOT = Path(__file__).resolve().parents[1]
SHIFTS = (0, 1, 2, 3, 4, 6, 8, 12)


def shifted(model, x, shift):
    delta = pe(torch.arange(x.shape[1]) + shift, 16) - pe(torch.arange(x.shape[1]), 16)
    handle = model.embedding.register_forward_hook(lambda module, args, out: out + delta)
    try:
        return model(x)[:, -1]
    finally:
        handle.remove()


def mapping(ids):
    return tuple(sorted(zip(ids[:6:2], ids[1:6:2])))


def weight_hash(model):
    return hashlib.sha256(json.dumps({k: v.tolist() for k, v in model.state_dict().items()}, sort_keys=True).encode()).hexdigest()


def metrics(scores, y):
    correct = scores.argmax(-1) == y
    return dict(accuracy=correct.double().mean().item(), loss=F.cross_entropy(scores, y).item(),
                every_order_query_correct=correct.reshape(-1, 18).all(1).double().mean().item())


def criteria(runs):
    pairs = [(next(r for r in runs if r['seed'] == seed and r['group'] == 'fixed'),
              next(r for r in runs if r['seed'] == seed and r['group'] == 'varied')) for seed in sorted({r['seed'] for r in runs})]
    outside = {}
    for s in (6, 8, 12):
        gain = statistics.mean(v['test'][str(s)]['accuracy'] - f['test'][str(s)]['accuracy'] for f, v in pairs)
        outside[str(s)] = dict(all_90=all(v['test'][str(s)]['accuracy'] >= .9 for f, v in pairs), mean_gain=gain, gain_5=gain >= .05)
    return dict(training_range_all_90=all(v['test'][str(s)]['accuracy'] >= .9 for f, v in pairs for s in range(5)),
                zero_preserved=all(f['test']['0']['accuracy'] - v['test']['0']['accuracy'] <= .01 for f, v in pairs),
                outside=outside, outside_conjunction=all(c['all_90'] and c['gain_5'] for c in outside.values()))


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = ROOT/'results/position_training.json'
    splitpath = ROOT/'results/position_training_splits.json'
    assert not out.exists() and not splitpath.exists()
    oldpath = ROOT/'results/reserved_cycle_splits.json'
    old = json.loads(oldpath.read_text())
    excluded = {mapping(x) for rows in old.values() for x, y in rows}
    rng = random.Random(20261221)
    maps = {n: [] for n in ('train', 'validation', 'test')}
    for values in itertools.combinations(range(12, 18), 3):
        candidates = [tuple(zip(keys, vs)) for keys in itertools.combinations(range(12), 3)
                      for vs in itertools.permutations(values) if tuple(zip(keys, vs)) not in excluded]
        rng.shuffle(candidates)
        for name, a, b in [('train', 0, 80), ('validation', 80, 100), ('test', 100, 120)]:
            maps[name].extend(candidates[a:b])
    splits = {name: [([t for pair in order for t in pair] + [18, q], target)
                    for m in ms for order in itertools.permutations(m) for q, target in m] for name, ms in maps.items()}
    assert len(set.union(*(set(ms) for ms in maps.values()))) == 2400
    assert not set.union(*(set(ms) for ms in maps.values())) & excluded
    for rows in splits.values():
        assert all(dict(mapping(x))[x[-1]] == y for x, y in rows)
        assert all(sum(y == v for x, y in rows) == len(rows)//6 for v in range(12, 18))
        assert all(sum(x[:6:2].index(x[-1]) == rank for x, y in rows) == len(rows)//3 for rank in range(3))
    splitpath.write_text(json.dumps(splits, separators=(',', ':'))+'\n')
    data = {n: (torch.tensor([x for x, y in rows]), torch.tensor([y for x, y in rows])) for n, rows in splits.items()}
    # Check differentiable hook against explicit attention execution before training.
    torch.manual_seed(461)
    check = model_new()
    x, y = (t[:8] for t in data['train'])
    a = shifted(check, x, 4)
    F.cross_entropy(a, y).backward()
    grads = [p.grad.clone() for p in check.parameters()]
    check.zero_grad()
    b, _ = forward(check, x, 4)
    F.cross_entropy(b, y).backward()
    forward_error = (a-b).abs().max().item()
    gradient_error = max((g-p.grad).abs().max().item() for g, p in zip(grads, check.parameters()))
    assert max(forward_error, gradient_error) < 1e-10
    report = dict(phase='training', torch=torch.__version__, device='cpu', dtype='float64',
                  split_sha256=digest(splitpath), excluded_split_sha256=digest(oldpath),
                  protocol_sha256=digest(ROOT/'docs/position-training-protocol.md'),
                  source_sha256={n: digest(ROOT/'experiments'/n) for n in ('position_training.py', 'reserved_cycle.py', 'compare_depth.py', 'position_distractors.py', 'activation_paths.py')},
                  forward_error=forward_error, gradient_error=gradient_error, runs=[])
    def save():
        out.write_text(json.dumps(report, indent=2)+'\n')
    directory = ROOT/'results/position_training_checkpoints'
    directory.mkdir(exist_ok=True)
    save()
    for seed in (461, 487, 509, 541, 571):
        for group in ('fixed', 'varied'):
            torch.manual_seed(seed)
            model = model_new()
            assert sum(p.numel() for p in model.parameters()) == 5088
            initial = weight_hash(model)
            batches = torch.Generator().manual_seed(seed+1000)
            offsets = random.Random(seed+2000)
            stream = hashlib.sha256()
            counts = [0]*5
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)
            curve = [dict(step=0, validation=evaluate(model, data['validation']))]
            x, y = data['train']
            for step in range(1, 3001):
                idx = torch.randint(len(x), (64,), generator=batches)
                stream.update(json.dumps(idx.tolist()).encode())
                shift = offsets.randrange(5) if group == 'varied' else 0
                counts[shift] += 1
                optimizer.zero_grad()
                loss = F.cross_entropy(shifted(model, x[idx], shift), y[idx])
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step()
                if step % 500 == 0:
                    m = evaluate(model, data['validation'])
                    curve.append(dict(step=step, validation=m))
                    print(json.dumps(dict(seed=seed, group=group, step=step, validation=m)), flush=True)
            path = directory/f'{group}_{seed}.pt'
            assert not path.exists()
            torch.save(model.state_dict(), path)
            run = dict(seed=seed, group=group, initial_sha256=initial, minibatch_sha256=stream.hexdigest(), offset_counts=counts,
                       steps=3000, examples_processed=192000, curve=curve, training=evaluate(model, data['train']),
                       checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=digest(path))
            report['runs'].append(run)
            if group == 'varied':
                baseline = report['runs'][-2]
                assert all(run[k] == baseline[k] for k in ('initial_sha256', 'minibatch_sha256'))
            save()
    report['phase'] = 'all_checkpoints_saved_before_test'
    save()
    x, y = data['test']
    for run in report['runs']:
        model = model_new().eval()
        path = ROOT/run['checkpoint']
        assert digest(path) == run['checkpoint_sha256']
        model.load_state_dict(torch.load(path, weights_only=True))
        run['test'] = {str(s): metrics(logits(model, x, (torch.arange(8)+s).expand_as(x)), y) for s in SHIFTS}
        print(json.dumps(dict(seed=run['seed'], group=run['group'], test=run['test'])), flush=True)
        save()
    report['criteria'] = criteria(report['runs'])
    report['phase'] = 'complete'
    save()
    print(json.dumps(report['criteria']), flush=True)


if __name__ == '__main__':
    main()
