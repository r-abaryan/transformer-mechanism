"""Frozen paired comparison on shuffled one/two-step directed-cycle lookup."""
import hashlib
import itertools
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from mqa_training import model_new, hash_state
from position_distractors import digest
from recurrent_retrieval import write

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2237, 2243, 2251)
ARMS = ('transformer', 'gru')
COUNTS = {'train_4':128, 'validation_4':32, 'test_4':32,
          'train_6':128, 'validation_6':32, 'test_6':32, 'test_8':32}


def family(edges):
    return min(tuple(sorted(((a+s)%12, (b+s)%12) for a, b in edges)) for s in range(12))


def make_data():
    rng = random.Random(2026100201)
    support = set()
    for nodes in itertools.combinations(range(12), 4):
        for rest in itertools.permutations(nodes[1:]):
            cycle = (nodes[0],)+rest
            support.add(family(list(zip(cycle, cycle[1:]+cycle[:1]))))
    assert len(support) >= 192
    available = sorted(support)
    rng.shuffle(available)
    used = set()
    pools = {}
    for key, count in COUNTS.items():
        n = int(key.rsplit('_', 1)[1])
        graphs = []
        for _ in range(count):
            if n == 4:
                edges = list(available.pop())
                start = edges[0][0]
                cycle = [start]
                lookup = dict(edges)
                for _ in range(3):
                    cycle.append(lookup[cycle[-1]])
            else:
                for attempt in range(100000):
                    cycle = rng.sample(range(12), n)
                    edges = list(zip(cycle, cycle[1:]+cycle[:1]))
                    if family(edges) not in used:
                        break
                else:
                    raise RuntimeError('Family support exhausted')
            edges = list(zip(cycle, cycle[1:]+cycle[:1]))
            signature = family(edges)
            assert signature not in used
            used.add(signature)
            order = rng.sample(range(n), n)
            graphs.append(dict(cycle=cycle, order=order))
        pools[key] = graphs
    return dict(generator_seed=2026100201, four_edge_support=len(support), families=pools)


def expand(payload):
    data = {}
    for key, graphs in payload['families'].items():
        rows, labels = [], []
        for graph in graphs:
            cycle, order = graph['cycle'], graph['order']
            n = len(cycle)
            for shift in range(12):
                renamed = [(v+shift)%12 for v in cycle]
                for ordering in (order, order[::-1]):
                    context = [v for i in ordering for v in (renamed[i], renamed[(i+1)%n])]
                    for j, q in enumerate(renamed):
                        for hops in (1, 2):
                            rows.append(context+[12, q, 12+hops])
                            labels.append(renamed[(j+hops)%n])
        data[key] = (torch.tensor(rows), torch.tensor(labels))
    return data


class CompositionGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(15, 16)
        self.gru = nn.GRU(16, 28, batch_first=True)
        self.output = nn.Linear(28, 15, bias=False)
        self.double()

    def forward(self, ids):
        states, _ = self.gru(self.embedding(ids))
        return self.output(states)


def pair(seed):
    torch.manual_seed(seed)
    transformer = model_new('mqa')
    transformer.embedding = nn.Embedding(15, 16).double()
    transformer.output = nn.Linear(16, 15, bias=False).double()
    torch.manual_seed(seed)
    gru = CompositionGRU()
    with torch.no_grad():
        gru.embedding.weight.copy_(transformer.embedding.weight)
    assert [sum(p.numel() for p in m.parameters()) for m in (transformer, gru)] == [4416, 4524]
    return dict(transformer=transformer, gru=gru)


def metrics(scores, x, y):
    correct = scores.argmax(-1) == y
    losses = scores.logsumexp(-1)-scores.gather(1, y[:, None]).squeeze(1)
    result = dict(accuracy=correct.double().mean().item(), loss=losses.mean().item())
    first = (x[:, :-3:2] == x[:, -2: -1]).long().argmax(1)
    intermediate = x[:, 1:-3:2].gather(1, first[:, None])
    second = (x[:, :-3:2] == intermediate).long().argmax(1)
    masks = {'one_step':x[:, -1] == 13, 'two_step':x[:, -1] == 14,
             'two_forward':(x[:, -1] == 14) & (first < second),
             'two_reverse':(x[:, -1] == 14) & (first > second)}
    for name, mask in masks.items():
        assert bool(mask.any())
        result[name] = dict(count=int(mask.sum()), accuracy=correct[mask].double().mean().item(),
                            loss=losses[mask].mean().item())
    return result


def evaluate(model, data, forward=None):
    x, y = data
    with torch.no_grad():
        scores = torch.cat([(model(c) if forward is None else forward(model, c))[:, -1] for c in x.split(256)])
    return metrics(scores, x, y), scores


def decision(runs):
    fit, reliable, gains = {}, {}, []
    for arm in ARMS:
        rows = [r for r in runs if r['arm'] == arm]
        fit[arm] = len(rows) == 3 and all(r['status'] == 'trained' and all(
            r['metrics'][f'{split}_{n}'][h]['accuracy'] >= .99
            for split in ('train', 'validation') for n in (4, 6)
            for h in ('one_step', 'two_step')) for r in rows)
        reliable[arm] = fit[arm] and all(all(
            r['metrics'][f'test_{n}'][h]['accuracy'] >= (.90 if n == 8 else .99)
            for n in (4, 6, 8) for h in ('one_step', 'two_step', 'two_forward', 'two_reverse')) for r in rows)
    for seed in SEEDS:
        rows = {r['arm']:r for r in runs if r['seed'] == seed}
        if any(rows[a]['status'] != 'trained' for a in ARMS):
            return dict(short_fit=fit, reliability=reliable, consistent_advantage=None, pairs=gains)
        means = {a:sum(rows[a]['metrics'][f'validation_{n}']['two_step']['accuracy'] for n in (4, 6))/2 for a in ARMS}
        gains.append(dict(seed=seed, **means, gru_minus_transformer=means['gru']-means['transformer']))
    winner = 'gru' if all(g['gru_minus_transformer'] >= .10 for g in gains) else 'transformer' if all(g['gru_minus_transformer'] <= -.10 for g in gains) else None
    return dict(short_fit=fit, reliability=reliable, consistent_advantage=winner, pairs=gains)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rp = ROOT/'results/recurrent_composition.json'
    ip = ROOT/'results/recurrent_composition_data.json'
    assert not rp.exists() and not ip.exists(), 'Preserve attempts; no automatic restart.'
    payload = make_data()
    write(ip, payload)
    data = expand(payload)
    from audit_recurrent_composition import audit_inputs, baselines, preflight
    provenance = {str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'experiments').glob('*.py')}
    provenance['docs/protocols/recurrent-composition.md'] = digest(ROOT/'docs/protocols/recurrent-composition.md')
    historical = {str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'results').glob('recurrent_*.json') if 'composition' not in p.name}
    report = dict(phase='preflight', torch=torch.__version__, provenance=provenance, historical=historical,
                  input_sha256=digest(ip), input_checks=audit_inputs(payload, data),
                  preflight=preflight(data), baselines=baselines(data), runs=[])
    write(rp, report)
    directory = ROOT/'results/recurrent_composition_checkpoints'
    directory.mkdir(exist_ok=True)
    groups = {n:[torch.where(data[f'train_{n}'][0][:, -1] == h)[0] for h in (13, 14)] for n in (4, 6)}
    for index, seed in enumerate(SEEDS):
        models = pair(seed)
        for arm in ARMS[::1 if index % 2 == 0 else -1]:
            model = models[arm]
            run = dict(seed=seed, arm=arm, parameters=sum(p.numel() for p in model.parameters()),
                       initial_sha256=hash_state(model.state_dict()),
                       embedding_sha256=hash_state({'embedding':model.embedding.weight}),
                       status='running', curve=[], metrics={})
            report['runs'].append(run)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            generator = torch.Generator().manual_seed(seed+1000)
            stream = hashlib.sha256()
            tick = time.perf_counter()
            for step in range(1, 6001):
                n = 4 if step % 2 else 6
                x, y = data[f'train_{n}']
                idx = torch.cat([g[torch.randint(len(g), (32,), generator=generator)] for g in groups[n]])
                stream.update(json.dumps(idx.tolist()).encode())
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x[idx])[:, -1], y[idx])
                finite = bool(torch.isfinite(loss))
                if finite:
                    loss.backward()
                    finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                if not finite:
                    run.update(status='failed', failure_step=step)
                    break
                optimizer.step()
                if step % 1000 == 0:
                    row = dict(step=step, minibatch_loss=loss.item())
                    if step in (5000, 6000):
                        row['pools'] = {key:evaluate(model, pool)[0] for key, pool in data.items() if not key.startswith('test_')}
                    run['curve'].append(row)
                    print(json.dumps(dict(seed=seed, arm=arm, step=step, minibatch_loss=loss.item())), flush=True)
            if run['status'] != 'failed':
                run['status'] = 'trained'
                run['metrics'] = run['curve'][-1]['pools'].copy()
            cp = directory/f'{seed}_{arm}.pt'
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step), cp)
            run.update(step=step, elapsed_seconds=time.perf_counter()-tick, stream_sha256=stream.hexdigest(),
                       checkpoint=str(cp.relative_to(ROOT)), checkpoint_sha256=digest(cp))
            report['phase'] = 'training'
            write(rp, report)
    report['phase'] = 'all_attempts_finished_before_test'
    write(ROOT/'results/recurrent_composition_pretest.json', report)
    for run in report['runs']:
        if run['status'] != 'trained':
            continue
        model = pair(run['seed'])[run['arm']]
        model.load_state_dict(torch.load(ROOT/run['checkpoint'], weights_only=True)['model'])
        scores = {}
        for key, pool in data.items():
            metric, scores[key] = evaluate(model, pool)
            if key in run['metrics']:
                assert metric == run['metrics'][key]
            run['metrics'][key] = metric
        sp = directory/f"{run['seed']}_{run['arm']}_scores.pt"
        torch.save(scores, sp)
        run.update(scores=str(sp.relative_to(ROOT)), scores_sha256=digest(sp))
        write(rp, report)
    report.update(phase='complete', decision=decision(report['runs']))
    write(rp, report)
    print(json.dumps(report['decision']), flush=True)


if __name__ == '__main__':
    main()
