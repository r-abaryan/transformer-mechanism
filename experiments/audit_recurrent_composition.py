"""Independent graph walk, model arithmetic and full-pool decision audit."""
import collections
import hashlib
import json

import torch
from torch.nn import functional as F
from audit_recurrent_retrieval import independent
from recurrent_composition import ROOT, SEEDS, ARMS, COUNTS, pair, expand, evaluate, decision
from recurrent_retrieval import write
from position_distractors import digest
from mqa_training import hash_state


def oracle(row, one_only=False):
    edges = dict(zip(row[:-3:2], row[1:-3:2]))
    result = row[-2]
    for _ in range(1 if one_only else row[-1]-12):
        result = edges[result]
    return result


def audit_inputs(payload, data):
    assert payload['generator_seed'] == 2026100201 and payload['four_edge_support'] >= 192
    assert set(payload['families']) == set(COUNTS) == set(data)
    seen = set()
    rows_checked, wrong_labels = 0, 0
    for key, graphs in payload['families'].items():
        n = int(key.rsplit('_', 1)[1])
        assert len(graphs) == COUNTS[key]
        expected = []
        for graph in graphs:
            cycle, order = graph['cycle'], graph['order']
            assert len(cycle) == n and len(set(cycle)) == n and set(order) == set(range(n))
            assert all(0 <= v < 12 for v in cycle)
            edges = {a:b for a, b in zip(cycle, cycle[1:]+cycle[:1])}
            signature = min(tuple(sorted(((a+s)%12, (b+s)%12) for a, b in edges.items())) for s in range(12))
            assert signature not in seen, 'A renamed mapping family crossed pools or was duplicated.'
            seen.add(signature)
            for shift in range(12):
                mapping = {(a+shift)%12:(b+shift)%12 for a, b in edges.items()}
                for ordering in (order, list(reversed(order))):
                    context = []
                    for i in ordering:
                        source = (cycle[i]+shift)%12
                        context.extend([source, mapping[source]])
                    for node in cycle:
                        q = (node+shift)%12
                        for hop in (1, 2):
                            row = context+[12, q, 12+hop]
                            expected.append((row, oracle(row)))
        x, y = data[key]
        assert x.tolist() == [row for row, _ in expected]
        assert y.tolist() == [label for _, label in expected]
        assert len(y) == len(graphs)*48*n and x.shape[1] == 2*n+3
        for hop in (13, 14):
            labels = y[x[:, -1] == hop].tolist()
            assert set(collections.Counter(labels).values()) == {len(labels)//12}
        forward = reverse = 0
        for row, label in expected:
            rows_checked += 1
            assert row[-3] == 12 and oracle(row) == label
            if row[-1] == 14:
                assert oracle(row, one_only=True) != label
                wrong_labels += 1
                sources = row[:-3:2]
                first = sources.index(row[-2])
                second = sources.index(oracle(row, one_only=True))
                forward += int(first < second)
                reverse += int(first > second)
        assert forward == reverse == len(y)//4
    fixture = [7,9,0,2,9,0,2,7,12,0,14]
    assert oracle(fixture) == 7 and oracle(fixture, one_only=True) == 2
    assert wrong_labels == rows_checked//2 > 0
    return dict(passed=True, graph_families=len(seen), rows_checked=rows_checked,
                intentional_one_step_label_errors=wrong_labels, fixture_two_step=7)


def baselines(data):
    result = {}
    for key, (x, y) in data.items():
        rows = x.tolist()
        outputs = {'constant0':torch.zeros_like(y), 'query':x[:, -2], 'final_destination':x[:, -4],
                   'one_step_only':torch.tensor([oracle(row, one_only=True) for row in rows]),
                   'oracle':torch.tensor([oracle(row) for row in rows])}
        assert torch.equal(outputs['oracle'], y)
        result[key] = {}
        for label, h in (('one_step', 13), ('two_step', 14)):
            mask = x[:, -1] == h
            result[key][label] = {name:(pred[mask] == y[mask]).double().mean().item() for name, pred in outputs.items()}
            result[key][label]['uniform_expected'] = 1/12
    return result


def preflight(data):
    checks = []
    for seed in SEEDS:
        models = pair(seed)
        assert torch.equal(models['transformer'].embedding.weight, models['gru'].embedding.weight)
        for arm, model in models.items():
            x, y = [t[:9] for t in data['train_6']]
            native = model(x)
            other = independent(model, x)
            error = (native-other).abs().max().item()
            a = torch.autograd.grad(F.cross_entropy(native[:, -1], y), tuple(model.parameters()))
            b = torch.autograd.grad(F.cross_entropy(other[:, -1], y), tuple(model.parameters()))
            grad_error = max((u-v).abs().max().item() for u, v in zip(a, b))
            with torch.no_grad():
                prefix_error = (model(x[:, :7])-native[:, :7]).abs().max().item()
                batch_error = (model(x[:1])-native[:1]).abs().max().item()
                changed = x.clone()
                changed[:, 7:] = (changed[:, 7:]+1)%15
                future_error = (model(changed)[:, :7]-native[:, :7]).abs().max().item()
                repeat_error = (model(x)-native).abs().max().item()
                negative = (independent(model, x, reset=True)-native).abs().max().item() if arm == 'gru' else None
                if arm == 'gru':
                    assert negative > 1e-6
            assert max(error, grad_error, prefix_error, batch_error, future_error, repeat_error) < 1e-10
            checks.append(dict(seed=seed, arm=arm, logit_error=error, gradient_error=grad_error,
                               prefix_error=prefix_error, batch_error=batch_error, future_error=future_error,
                               repeat_error=repeat_error, intentional_reset_error=negative))
    return dict(passed=True, models=checks)


def independent_metrics(scores, x, y):
    predictions = scores.argmax(-1).tolist()
    loss = F.cross_entropy(scores, y, reduction='none').tolist()
    groups = {name:[] for name in ('one_step', 'two_step', 'two_forward', 'two_reverse')}
    for i, row in enumerate(x.tolist()):
        if row[-1] == 13:
            groups['one_step'].append(i)
        else:
            groups['two_step'].append(i)
            sources = row[:-3:2]
            first = sources.index(row[-2])
            second = sources.index(oracle(row, one_only=True))
            groups['two_forward' if first < second else 'two_reverse'].append(i)
    targets = y.tolist()
    result = dict(accuracy=sum(p == t for p, t in zip(predictions, targets))/len(y), loss=sum(loss)/len(y))
    for key, indices in groups.items():
        result[key] = dict(count=len(indices), accuracy=sum(predictions[i] == targets[i] for i in indices)/len(indices),
                           loss=sum(loss[i] for i in indices)/len(indices))
    return result


def difference(a, b):
    assert a.keys() == b.keys()
    return max(difference(a[k], b[k]) if isinstance(a[k], dict) else abs(a[k]-b[k]) for k in a)


def independent_decision(runs):
    fit, reliable, gains = {}, {}, []
    for arm in ARMS:
        rows = [r for r in runs if r['arm'] == arm]
        fit[arm] = len(rows) == 3
        reliable[arm] = True
        for r in rows:
            if r['status'] != 'trained':
                fit[arm] = reliable[arm] = False
                continue
            for split in ('train', 'validation'):
                for n in (4, 6):
                    for h in ('one_step', 'two_step'):
                        fit[arm] &= r['metrics'][f'{split}_{n}'][h]['accuracy'] >= .99
            for n in (4, 6, 8):
                for h in ('one_step', 'two_step', 'two_forward', 'two_reverse'):
                    reliable[arm] &= r['metrics'][f'test_{n}'][h]['accuracy'] >= (.90 if n == 8 else .99)
        reliable[arm] &= fit[arm]
    for seed in SEEDS:
        rows = {r['arm']:r for r in runs if r['seed'] == seed}
        if any(r['status'] != 'trained' for r in rows.values()):
            return dict(short_fit=fit, reliability=reliable, consistent_advantage=None, pairs=gains)
        means = {a:sum(rows[a]['metrics'][f'validation_{n}']['two_step']['accuracy'] for n in (4, 6))/2 for a in ARMS}
        gains.append(dict(seed=seed, **means, gru_minus_transformer=means['gru']-means['transformer']))
    winner = None
    if min(g['gru_minus_transformer'] for g in gains) >= .10:
        winner = 'gru'
    if max(g['gru_minus_transformer'] for g in gains) <= -.10:
        winner = 'transformer'
    return dict(short_fit=fit, reliability=reliable, consistent_advantage=winner, pairs=gains)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rp = ROOT/'results/recurrent_composition.json'
    report = json.loads(rp.read_text())
    assert report['phase'] == 'complete'
    assert torch.__version__ == report['torch']
    for path, expected in {**report['provenance'], **report['historical']}.items():
        assert digest(ROOT/path) == expected, path
    ip = ROOT/'results/recurrent_composition_data.json'
    assert digest(ip) == report['input_sha256']
    payload = json.loads(ip.read_text())
    data = expand(payload)
    checks = audit_inputs(payload, data)
    assert baselines(data) == report['baselines']
    pretest = json.loads((ROOT/'results/recurrent_composition_pretest.json').read_text())
    assert pretest['phase'] == 'all_attempts_finished_before_test'
    assert {(r['seed'], r['arm']) for r in report['runs']} == {(s, a) for s in SEEDS for a in ARMS}
    cells, streams = [], []
    for seed in SEEDS:
        paired = [r for r in report['runs'] if r['seed'] == seed]
        assert len({r['embedding_sha256'] for r in paired}) == 1
        generator = torch.Generator().manual_seed(seed+1000)
        stream = hashlib.sha256()
        groups = {n:[torch.where(data[f'train_{n}'][0][:, -1] == h)[0] for h in (13, 14)] for n in (4, 6)}
        for step in range(1, 6001):
            n = 4 if step % 2 else 6
            idx = torch.cat([g[torch.randint(len(g), (32,), generator=generator)] for g in groups[n]])
            stream.update(json.dumps(idx.tolist()).encode())
        streams.append(dict(seed=seed, stream_sha256=stream.hexdigest()))
        for run in paired:
            model = pair(seed)[run['arm']]
            assert hash_state(model.state_dict()) == run['initial_sha256']
            assert sum(p.numel() for p in model.parameters()) == run['parameters']
            cp = ROOT/run['checkpoint']
            assert digest(cp) == run['checkpoint_sha256']
            earlier = next(r for r in pretest['runs'] if r['seed'] == seed and r['arm'] == run['arm'])
            assert earlier['checkpoint_sha256'] == run['checkpoint_sha256']
            assert not any(key.startswith('test_') for key in earlier['metrics'])
            if run['status'] != 'trained':
                continue
            assert stream.hexdigest() == run['stream_sha256']
            state = torch.load(cp, weights_only=True)
            assert state['step'] == 6000 and all(v['step'].item() == 6000 for v in state['optimizer']['state'].values())
            model.load_state_dict(state['model'])
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
            sp = ROOT/run['scores']
            assert digest(sp) == run['scores_sha256']
            saved = torch.load(sp, weights_only=True)
            for key, (x, y) in data.items():
                _, scores = evaluate(model, (x, y), independent)
                error = (scores-saved[key]).abs().max().item()
                changes = int((scores.argmax(-1) != saved[key].argmax(-1)).sum())
                actual = independent_metrics(scores, x, y)
                metric_error = difference(actual, run['metrics'][key])
                assert error < 1e-10 and changes == 0 and metric_error < 1e-10
                if key in earlier['metrics']:
                    assert earlier['metrics'][key] == run['metrics'][key]
                cells.append(dict(seed=seed, arm=run['arm'], split=key, predictions=len(y),
                                  logit_error=error, metric_error=metric_error, argmax_changes=changes))
            print(json.dumps(dict(audited_seed=seed, arm=run['arm'])), flush=True)
    derived = independent_decision(report['runs'])
    assert derived == report['decision'] == decision(report['runs'])
    mutated = dict(report['decision'], consistent_advantage='invalid')
    assert mutated != derived
    result = dict(passed=True, report_sha256=digest(rp), input_checks=checks, cells=cells,
                  streams=streams, decision=derived, intentional_wrong_decision_detected=True,
                  evaluations=len(cells), predictions=sum(c['predictions'] for c in cells),
                  max_logit_error=max(c['logit_error'] for c in cells), max_metric_error=max(c['metric_error'] for c in cells))
    write(ROOT/'results/recurrent_composition_audit.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cells', 'input_checks', 'streams', 'decision')}), flush=True)


if __name__ == '__main__':
    main()
