"""Saved-reader query-write order diagnostic, with an independent trace audit."""
import argparse
import csv
import importlib.util
import itertools
import json
import unittest

import torch
import history_repetition as prior

study = prior.study
ROOT = study.ROOT
OUT = ROOT / 'results/query_write_order'
VARIANTS = ('native', 'first', 'middle', 'last')
POSITIONS = dict(first=0, middle=15, last=31)


def dependencies():
    paths = ('experiments/query_write_order.py', 'tests/test_query_write_order.py',
             'docs/protocols/query-write-order.md', *prior.dependencies())
    return {p: study.parent.digest(ROOT / p) for p in paths}


def validate(row):
    events = row['events']
    assert len(events) == 32 and len({k for k, _ in events}) == 32, 'Expected 32 unique keys'
    assert dict(events)[row['query']] == row['label'], 'Wrong query label'


def rearrange(rows, variant):
    assert variant in VARIANTS, 'Undeclared order variant'
    result, positions = [], []
    for row in rows:
        validate(row)
        events = [list(x) for x in row['events']]
        index = next(i for i, (k, _) in enumerate(events) if k == row['query'])
        if variant != 'native':
            event = events.pop(index)
            index = POSITIONS[variant]
            events.insert(index, event)
        assert sorted(events) == sorted(row['events'])
        assert [e for e in events if e[0] != row['query']] == [e for e in row['events'] if e[0] != row['query']]
        result.append(dict(row, events=events)); positions.append(index)
    return result, positions


def evaluate(model, rows, variant):
    changed, positions = rearrange(rows, variant)
    e, q, _ = study.parent.tensors(changed)
    before = study.parent.state_hash(model.state_dict())
    with torch.no_grad():
        # Actual Reader forward at each prefix, without introducing a second native recurrence.
        trace = torch.stack([torch.cat([model(x[:, :t], y)
                           for x, y in zip(e.split(128), q.split(128))]) for t in range(1, 33)], dim=1)
    assert before == study.parent.state_hash(model.state_dict())
    return trace, positions


def reference_trace(model, rows, variant):
    rebuilt, positions = [], []
    for row in rows:
        validate(row)
        event = next(e for e in row['events'] if e[0] == row['query'])
        index = next(i for i, e in enumerate(row['events']) if e[0] == row['query'])
        others = [e for e in row['events'] if e[0] != row['query']]
        if variant != 'native':
            index = POSITIONS[variant]
            events = others[:index] + [event] + others[index:]
        else:
            events = row['events']
        rebuilt.append(events); positions.append(index)
    e = torch.tensor(rebuilt); qids = torch.tensor([r['query'] for r in rows])
    with torch.no_grad():
        k = model.key.weight[e[..., 0]]
        k = k / k.square().sum(-1, keepdim=True).clamp_min(1e-24).sqrt()
        v = model.value.weight[e[..., 1]]
        q = model.key.weight[qids]
        q = q / q.square().sum(-1, keepdim=True).clamp_min(1e-24).sqrt()
        state = v.new_zeros(len(rows), 16, 8); eye = torch.eye(16, dtype=v.dtype)
        result = []
        for t in range(32):
            if model.arm == 'delta':
                kt = k[:, t]; beta = model.write_logit.sigmoid()
                state = (eye - beta * kt[:, :, None] * kt[:, None, :]) @ state + beta * kt[:, :, None] * v[:, t, None, :]
                read = (q[:, None, :] @ state).squeeze(1)
            else:
                scores = (k[:, :t + 1] @ q[:, :, None]).squeeze(-1) * model.log_scale.exp()
                weights = (scores - scores.max(-1, keepdim=True).values).exp()
                weights = weights / weights.sum(-1, keepdim=True)
                read = (weights[:, None, :] @ v[:, :t + 1]).squeeze(1)
            result.append(read @ model.output.weight.T + model.output.bias)
    return torch.stack(result, dim=1), positions


def summarize(trace, positions, rows, reducer=None):
    reducer = study.parent.metrics if reducer is None else reducer
    immediate = trace[torch.arange(len(rows)), torch.tensor(positions)]
    final = trace[:, -1]
    curve = []
    for offset in range(32):
        indices = [i for i, p in enumerate(positions) if p + offset < 32]
        if not indices:
            continue
        scores = torch.stack([trace[i, positions[i] + offset] for i in indices])
        selected = [rows[i] for i in indices]
        curve.append(dict(later_writes=offset, **reducer(scores, selected)['all']))
    return dict(immediate=reducer(immediate, rows)['all'], final=reducer(final, rows)['all'],
                immediate_to_final=prior.paired_counts(immediate, final, rows), curve=curve)


def compare_summary(actual, ref):
    for stage in ('immediate', 'final'):
        prior.check_metrics({'all': actual[stage]}, {'all': ref[stage]})
    assert actual['immediate_to_final'] == ref['immediate_to_final']
    assert len(actual['curve']) == len(ref['curve'])
    for a, b in zip(actual['curve'], ref['curve']):
        assert a['later_writes'] == b['later_writes']
        prior.check_metrics({'all': {k: a[k] for k in ('count', 'correct', 'accuracy', 'loss')}},
                            {'all': {k: b[k] for k in ('count', 'correct', 'accuracy', 'loss')}})


def independent_pairs(a, b, rows):
    counts = dict(count=len(rows), rescued=0, harmed=0, both_correct=0, both_wrong=0)
    for x, y, row in zip(a.tolist(), b.tolist(), rows):
        ca = max(range(6), key=lambda i: x[i]) == row['label']
        cb = max(range(6), key=lambda i: y[i]) == row['label']
        key = 'both_correct' if ca and cb else 'both_wrong' if not ca and not cb else 'harmed' if ca else 'rescued'
        counts[key] += 1
    return counts


def audit_cell(checkpoint, arm, rows, variant, trace, positions, summary):
    model = prior.load(checkpoint, arm)
    ref, expected_positions = reference_trace(model, rows, variant)
    assert positions == expected_positions
    error = prior.check_scores(trace.reshape(-1, 6), ref.reshape(-1, 6))
    expected = summarize(ref, expected_positions, rows, prior.reduce_scores)
    immediate = ref[torch.arange(len(rows)), torch.tensor(expected_positions)]
    expected['immediate_to_final'] = independent_pairs(immediate, ref[:, -1], rows)
    compare_summary(summary, expected)
    return error, ref


def check_coverage(cells, seeds):
    keys = [(c['seed'], c['arm'], c['variant']) for c in cells]
    expected = set(itertools.product(seeds, study.parent.ARMS, VARIANTS))
    assert len(keys) == len(expected) and set(keys) == expected, 'Missing or duplicate order cells'


def run():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    _, data, chosen = prior.parent_evidence()
    rows = data['pools']['test_32']; assert len(rows) == 384
    assert not OUT.exists(), 'Preserve existing execution'
    source = dependencies(); OUT.mkdir()
    protected = json.loads((prior.OUT / 'preserved.json').read_text())
    for p in prior.OUT.rglob('*'):
        if p.is_file():
            protected[p.relative_to(ROOT).as_posix()] = study.parent.digest(p)
    for p in ('docs/topics/attention/history-repetition-results.md', 'docs/topics/attention/history-repetition-guide.md'):
        protected[p] = study.parent.digest(ROOT / p)
    assert all(study.parent.digest(ROOT / p) == h for p, h in protected.items())
    study.parent.write(OUT / 'preserved.json', protected)
    spec = importlib.util.spec_from_file_location('test_query_write_order', ROOT / 'tests/test_query_write_order.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    study.parent.write(OUT / 'preflight.json', dict(passed=result.wasSuccessful(), methods=result.testsRun, source_dependencies=source))
    assert result.wasSuccessful(), 'Preflight failed; no saved-model evaluation'
    study.parent.write(OUT / 'lock.json', dict(source_dependencies=source, seeds=study.CONFIRM_SEEDS,
        variants=VARIANTS, expected_cells=len(chosen)*len(VARIANTS), expected_prefix_predictions=len(chosen)*len(VARIANTS)*len(rows)*32,
        data_sha256=study.parent.digest(study.OUT / 'confirmation_data.json'),
        parent_report_sha256=study.parent.digest(study.OUT / 'report.json')))
    cells, scores = [], {}
    for item in chosen:
        cp = ROOT / item['checkpoint']; assert study.parent.digest(cp) == item['checkpoint_sha256']
        model = prior.load(cp, item['arm']); native = None
        for variant in VARIANTS:
            trace, positions = evaluate(model, rows, variant); summary = summarize(trace, positions, rows)
            error, _ = audit_cell(cp, item['arm'], rows, variant, trace, positions, summary)
            if variant == 'native':
                native = trace[:, -1]
                prior.check_metrics({'all': summary['final']}, {'all': item['metrics']['test_32']['all']})
            difference = float((trace[:, -1] - native).abs().max())
            if item['arm'] == 'softmax':
                assert difference < 1e-9, 'Softmax order invariance failed'
            scores[f"{item['seed']}_{item['arm']}_{variant}"] = trace
            cells.append(dict(seed=item['seed'], arm=item['arm'], variant=variant, positions=positions,
                checkpoint=item['checkpoint'], checkpoint_sha256=item['checkpoint_sha256'], summary=summary,
                final_vs_native=prior.paired_counts(native, trace[:, -1], rows), native_max_logit_change=difference,
                max_logit_error=error))
        print(f"Evaluated {item['seed']} {item['arm']}", flush=True)
    check_coverage(cells, study.CONFIRM_SEEDS)
    assert source == dependencies()
    assert all(study.parent.digest(ROOT / p) == h for p, h in protected.items())
    torch.save(scores, OUT / 'traces.pt')
    study.parent.write(OUT / 'report.json', dict(status='complete', training_updates=0, device='cpu', dtype='float64', threads=1,
        source_dependencies=source, cells=cells, final_predictions=len(cells)*len(rows), prefix_predictions=len(cells)*len(rows)*32,
        traces_sha256=study.parent.digest(OUT / 'traces.pt'), protected_files_unchanged=len(protected),
        scope='Opened 32-key test pool, fixed maps and length, query-aware reordering; no learned repair. Prefix quality reported only once queried key has appeared.'))


def audit():
    torch.set_num_threads(1)
    _, data, chosen = prior.parent_evidence(); rows = data['pools']['test_32']
    r = json.loads((OUT / 'report.json').read_text()); lock = json.loads((OUT / 'lock.json').read_text())
    assert r['source_dependencies'] == lock['source_dependencies'] == dependencies()
    assert lock['data_sha256'] == study.parent.digest(study.OUT / 'confirmation_data.json')
    assert lock['parent_report_sha256'] == study.parent.digest(study.OUT / 'report.json')
    assert r['traces_sha256'] == study.parent.digest(OUT / 'traces.pt')
    saved = torch.load(OUT / 'traces.pt', weights_only=True)
    check_coverage(r['cells'], study.CONFIRM_SEEDS); assert len(saved) == len(r['cells']) == 24
    assert r['training_updates'] == 0 and r['final_predictions'] == 9216 and r['prefix_predictions'] == 294912
    errors, metrics = [], []
    chosen = {(x['seed'], x['arm']): x for x in chosen}
    for c in r['cells']:
        item = chosen[c['seed'], c['arm']]
        assert c['checkpoint'] == item['checkpoint'] and c['checkpoint_sha256'] == study.parent.digest(ROOT / c['checkpoint'])
        key = f"{c['seed']}_{c['arm']}_{c['variant']}"
        assert saved[key].shape == (384, 32, 6)
        error, ref = audit_cell(ROOT / c['checkpoint'], c['arm'], rows, c['variant'], saved[key], c['positions'], c['summary'])
        assert error == c['max_logit_error']; errors.append(error)
        native = saved[f"{c['seed']}_{c['arm']}_native"][:, -1]
        assert independent_pairs(native, ref[:, -1], rows) == c['final_vs_native']
        assert c['native_max_logit_change'] == float((saved[key][:, -1]-native).abs().max())
        if c['arm'] == 'softmax': assert c['native_max_logit_change'] < 1e-9
        if c['variant'] == 'native': prior.check_metrics({'all': c['summary']['final']}, {'all': item['metrics']['test_32']['all']})
        metrics.append(dict(seed=c['seed'], arm=c['arm'], variant=c['variant'],
            immediate_accuracy=c['summary']['immediate']['accuracy'], immediate_ce=c['summary']['immediate']['loss'],
            final_accuracy=c['summary']['final']['accuracy'], final_ce=c['summary']['final']['loss'],
            **{k:c['summary']['immediate_to_final'][k] for k in ('rescued','harmed','both_correct','both_wrong')}))
    protected = json.loads((OUT / 'preserved.json').read_text())
    assert all(study.parent.digest(ROOT / p) == h for p, h in protected.items())
    assert not (OUT / 'independent_audit.json').exists()
    with (OUT / 'metrics.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)
    study.parent.write(OUT / 'independent_audit.json', dict(passed=True, report_sha256=study.parent.digest(OUT / 'report.json'),
        source_dependencies=dependencies(), cells=24, prefix_predictions=294912, final_predictions=9216,
        max_logit_error=max(errors), argmax_disagreements=0, protected_files_unchanged=len(protected),
        softmax_max_logit_change=max(c['native_max_logit_change'] for c in r['cells'] if c['arm']=='softmax'),
        scope='Independent list reordering, ordered matrix transitions, explicit exponential normalization and Python CE/category reductions. Shared data/weights/PyTorch.'))
    print('Query-write order audit passes.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=('run', 'audit'))
    args = p.parse_args(); run() if args.phase == 'run' else audit()
