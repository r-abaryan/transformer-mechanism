"""Saved-reader whole-history repetition and separate post-run scalar audit."""
import argparse
import importlib.util
import itertools
import json
import sys
import unittest
from pathlib import Path

import torch
import association_load as study
from audit_delta_retrieval import reduce_scores, check_metrics

ROOT = study.ROOT
OUT = ROOT / 'results/history_repetition'
FACTORS = (1, 2, 4)
POOLS = tuple(f'test_{m}' for m in study.LOADS)


def dependencies():
    paths = ('experiments/history_repetition.py', 'tests/test_history_repetition.py',
             'docs/protocols/history-repetition.md', 'experiments/association_load.py',
             'experiments/delta_retrieval.py', 'experiments/audit_delta_retrieval.py')
    return {p: study.parent.digest(ROOT / p) for p in paths}


def load(checkpoint, arm):
    model = study.LoadReader(arm)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
    model.eval()
    if arm == 'softmax':
        assert float(model.log_recency.exp()) == 0
    return model


def repeat_events(events, factor):
    assert factor in FACTORS, 'Undeclared repetition factor'
    return events.repeat(1, factor, 1)


def evaluate(model, rows, factor):
    before = study.parent.state_hash(model.state_dict())
    events, query, _ = study.parent.tensors(rows)
    with torch.no_grad():
        scores = torch.cat([model(repeat_events(e, factor), q)
                            for e, q in zip(events.split(128), query.split(128))])
    assert study.parent.state_hash(model.state_dict()) == before
    # Group tags describe the original history; only all-example metrics are used.
    return study.parent.metrics(scores, rows)['all'], scores


def reference_scores(model, rows, factor):
    # Independently rebuild repeated lists, rather than using the producer's tensor transform.
    events = []
    for row in rows:
        original = dict(row['events'])
        repeated = [list(pair) for _ in range(factor) for pair in row['events']]
        assert dict(repeated) == original and original[row['query']] == row['label'], 'Changed map or label'
        assert len(repeated) == len(row['events']) * factor
        events.append(repeated)
    e = torch.tensor(events); q = torch.tensor([r['query'] for r in rows])
    with torch.no_grad():
        return torch.cat([study.parent.reference(model, a, b) for a, b in zip(e.split(128), q.split(128))])


def check_scores(actual, reference):
    error = float((actual - reference).abs().max())
    changes = int((actual.argmax(-1) != reference.argmax(-1)).sum())
    assert torch.isfinite(actual).all() and error < 1e-9 and changes == 0, 'Repetition equations disagree'
    return error


def audit_cell(checkpoint, arm, rows, factor, metrics, scores):
    model = load(checkpoint, arm)
    ref = reference_scores(model, rows, factor)
    error = check_scores(scores, ref)
    check_metrics({'all': metrics}, {'all': reduce_scores(ref, rows)['all']})
    return error, ref


def paired_counts(native, repeated, rows):
    labels = torch.tensor([r['label'] for r in rows])
    a, b = native.argmax(-1) == labels, repeated.argmax(-1) == labels
    return dict(count=len(rows), rescued=int((~a & b).sum()), harmed=int((a & ~b).sum()),
                both_correct=int((a & b).sum()), both_wrong=int((~a & ~b).sum()))


def decision(cells):
    expected = set(itertools.product(study.CONFIRM_SEEDS, study.parent.ARMS, POOLS, FACTORS))
    keys = [(c['seed'], c['arm'], c['pool'], c['factor']) for c in cells]
    assert len(keys) == len(expected) and set(keys) == expected, 'Missing or duplicate repetition cells'
    lookup = {(c['seed'], c['arm'], c['pool'], c['factor']): c for c in cells}
    delta = {}
    for factor in FACTORS[1:]:
        high = [lookup[s, 'delta', 'test_32', factor] for s in study.CONFIRM_SEEDS]
        delta[str(factor)] = dict(
            all_seed_10_point_gain_at_32=all(c['metrics']['accuracy'] - lookup[c['seed'], 'delta', 'test_32', 1]['metrics']['accuracy'] >= .10 for c in high),
            all_seed_95_percent_at_32=all(c['metrics']['accuracy'] >= .95 for c in high),
            low_load_preservation=all(lookup[s, 'delta', f'test_{m}', factor]['metrics']['accuracy'] + .02 >=
                                      lookup[s, 'delta', f'test_{m}', 1]['metrics']['accuracy']
                                      for s in study.CONFIRM_SEEDS for m in (2, 8)))
    return dict(delta=delta, softmax_logit_invariance=all(c['native_max_logit_change'] < 1e-9 for c in cells if c['arm'] == 'softmax'))


def parent_evidence():
    report = json.loads((study.OUT / 'report.json').read_text())
    assert report['status'] == 'complete' and report['source_dependencies'] == study.dependencies()
    audit = json.loads((study.OUT / 'independent_audit.json').read_text())
    assert audit['passed'] and audit['report_sha256'] == study.parent.digest(study.OUT / 'report.json')
    assert audit['audit_source_sha256'] == study.parent.digest(ROOT / 'experiments/audit_association_load.py')
    path = study.OUT / 'confirmation_data.json'
    assert report['data_sha256']['confirmation'] == study.parent.digest(path)
    data = json.loads(path.read_text()); study.audit_data(data)
    chosen = [r for r in report['runs'] if r['phase'] == 'confirmation']
    assert len(chosen) == 6 and {(r['seed'], r['arm']) for r in chosen} == set(itertools.product(study.CONFIRM_SEEDS, study.parent.ARMS))
    return report, data, chosen


def run():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    old, data, chosen = parent_evidence()
    source = dependencies()
    assert not OUT.exists(), 'Diagnostic directory already exists; preserve previous evidence'
    OUT.mkdir()
    prior = json.loads((study.OUT / 'preserved.json').read_text())
    prior.update(old['source_dependencies'])
    prior['docs/topics/attention/association-load-results.md'] = study.parent.digest(ROOT / 'docs/topics/attention/association-load-results.md')
    prior['docs/topics/attention/association-load-guide.md'] = study.parent.digest(ROOT / 'docs/topics/attention/association-load-guide.md')
    for p in study.OUT.rglob('*'):
        if p.is_file():
            prior[str(p.relative_to(ROOT)).replace('\\', '/')] = study.parent.digest(p)
    assert all(study.parent.digest(ROOT / p) == h for p, h in prior.items())
    study.parent.write(OUT / 'preserved.json', prior)
    spec = importlib.util.spec_from_file_location('test_history_repetition', ROOT / 'tests/test_history_repetition.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    study.parent.write(OUT / 'preflight.json', dict(passed=result.wasSuccessful(), methods=result.testsRun, source_dependencies=source))
    assert result.wasSuccessful(), 'Preflight failed; saved readers not evaluated'
    study.parent.write(OUT / 'lock.json', dict(source_dependencies=source, seeds=study.CONFIRM_SEEDS,
                       arms=study.parent.ARMS, pools=POOLS, factors=FACTORS, expected_cells=len(chosen) * len(POOLS) * len(FACTORS),
                       parent_report_sha256=study.parent.digest(study.OUT / 'report.json'),
                       parent_data_sha256=study.parent.digest(study.OUT / 'confirmation_data.json'),
                       preflight_sha256=study.parent.digest(OUT / 'preflight.json')))
    cells, scores, pairs = [], {}, []
    for item in chosen:
        cp = ROOT / item['checkpoint']; assert study.parent.digest(cp) == item['checkpoint_sha256']
        model = load(cp, item['arm'])
        for pool in POOLS:
            rows = data['pools'][pool]
            original = None
            for factor in FACTORS:
                metrics, actual = evaluate(model, rows, factor)
                error, _ = audit_cell(cp, item['arm'], rows, factor, metrics, actual)
                if factor == 1:
                    original = actual
                    check_metrics({'all': item['metrics'][pool]['all']}, {'all': metrics})
                else:
                    pairs.append(dict(seed=item['seed'], arm=item['arm'], pool=pool, factor=factor,
                                      **paired_counts(original, actual, rows)))
                difference = float((actual - original).abs().max())
                if item['arm'] == 'softmax':
                    assert difference < 1e-9, 'Softmax repetition identity failed'
                key = f"{item['seed']}_{item['arm']}_{pool}_{factor}"
                scores[key] = actual
                cells.append(dict(seed=item['seed'], arm=item['arm'], pool=pool, load=rows[0]['load'], factor=factor,
                                  events=32 * factor, metrics=metrics, max_logit_error=error, native_max_logit_change=difference,
                                  checkpoint=item['checkpoint'], checkpoint_sha256=item['checkpoint_sha256']))
    assert dependencies() == source
    assert all(study.parent.digest(ROOT / p) == h for p, h in prior.items())
    torch.save(scores, OUT / 'scores.pt')
    study.parent.write(OUT / 'report.json', dict(status='complete', training_updates=0, device='cpu', dtype='float64', threads=1,
        source_dependencies=source, cells=cells, paired_outcomes=pairs, decision=decision(cells),
        expected_cells=len(chosen) * len(POOLS) * len(FACTORS), predictions=sum(c['metrics']['count'] for c in cells),
        scores_sha256=study.parent.digest(OUT / 'scores.pt'), prior_files_unchanged=len(prior),
        scope='Saved-model diagnostic on opened test pools; fixed maps, whole-history repetition, new lengths outside training support.'))
    print('All saved-reader repetition cases complete. Run audit next.', flush=True)


def audit():
    import csv
    torch.set_num_threads(1)
    old, data, chosen = parent_evidence()
    report = json.loads((OUT / 'report.json').read_text()); lock = json.loads((OUT / 'lock.json').read_text())
    assert report['status'] == 'complete' and report['training_updates'] == 0
    assert report['source_dependencies'] == dependencies() == lock['source_dependencies']
    assert lock['parent_report_sha256'] == study.parent.digest(study.OUT / 'report.json')
    assert lock['parent_data_sha256'] == study.parent.digest(study.OUT / 'confirmation_data.json')
    assert lock['preflight_sha256'] == study.parent.digest(OUT / 'preflight.json')
    assert report['scores_sha256'] == study.parent.digest(OUT / 'scores.pt')
    saved = torch.load(OUT / 'scores.pt', weights_only=True)
    assert len(report['cells']) == len(saved) == report['expected_cells'] == 3 * 2 * 4 * 3
    assert len(report['paired_outcomes']) == 3 * 2 * 4 * 2
    chosen = {(r['seed'], r['arm']): r for r in chosen}
    refs, errors, csv_rows = {}, [], []
    for c in report['cells']:
        item = chosen[c['seed'], c['arm']]
        assert c['checkpoint'] == item['checkpoint'] and c['checkpoint_sha256'] == item['checkpoint_sha256']
        assert study.parent.digest(ROOT / c['checkpoint']) == c['checkpoint_sha256']
        assert c['load'] == int(c['pool'].split('_')[1]) and c['events'] == 32 * c['factor']
        key = f"{c['seed']}_{c['arm']}_{c['pool']}_{c['factor']}"
        error, ref = audit_cell(ROOT / c['checkpoint'], c['arm'], data['pools'][c['pool']], c['factor'], c['metrics'], saved[key])
        assert c['max_logit_error'] == error
        refs[c['seed'], c['arm'], c['pool'], c['factor']] = ref
        errors.append(error)
        csv_rows.append({**{k: c[k] for k in ('seed', 'arm', 'load', 'factor', 'events')}, **c['metrics']})
    for pair in report['paired_outcomes']:
        rows = data['pools'][pair['pool']]
        a = refs[pair['seed'], pair['arm'], pair['pool'], 1].tolist()
        b = refs[pair['seed'], pair['arm'], pair['pool'], pair['factor']].tolist()
        counts = dict(count=len(rows), rescued=0, harmed=0, both_correct=0, both_wrong=0)
        for av, bv, row in zip(a, b, rows):
            correct_a = max(range(6), key=lambda i: av[i]) == row['label']
            correct_b = max(range(6), key=lambda i: bv[i]) == row['label']
            category = 'both_correct' if correct_a and correct_b else 'both_wrong' if not correct_a and not correct_b else 'harmed' if correct_a else 'rescued'
            counts[category] += 1
        assert all(pair[k] == v for k, v in counts.items())
    for c in report['cells']:
        key = f"{c['seed']}_{c['arm']}_{c['pool']}_{c['factor']}"
        native = f"{c['seed']}_{c['arm']}_{c['pool']}_1"
        assert c['native_max_logit_change'] == float((saved[key] - saved[native]).abs().max())
        if c['factor'] == 1:
            check_metrics({'all': chosen[c['seed'], c['arm']]['metrics'][c['pool']]['all']}, {'all': c['metrics']})
        if c['arm'] == 'softmax':
            assert c['native_max_logit_change'] < 1e-9
    assert decision(report['cells']) == report['decision']
    prior = json.loads((OUT / 'preserved.json').read_text())
    assert all(study.parent.digest(ROOT / p) == h for p, h in prior.items())
    assert report['predictions'] == sum(c['metrics']['count'] for c in report['cells']) == 27648
    assert not (OUT / 'metrics.csv').exists() and not (OUT / 'independent_audit.json').exists()
    with (OUT / 'metrics.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    study.parent.write(OUT / 'independent_audit.json', dict(passed=True, report_sha256=study.parent.digest(OUT / 'report.json'),
        source_dependencies=dependencies(), cells=72, paired_cells=48, predictions=27648, argmax_changes=0,
        max_logit_error=max(errors), prior_files_unchanged=len(prior), baseline_matches_archive=True,
        softmax_max_logit_change=max(c['native_max_logit_change'] for c in report['cells'] if c['arm'] == 'softmax'),
        scope='Independent repeated-list construction, matrix equations and Python scalar/paired reductions; shared parameters/data/PyTorch.'))
    print('Independent repetition audit passes.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('run', 'audit'))
    args = parser.parse_args()
    run() if args.phase == 'run' else audit()
