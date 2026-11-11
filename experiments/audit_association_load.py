"""Independent scalar, stream, fitting and controlled-write audit."""
import csv
import hashlib
import itertools
import json
import struct
from collections import Counter
from pathlib import Path

import torch
import association_load as study
from audit_delta_retrieval import reduce_scores, check_metrics


def audit_run(run, payload, expected_steps=study.STEPS):
    path = study.ROOT / run['checkpoint']
    assert study.parent.digest(path) == run['checkpoint_sha256'], 'Checkpoint hash'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['step'] == expected_steps
    if expected_steps:
        assert all(float(s['step']) == expected_steps for s in saved['optimizer']['state'].values())
        assert len(saved['optimizer']['state']) == 5
    model = study.LoadReader(run['arm']); model.load_state_dict(saved['model'])
    cells = []
    for name, metrics in run['metrics'].items():
        rows = payload['pools'][name]
        e, q, _ = study.parent.tensors(rows)
        with torch.no_grad():
            native = torch.cat([model(a, b) for a, b in zip(e.split(128), q.split(128))])
            ref = torch.cat([study.parent.reference(model, a, b) for a, b in zip(e.split(128), q.split(128))])
        error = float((native - ref).abs().max())
        assert error < 1e-9 and torch.equal(native.argmax(-1), ref.argmax(-1)), 'Equation disagreement'
        check_metrics(metrics, reduce_scores(ref, rows))
        cells.append(dict(seed=run['seed'], arm=run['arm'], phase=run['phase'], recipe=run['recipe'],
                          pool=name, predictions=len(rows), max_logit_error=error))
    return cells


def independent_selection(runs):
    output = {}
    for arm in ('softmax', 'delta'):
        eligible = []
        for index, recipe in enumerate(('constant', 'cosine')):
            subset = [r for r in runs if r['arm'] == arm and r['recipe'] == recipe]
            valid = len(subset) == 2 and {r['seed'] for r in subset} == {3313, 3319}
            valid &= all(r['status'] == 'complete' for r in subset)
            if valid and all(r['metrics']['train_2']['all']['correct'] / r['metrics']['train_2']['all']['count'] >= .99 for r in subset):
                eligible.append((max(r['metrics'][f'train_{m}']['all']['loss'] for r in subset for m in (2, 8, 16, 32)), index, recipe))
        output[arm] = min(eligible)[2] if eligible else None
    return output


def independent_decision(runs):
    out = {}
    for arm in ('softmax', 'delta'):
        subset = [r for r in runs if r['phase'] == 'confirmation' and r['arm'] == arm]
        assert len(subset) == 3 and {r['seed'] for r in subset} == {3323, 3329, 3331}
        acc = lambda r, pool: r['metrics'][pool]['all']['correct'] / r['metrics'][pool]['all']['count']
        out[arm] = dict(loads={str(m): dict(all_seed_training_fit=min(acc(r, f'train_{m}') for r in subset) >= .99,
                                          all_seed_test_reliability=min(acc(r, f'test_{m}') for r in subset) >= .95)
                              for m in (2, 8, 16, 32)},
            low_load_control=all(acc(r, 'train_2') >= .99 and acc(r, 'test_2') >= .95 for r in subset),
            all_seed_10_point_load_drop=all(acc(r, 'test_2') - acc(r, 'test_32') >= .10 for r in subset))
    return out


def audit_probe(run, pool, rows, saved):
    model = study.LoadReader(run['arm'])
    model.load_state_dict(torch.load(study.ROOT / run['checkpoint'], weights_only=True)['model'])
    # Labels are unchanged because the added write repeats a different key's value.
    events = []
    for row in rows:
        q = row['query']
        k, v = next((k, v) for k, v in row['events'] if k != q)
        original_label = dict(row['events'])[q]
        after = row['events'] + [[k, v]]
        assert dict(after)[q] == original_label == row['label']
        events.append(after)
    e, q, y = study.parent.tensors(rows)
    changed = torch.tensor(events)
    with torch.no_grad():
        before = torch.cat([study.parent.reference(model, a, b) for a, b in zip(e.split(128), q.split(128))])
        after = torch.cat([study.parent.reference(model, a, b) for a, b in zip(changed.split(128), q.split(128))])
    errors = []
    for key, ref in (('before', before), ('after', after)):
        errors.append(float((saved[key] - ref).abs().max()))
        assert errors[-1] < 1e-9 and torch.equal(saved[key].argmax(-1), ref.argmax(-1))
    summary = run['probes'][pool]
    a, b = before.argmax(-1) == y, after.argmax(-1) == y
    for key, expected in dict(count=len(rows), harmed=int((a & ~b).sum()), rescued=int((~a & b).sum()),
                              before_correct=int(a.sum()), after_correct=int(b.sum())).items():
        assert summary[key] == expected
    assert abs(summary['mean_logit_shift_l2'] - float((after - before).norm(dim=-1).mean())) < 1e-10
    check_metrics(summary['after_metrics'], reduce_scores(after, rows))
    # Re-exercise the cross-write identity against the actual final states.
    measured, _ = study.probes(model, rows)
    for key in ('mean_read_shift_l2', 'mean_absolute_query_write_dot'):
        if key in measured:
            assert abs(measured[key] - summary[key]) < 1e-12
    return dict(seed=run['seed'], arm=run['arm'], pool=pool, predictions=len(rows), max_logit_error=max(errors))


def main():
    torch.set_num_threads(1)
    out = study.OUT
    report = json.loads((out / 'report.json').read_text())
    lock = json.loads((out / 'lock.json').read_text())
    assert report['status'] == 'complete' and report['source_dependencies'] == study.dependencies() == lock['source_dependencies']
    assert study.parent.digest(out / 'preflight.json') == lock['preflight_sha256']
    data = {phase: json.loads((out / f'{phase}_data.json').read_text()) for phase in report['data_sha256']}
    all_pools = {}
    for phase, payload in data.items():
        assert study.parent.digest(out / f'{phase}_data.json') == report['data_sha256'][phase]
        assert report['baselines'][phase] == study.audit_data(payload)
        expected_splits = ('train', 'validation', 'test') if phase == 'confirmation' else ('train', 'validation')
        assert set(payload['pools']) == {f'{s}_{m}' for s in expected_splits for m in (2, 8, 16, 32)}
        for name, rows in payload['pools'].items():
            assert len(rows) == 6 * {'train': 128, 'validation': 32, 'test': 64}[name.split('_')[0]]
            families = {}
            for r in rows:
                # Canonicalization removes cyclic value naming, not key identity/order.
                f = min(tuple((k, (v + offset) % 6) for k, v in r['events']) for offset in range(6))
                families.setdefault(f, []).append(r)
            assert all(len(v) == 6 and {r['label'] for r in v} == set(range(6)) for v in families.values())
            all_pools[phase + '_' + name] = rows
    study.audit_data(dict(pools=all_pools))
    assert len(report['runs']) in (8, 14)
    assert {(r['seed'], r['arm'], r['recipe']) for r in report['runs'] if r['phase'] == 'development'} == set(
        itertools.product((3313, 3319), ('softmax', 'delta'), ('constant', 'cosine')))
    selected = independent_selection([r for r in report['runs'] if r['phase'] == 'development'])
    assert selected == {a: c['selected_recipe'] for a, c in report['selection'].items()}
    locked = json.loads((out / 'development_locked.json').read_text())
    assert locked['selection'] == report['selection'] and all(r['phase'] == 'development' for r in locked['runs'])
    cells, probes, csv_rows = [], [], []
    if 'confirmation' in data:
        pretest = json.loads((out / 'pretest.json').read_text())
        assert len(pretest['runs']) == 14 and all(r['status'] == 'complete' for r in pretest['runs'])
        assert all(not any(k.startswith('test_') for k in r['metrics']) for r in pretest['runs'])
        assert {(r['seed'], r['arm']) for r in report['runs'] if r['phase'] == 'confirmation'} == set(
            itertools.product((3323, 3329, 3331), ('softmax', 'delta')))
        assert study.parent.digest(out / 'probe_scores.pt') == report['probe_scores_sha256']
        saved_probes = torch.load(out / 'probe_scores.pt', weights_only=True)
        assert len(saved_probes) == 6
    for run in report['runs']:
        assert run['status'] == 'complete' and run['updates'] == 3000 and run['parameters'] == 615
        assert run['examples'] == 192000 and run['input_events'] == 6144000 and run['examples_per_load'] == 48000
        if run['phase'] == 'confirmation':
            assert run['recipe'] == selected[run['arm']]
        payload = data[run['phase']]
        expected_pools = set(payload['pools'])
        assert set(run['metrics']) == expected_pools
        cells.extend(audit_run(run, payload))
        rng, stream = torch.Generator().manual_seed(run['seed'] + 10000), hashlib.sha256()
        for step in range(3000):
            m = (2, 8, 16, 32)[step % 4]
            idx = torch.randint(len(payload['pools'][f'train_{m}']), (64,), generator=rng)
            stream.update(struct.pack('<I', m)); stream.update(bytes(idx.view(torch.uint8).tolist()))
        assert stream.hexdigest() == run['stream_sha256']
        initial = study.pair(run['seed'])[run['arm']]
        assert study.parent.state_hash(initial.state_dict()) == run['initial_sha256']
        if run['phase'] == 'confirmation':
            assert set(run['probes']) == {f'test_{m}' for m in (2, 8, 16, 32)}
            for name, summary in run['probes'].items():
                probes.append(audit_probe(run, name, payload['pools'][name], saved_probes[f"{run['seed']}_{run['arm']}"][name]))
        for pool, metrics in run['metrics'].items():
            csv_rows.append(dict(phase=run['phase'], seed=run['seed'], arm=run['arm'], recipe=run['recipe'],
                                 pool=pool, **metrics['all']))
    for phase in data:
        for seed in (3313, 3319) if phase == 'development' else (3323, 3329, 3331):
            subset = [r for r in report['runs'] if (r['phase'], r['seed']) == (phase, seed)]
            assert len({r['shared_sha256'] for r in subset}) == len({r['stream_sha256'] for r in subset}) == 1
    if 'confirmation' in data:
        assert independent_decision(report['runs']) == report['confirmation_decision']
    preserved = json.loads((out / 'preserved.json').read_text())
    assert all(study.parent.digest(study.ROOT / p) == h for p, h in preserved.items())
    target = out / 'metrics.csv'; assert not target.exists()
    with target.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    target = out / 'independent_audit.json'; assert not target.exists()
    study.parent.write(target, dict(passed=True, report_sha256=study.parent.digest(out / 'report.json'),
        audit_source_sha256=study.parent.digest(Path(__file__)), cells=cells, probes=probes,
        predictions=sum(c['predictions'] for c in cells), probe_predictions=sum(c['predictions'] for c in probes),
        max_logit_error=max(c['max_logit_error'] for c in cells + probes), argmax_changes=0,
        streams_verified=len(report['runs']), prior_files_unchanged=len(preserved), selection=selected,
        scope='CPU float64; independent matrix equations and Python scalar reductions, shared parameters/data/software.'))
    print('Independent equations, scalar metrics, write interventions and stream audit pass.', flush=True)


if __name__ == '__main__':
    main()
