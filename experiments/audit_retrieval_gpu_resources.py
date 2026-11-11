"""Independent equations, storage formulas and raw-sample reductions."""
import itertools
import json
import math
import statistics

import torch
import retrieval_gpu_resources as study


def audit_cell(cell, evidence, run):
    b, n = cell['batch'], cell['length']
    # Independent dimensions/formula, not the producer's payload helper.
    shapes = [[b, n, 16], [b, n, 8]] if run['arm'] == 'softmax' else [[b, 16, 8]]
    expected = sum(math.prod(s) * 4 for s in shapes)
    assert cell['state_shapes'] == shapes and cell['payload_bytes'] == expected, 'Payload disagrees'
    assert cell['retained_allocated_bytes'] >= expected
    baseline = cell['baseline_allocated_bytes']
    assert cell['prefill_peak_allocated_bytes'] >= baseline + cell['retained_allocated_bytes']
    assert cell['query_peak_allocated_bytes'] >= baseline + cell['retained_allocated_bytes']
    cpu = study.load(run, rounded=True).double()
    with torch.no_grad():
        ref = study.parent.reference(cpu, evidence['events'], evidence['query'])
    torch.testing.assert_close(ref, evidence['reference'], atol=1e-12, rtol=1e-12)
    checked = study.check_scores(evidence['gpu_scores'], ref)
    assert cell['numerical'] == checked
    for mode in ('prefill', 'query'):
        for label in ('wall', 'event'):
            samples = cell[mode][label + '_ms']
            assert len(samples) == 10 and all(math.isfinite(v) and v > 0 for v in samples)
            assert cell[mode][label + '_median_ms'] == statistics.median(samples)
    assert cell['checkpoint_sha256'] == run['checkpoint_sha256']
    return checked


def main():
    torch.set_num_threads(1)
    out = study.OUT
    lock = json.loads((out / 'lock.json').read_text())
    assert lock['source_dependencies'] == study.dependencies()
    pre = json.loads((out / 'preflight.json').read_text())
    assert lock['preflight_sha256'] == study.parent.digest(out / 'preflight.json')
    assert pre['scores_sha256'] == study.parent.digest(out / 'preflight_scores.pt')
    chosen = {(r['seed'], r['arm']): r for r in study.runs()}
    data = json.loads((study.parent.OUT / 'confirmation_data.json').read_text())
    pool_scores = torch.load(out / 'preflight_scores.pt', weights_only=True)
    assert len(pool_scores) == 48
    assert {(e['seed'], e['arm'], e['pool']) for e in pool_scores} == set(
        (s, a, p) for s, a in chosen for p in data['pools'])
    predictions = 0
    for saved, recorded in zip(pool_scores, pre['pool_checks']):
        run = chosen[saved['seed'], saved['arm']]
        assert (saved['seed'], saved['arm'], saved['pool']) == (recorded['seed'], recorded['arm'], recorded['pool'])
        e, q, _ = study.parent.tensors(data['pools'][saved['pool']])
        for rounded, key in ((True, 'rounded_reference'), (False, 'original_reference')):
            cpu = study.load(run, rounded=rounded).double()
            with torch.no_grad():
                ref = torch.cat([study.parent.reference(cpu, a, b) for a, b in zip(e.split(128), q.split(128))])
            assert recorded[key] == study.check_scores(saved['gpu_scores'], ref)
        predictions += len(saved['gpu_scores'])
    assert predictions == pre['predictions'] == 20736
    workers, hashes, changes, error = [], {}, 0, 0.
    expected = {(s, a, b, n) for s, a in chosen for b, n in study.SHAPES}
    for index in study.WORKERS:
        path = out / f'worker_{index}.json'
        worker = json.loads(path.read_text())
        assert worker['status'] == 'complete' and worker['worker'] == index
        assert worker['source_dependencies'] == lock['source_dependencies']
        assert worker['lock_sha256'] == study.parent.digest(out / 'lock.json')
        tp = out / f'worker_{index}_scores.pt'
        assert worker['scores_sha256'] == study.parent.digest(tp)
        saved = torch.load(tp, weights_only=True)
        assert len(worker['cells']) == len(saved) == len(expected)
        assert {(c['seed'], c['arm'], c['batch'], c['length']) for c in worker['cells']} == expected
        for cell, evidence in zip(worker['cells'], saved):
            assert all(cell[k] == evidence[k] for k in ('seed', 'arm', 'batch', 'length'))
            checked = audit_cell(cell, evidence, chosen[cell['seed'], cell['arm']])
            changes += checked['argmax_changes']; error = max(error, checked['max_absolute_error'])
        profiles = worker['profiles']
        assert len(profiles) == 18
        assert {(c['seed'], c['arm'], c['length']) for c in profiles} == {(s, a, n) for s, a in chosen for n in (8, 32, 128)}
        for c in profiles:
            assert c['batch'] == 64 and c['updates'] == 1 and c['targets'] == 64 and c['events'] == 64 * c['length']
            parameters = 232 if c['arm'] == 'softmax' else 231
            tensors = 6 if c['arm'] == 'softmax' else 5
            assert c['parameter_bytes'] == parameters * 4
            assert c['optimizer_parameter_states'] == tensors
            assert c['optimizer_state_bytes'] == 2 * parameters * 4 + tensors * 4
            assert c['peak_reserved_bytes'] >= c['peak_allocated_bytes'] >= c['baseline_allocated_bytes']
            assert math.isfinite(c['loss'])
        workers.append(worker); hashes[path.name] = study.parent.digest(path)
    assert len({w['process_id'] for w in workers}) == 3
    preserved = json.loads((out / 'preserved.json').read_text())
    assert all(study.parent.digest(study.ROOT / p) == h for p, h in preserved.items())
    report = dict(status='complete', scope='resource-only; original quality gate remains false',
                  source_dependencies=lock['source_dependencies'], worker_sha256=hashes,
                  cells=[dict(worker=w['worker'], **c) for w in workers for c in w['cells']],
                  profiles=[dict(worker=w['worker'], **c) for w in workers for c in w['profiles']],
                  hardware=workers[0]['hardware'])
    assert not (out / 'report.json').exists()
    study.parent.write(out / 'report.json', report)
    study.parent.write(out / 'independent_audit.json', dict(passed=True,
        report_sha256=study.parent.digest(out / 'report.json'), lock_sha256=study.parent.digest(out / 'lock.json'),
        audit_source_sha256=study.parent.digest(study.ROOT / 'experiments/audit_retrieval_gpu_resources.py'),
        numerical_cells=108, resource_predictions=sum(c['batch'] for c in report['cells']),
        original_pool_predictions=predictions, argmax_changes=changes, max_resource_logit_error=error,
        timing_samples=108 * 2 * 10, memory_profiles=54, disposable_updates=54,
        preserved_files=len(preserved), original_quality_gate_unchanged=True))
    print('Independent audit passes; original quality gate remains false.')


if __name__ == '__main__':
    main()
