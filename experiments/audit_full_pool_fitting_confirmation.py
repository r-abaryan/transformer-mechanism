"""Fresh-data and warm-start lineage checks, reusing independent numerical audits."""
import inspect
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.optim.lbfgs as lbfgs_source

import audit_full_pool_fitting as numerical
import confirm_full_pool_fitting as study
from audit_shared_state_feasibility import compare


def family(events):
    offset = events[0][1]
    return tuple((k, (v-offset) % 6) for k, v in events)


def check_data(pools, fixture):
    assert set(pools) == {f'{s}_{m}' for s in ('train', 'validation', 'test') for m in study.prior.LOADS}
    ownership = {}
    historical = {family(row['events']) for path in study.OLD_DATA
                  for rows in json.loads((study.ROOT/path).read_text())['pools'].values() for row in rows}
    for name, rows in pools.items():
        numerical.reference.independent.check_rows(rows)
        split, load = name.split('_')
        load = int(load)
        if not fixture:
            assert len(rows) == {'train': 128, 'validation': 32, 'test': 64}[split]*6
        labels = [0]*6
        for row in rows:
            events = row['events']
            keys = {k for k, v in events}
            assert len(events) == 32 and len(keys) == load
            assert all(sum(k == key for k, v in events) == 32//load for key in keys)
            assert all(len({v for k, v in events if k == key}) == 1 for key in keys)
            identity = family(events)
            assert ownership.setdefault(identity, name) == name, 'Family leakage'
            assert fixture or identity not in historical, 'Historical family reused'
            labels[row['label']] += 1
        assert len(set(labels)) == 1
    if not fixture:
        regenerated = study.data_source.generate(study.DATA_SEED, confirmation=True,
                                                  exclude=study.historical_families())['pools']
        assert pools == regenerated
    return len(ownership)


def check_seed(warm, runs, pools, lock):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    replay = study.replay
    initial = torch.load(study.ROOT/warm['initial_checkpoint'], weights_only=True)
    with torch.random.fork_rng(devices=[]):
        model = study.data_source.pair(warm['seed'])['delta']
    assert initial['step'] == 0 and initial['optimizer']['state'] == {}
    assert replay.equal(initial['model'], model.state_dict())
    assert replay.digest(study.ROOT/warm['initial_checkpoint']) == warm['initial_sha256']
    case = dict(seed=warm['seed'], checkpoint=warm['initial_checkpoint'])
    assert warm['source_step'] == 0 and warm['optimizer_calls'] == lock['warm_steps']
    checks = [numerical.check_adam(case, warm, pools, lock, lock['warm_steps']//lock['ratio'])]
    endpoint = torch.load(study.ROOT/warm['final_checkpoint'], weights_only=True)
    assert endpoint['step'] == lock['warm_steps']
    for run in runs:
        start = torch.load(study.ROOT/run['snapshots'][0]['checkpoint'], weights_only=True)
        assert replay.equal(start['model'], endpoint['model']) and run['source_step'] == endpoint['step']
        assert start['source_step'] == endpoint['step'] and start['procedure_step'] == 0
        if run['arm'] == 'adam':
            assert replay.equal(start['optimizer'], endpoint['optimizer'])
        else:
            assert start['optimizer']['state'] == {}
    case = dict(seed=warm['seed'], checkpoint=warm['final_checkpoint'])
    a, b = runs
    checks.append(numerical.check_lbfgs(case, a, pools, lock))
    checks.append(numerical.check_adam(case, b, pools, lock, a['closure_evaluations']))
    if not lock['fixture']:
        assert (a['histories'], a['answer_targets'], a['input_events']) == (b['histories'], b['answer_targets'], b['input_events'])
    return checks


def audit(out, persist=True):
    report = json.loads((out/'report.json').read_text())
    lock = json.loads((out/'lock.json').read_text())
    replay = study.replay
    assert report['status'] == 'complete'
    replay.check_files(lock['files'])
    assert lock['torch'] == torch.__version__
    assert lock['lbfgs_implementation_sha256'] == replay.digest(Path(inspect.getsourcefile(lbfgs_source)))
    for field, name in (('scores_sha256', 'scores.pt'), ('lock_sha256', 'lock.json'),
                        ('pretest_sha256', 'pretest_decision.json')):
        assert report[field] == replay.digest(out/name)
    assert lock['pools_sha256'] == replay.digest(out/'pools.pt')
    pools = torch.load(out/'pools.pt', weights_only=True)
    families = check_data(pools, lock['fixture'])
    if not lock['fixture']:
        assert (tuple(lock['seeds']), lock['warm_steps'], lock['budget'], lock['ratio'], lock['interval']) == (study.SEEDS, 12000, 128, 48, 32)
        manifest = json.loads((study.OUT/'data_manifest.json').read_text())
        assert manifest['data_sha256'] == replay.digest(study.OUT/'data.pt')
        assert pools == torch.load(study.OUT/'data.pt', weights_only=True)['pools']
        assert report['fresh_test'] and report['selection'] == 'none_locked_confirmation'
    assert [r['seed'] for r in report['warmups']] == lock['seeds']
    assert [(r['seed'], r['arm']) for r in report['runs']] == [(s, a) for s in lock['seeds'] for a in ('lbfgs', 'adam')]
    assert report['workers'] == min(3, len(lock['seeds']))
    checks = []
    with ProcessPoolExecutor(max_workers=report['workers']) as workers:
        futures = [workers.submit(check_seed, warm, [r for r in report['runs'] if r['seed'] == warm['seed']], pools, lock)
                   for warm in report['warmups']]
        for future in futures:
            checks.extend(future.result())
    for run in report['warmups']+report['runs']:
        assert replay.digest(study.ROOT/run['trace']) == run['trace_sha256']
        for snap in run['snapshots']:
            assert replay.digest(study.ROOT/snap['checkpoint']) == snap['sha256']
            cp = torch.load(study.ROOT/snap['checkpoint'], weights_only=True)
            assert cp['procedure_step'] == snap['step'] and cp['source_step'] == run['source_step']
        count = run['closure_evaluations'] if run['arm'] == 'lbfgs' else run['optimizer_calls']
        histories = count*sum(len(pools[f'train_{m}']) for m in study.prior.LOADS) if run['arm'] == 'lbfgs' else count*64
        targets = count*sum(len(pools[f'train_{m}'])*m for m in study.prior.LOADS) if run['arm'] == 'lbfgs' else sum(64*study.prior.LOADS[(run['source_step']+i)%4] for i in range(count))
        assert (run['histories'], run['answer_targets'], run['input_events']) == (histories, targets, histories*32)
    scores = torch.load(out/'scores.pt', weights_only=True)
    expected = {(r['stage'], r['seed'], r['arm'], snap['step'], f'train_{m}')
                for r in report['warmups']+report['runs'] for snap in r['snapshots'] for m in study.prior.LOADS}
    expected |= {('fitting', r['seed'], r['arm'], r['snapshots'][-1]['step'], f'{split}_{m}')
                 for r in report['runs'] for split in ('validation', 'test') for m in study.prior.LOADS}
    identities = [(c['stage'], c['seed'], c['arm'], c['step'], c['pool']) for c in scores]
    assert len(identities) == len(expected) and set(identities) == expected
    assert len(report['cells']) == len(scores)
    targets = 0
    max_logit = 0.
    for cell, recorded in zip(scores, report['cells']):
        compare(recorded, {k: v for k, v in cell.items() if k not in ('logits', 'labels', 'repeat')})
        with torch.random.fork_rng(devices=[]):
            result = numerical.reference.audit_cell(study.ROOT/cell['checkpoint'], pools[cell['pool']], cell)
        targets += result['targets']
        max_logit = max(max_logit, result['max_logit_error'])
    for run in report['warmups']+report['runs']:
        for snap in run['snapshots']:
            metrics = {str(m): next(c['metrics'] for c in scores if
                       (c['stage'], c['seed'], c['arm'], c['step'], c['pool']) ==
                       (run['stage'], run['seed'], run['arm'], snap['step'], f'train_{m}')) for m in study.prior.LOADS}
            compare(snap['training'], metrics)
            nonempty = [g for v in metrics.values() for g in v['groups'].values() if g['count']]
            assert snap['fit'] == all(g['accuracy'] >= .99 for g in nonempty)
            assert snap['perfect'] == all(g['accuracy'] == 1 for g in nonempty)
    choice = {a: {key: all(r['snapshots'][-1][metric] for r in report['runs'] if r['arm'] == a)
                  for key, metric in (('all_seed_fit', 'fit'), ('all_seed_perfect', 'perfect'))}
              for a in ('lbfgs', 'adam')}
    compare(report['decision'], choice)
    compare(json.loads((out/'pretest_decision.json').read_text()), choice)
    assert targets == report['evaluation_targets']
    for stage, runs in (('warmup', report['warmups']), ('fitting', report['runs'])):
        expected_counts = {field: sum(r[field] for r in runs) for field in ('optimizer_calls', 'histories', 'input_events', 'answer_targets')}
        expected_counts['gradient_evaluations'] = sum(r['closure_evaluations'] if r['arm'] == 'lbfgs' else r['optimizer_calls'] for r in runs)
        compare(report['counts'][stage], expected_counts)
    replay.check_files(lock['files'])
    result = dict(passed=True, report_sha256=replay.digest(out/'report.json'),
                  cells=len(scores), evaluation_targets=targets, families=families,
                  counts=report['counts'], protected_files=len(lock['files']),
                  max_gradient_error=max(x['max_gradient_error'] for x in checks),
                  max_adam_error=max(x.get('max_update_error', 0.) for x in checks),
                  max_direction_error=max(x.get('max_direction_error', 0.) for x in checks),
                  max_logit_error=max_logit)
    if persist:
        assert not (out/'audit.json').exists()
        replay.write(out/'audit.json', result)
    return result


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    print(json.dumps(audit(study.OUT/'study')), flush=True)
