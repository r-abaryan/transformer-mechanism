"""Fresh confirmation with fixed warm-up and unchanged fitting procedures."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import inspect
import json
import time
from pathlib import Path

import torch
import torch.optim.lbfgs as lbfgs_source
from torch.nn import functional as F

import association_load as data_source
import full_pool_fitting as prior

ROOT = prior.ROOT
OUT = ROOT / 'results/full_pool_fitting_confirmation'
SEEDS = (3473, 3479, 3481)
DATA_SEED = 2026100517
WARM_STEPS = 12000
PROTOCOL = 'docs/protocols/full-pool-fitting-confirmation.md'
SOURCES = ('experiments/confirm_full_pool_fitting.py',
           'experiments/audit_full_pool_fitting_confirmation.py',
           'tests/test_full_pool_fitting_confirmation.py')
OLD_DATA = ('results/association_load/development_data.json',
            'results/association_load/confirmation_data.json',
            'results/delta_query_coverage/development_data.json',
            'results/query_feedback_training/test_data.json')
replay = prior.prior.training.replay


def historical_families():
    return {data_source.family(row['events']) for path in OLD_DATA
            for rows in json.loads((ROOT / path).read_text())['pools'].values()
            for row in rows}


def prepare():
    assert not OUT.exists(), 'Preserve earlier attempts'
    excluded = historical_families()
    payload = data_source.generate(DATA_SEED, confirmation=True, exclude=excluded)
    baselines = data_source.audit_data(payload)
    assert all(data_source.family(row['events']) not in excluded
               for rows in payload['pools'].values() for row in rows)
    OUT.mkdir()
    torch.save(payload, OUT / 'data.pt')
    replay.write(OUT / 'data_manifest.json', dict(
        seed=DATA_SEED, counts={k: len(v) for k, v in payload['pools'].items()},
        excluded_families=len(excluded), baselines=baselines,
        data_sha256=replay.digest(OUT / 'data.pt'),
        source_hashes={p: replay.digest(ROOT / p) for p in OLD_DATA}))


def protected():
    files = dict(json.loads((prior.OUT / 'lock.json').read_text())['files'])
    for path in (*SOURCES, *OLD_DATA,
                 'results/full_pool_fitting/report.json',
                 'results/full_pool_fitting/audit.json',
                 'results/full_pool_fitting/lock.json',
                 'docs/topics/attention/full-pool-fitting-results.md'):
        files[path] = replay.digest(ROOT / path)
    return files


def annotate(cells, stage):
    for cell in cells:
        cell['stage'] = stage


def fit_seed(out, seed, pools, warm_steps, budget, ratio, interval):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    initial = out / 'warmup' / f'{seed}_initial.pt'
    with torch.random.fork_rng(devices=[]):
        model = data_source.pair(seed)['delta']
    opt = torch.optim.Adam(model.parameters(), lr=.003, foreach=False)
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), step=0), initial)
    initial_case = dict(seed=seed, checkpoint=initial.relative_to(ROOT).as_posix())
    warm_scores = []
    warm = prior.train_adam(initial_case, pools, out / 'warmup', warm_scores,
                            warm_steps // ratio, ratio, warm_steps // ratio)
    warm.update(initial_checkpoint=initial_case['checkpoint'],
                initial_sha256=replay.digest(initial), stage='warmup')
    annotate(warm_scores, 'warmup')
    case = dict(seed=seed, checkpoint=warm['final_checkpoint'])
    fit_scores = []
    a = prior.train_lbfgs(case, pools, out, fit_scores, budget, interval)
    b = prior.train_adam(case, pools, out, fit_scores,
                         a['closure_evaluations'], ratio, interval)
    for run in (a, b):
        run['stage'] = 'fitting'
    annotate(fit_scores, 'fitting')
    return warm, (a, b), warm_scores+fit_scores


def execute(out, seeds, pools, warm_steps=WARM_STEPS, budget=prior.CLOSURES,
            ratio=prior.RATIO, interval=prior.INTERVAL, fixture=False):
    assert not out.exists(), 'Preserve earlier attempts'
    assert warm_steps > 0 and warm_steps % ratio == 0
    files = protected()
    if not fixture:
        assert tuple(seeds) == SEEDS
        assert (warm_steps, budget, ratio, interval) == (12000, 128, 48, 32)
        for path in (PROTOCOL, 'results/full_pool_fitting_confirmation/data.pt',
                     'results/full_pool_fitting_confirmation/data_manifest.json',
                     'results/full_pool_fitting_confirmation/preflight_final.json',
                     'results/full_pool_fitting_confirmation/profiles_final.json'):
            files[path] = replay.digest(ROOT / path)
        ready = json.loads((OUT / 'preflight_final.json').read_text())
        assert ready['passed'] and ready['sources'] == {p: files[p] for p in SOURCES}
        assert json.loads((OUT / 'profiles_final.json').read_text())['sources'] == ready['sources']
        assert pools == torch.load(OUT / 'data.pt', weights_only=True)['pools']
    replay.check_files(files)
    out.mkdir(parents=True)
    (out / 'warmup').mkdir()
    torch.save(pools, out / 'pools.pt')
    lock = dict(files=files, seeds=list(seeds), warm_steps=warm_steps, budget=budget,
                ratio=ratio, interval=interval, fixture=fixture,
                pools_sha256=replay.digest(out / 'pools.pt'), torch=torch.__version__,
                lbfgs_implementation_sha256=replay.digest(Path(inspect.getsourcefile(lbfgs_source))))
    replay.write(out / 'lock.json', lock)
    warmups, runs, scores = [], [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(3, len(seeds))) as workers:
        futures = [workers.submit(fit_seed, out, seed, pools, warm_steps, budget, ratio, interval) for seed in seeds]
        for future in futures:
            warm, pair, measured = future.result()
            a, b = pair
            if not fixture:
                assert (a['answer_targets'], a['histories'], a['input_events']) == (
                    b['answer_targets'], b['histories'], b['input_events'])
            warmups.append(warm)
            runs.extend(pair)
            scores.extend(measured)
            replay.write(out / 'progress.json', dict(completed_seeds=len(warmups), seeds=list(seeds)))
    training_wall_seconds = time.perf_counter()-started
    choice = prior.decision(runs)
    replay.write(out / 'pretest_decision.json', choice)
    for run in runs:
        with torch.random.fork_rng(devices=[]):
            model = prior.prior.training.previous.load_model(ROOT / run['final_checkpoint'])
        for split in ('validation', 'test'):
            for load in prior.LOADS:
                cell = prior.prior.training.evaluate(model, pools[f'{split}_{load}'])
                cell.update(stage='fitting', seed=run['seed'], arm=run['arm'],
                            step=run['snapshots'][-1]['step'], pool=f'{split}_{load}',
                            checkpoint=run['final_checkpoint'])
                scores.append(cell)
    torch.save(scores, out / 'scores.pt')
    stages = {}
    for name, records in (('warmup', warmups), ('fitting', runs)):
        stages[name] = {field: sum(r[field] for r in records)
                       for field in ('optimizer_calls', 'histories', 'input_events', 'answer_targets')}
        stages[name]['gradient_evaluations'] = sum(
            r['closure_evaluations'] if r['arm'] == 'lbfgs' else r['optimizer_calls'] for r in records)
    report = dict(status='complete', warmups=warmups, runs=runs, decision=choice,
                  cells=[{k: v for k, v in c.items() if k not in ('logits', 'labels', 'repeat')} for c in scores],
                  counts=stages, evaluation_targets=sum(c['labels'].numel() for c in scores),
                  device='cpu', dtype='float64', threads=1, selection='none_locked_confirmation',
                  fresh_test=not fixture, shared_data_across_seeds=True,
                  workers=min(3, len(seeds)), training_wall_seconds=training_wall_seconds,
                  scores_sha256=replay.digest(out / 'scores.pt'), lock_sha256=replay.digest(out / 'lock.json'),
                  pretest_sha256=replay.digest(out / 'pretest_decision.json'))
    replay.check_files(files)
    replay.write(out / 'report.json', report)
    print(json.dumps(dict(status='complete', decision=choice)), flush=True)


def profile():
    pools = torch.load(OUT / 'data.pt', weights_only=True)['pools']
    records = []
    for arm in ('adam', 'lbfgs'):
        with torch.random.fork_rng(devices=[]):
            model = data_source.pair(2026100599)['delta']
        opt = torch.optim.Adam(model.parameters(), lr=.003, foreach=False) if arm == 'adam' else prior.make_lbfgs(model)
        table = prior.tables(pools)
        calls = 0
        started = time.perf_counter()
        if arm == 'adam':
            e, q, y, _ = table[32]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(prior.prior.training.forward_queries(model, e[:64], q[:64]).flatten(0, 1), y[:64].flatten())
            loss.backward()
            opt.step()
            calls = 1
        else:
            def closure():
                nonlocal calls
                opt.zero_grad(set_to_none=True)
                loss = prior.objective(model, table)
                calls += 1
                return torch.tensor(loss, dtype=torch.float64)
            opt.step(closure)
        records.append(dict(arm=arm, seconds=time.perf_counter()-started,
                            gradient_evaluations=calls, optimizer_state_present=bool(opt.state),
                            histories_per_gradient=64 if arm == 'adam' else 3072,
                            answer_targets=calls*(2048 if arm == 'adam' else 44544),
                            peak_allocated_bytes=None, peak_reserved_bytes=None))
    path = OUT / 'profiles_final.json'
    assert not path.exists()
    replay.write(path, dict(records=records, device='cpu', dtype='float64', cpu_peak_memory='unmeasured',
                           sources={p: replay.digest(ROOT / p) for p in SOURCES}))


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'profile', 'train'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'profile':
        profile()
    else:
        execute(OUT / 'study', SEEDS, torch.load(OUT / 'data.pt', weights_only=True)['pools'])
