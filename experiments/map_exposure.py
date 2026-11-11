"""Matched Adam continuation with half-new high-load value maps."""
import argparse
import copy
import json
import random
import time

import torch
from torch.nn import functional as F
import association_load as data_source
import full_pool_fitting as prior

ROOT = prior.ROOT
OUT = ROOT/'results/map_exposure'
SOURCE = ROOT/'results/full_pool_fitting_confirmation/study'
POLICIES = ('reuse', 'half_new')
SEEDS = (3473, 3479, 3481)
DATA_SEED = 2026100529
SOURCES = ('experiments/map_exposure.py', 'experiments/audit_map_exposure.py', 'tests/test_map_exposure.py')
PROTOCOL = 'docs/protocols/map-exposure.md'
replay = prior.prior.training.replay
training = prior.prior.training


def map_family(events):
    items = sorted(dict(events).items())
    return min(tuple((k, (v+s) % 6) for k, v in items) for s in range(6))


def inputs():
    report = json.loads((SOURCE/'report.json').read_text())
    cases = []
    for seed in SEEDS:
        old = next(r for r in report['runs'] if r['seed'] == seed and r['arm'] == 'adam')
        cases.append(dict(seed=seed, checkpoint=f'results/full_pool_fitting_confirmation/study/warmup/{seed}_adam_12000.pt',
                          closures=old['optimizer_calls']//48, historical=old))
    return cases, torch.load(SOURCE/'pools.pt', weights_only=True)


def build_data(original, seed, fixture=False):
    rng = random.Random(seed)
    broader = copy.deepcopy(original)
    rows = original['train_32']; assert len(rows) % 12 == 0
    excluded_maps = {map_family(r['events']) for pool in original.values() for r in pool if r['load'] == 32}
    blocks = rng.sample(range(len(rows)//6), len(rows)//12)
    for block in blocks:
        old = rows[6*block]; keys = [k for k, v in old['events']]
        for _ in range(10000):
            values = {k: rng.randrange(6) for k in keys}
            events = [[k, values[k]] for k in keys]
            if values[old['query']] != old['label'] and map_family(events) not in excluded_maps: break
        else: raise RuntimeError('New mapping generation exhausted')
        excluded_maps.add(map_family(events))
        for shift in range(6):
            row = copy.deepcopy(rows[6*block+shift])
            row['events'] = [[k, (values[k]+shift) % 6] for k in keys]
            row['label'] = (values[row['query']]+shift) % 6
            broader['train_32'][6*block+shift] = row
    excluded_histories = {data_source.family(r['events']) for pools in (original, broader) for pool in pools.values() for r in pool}
    counts = dict(train=0, validation=1 if fixture else 32, test=1 if fixture else 64)
    fresh = data_source.generate(seed+1, confirmation=True, exclude=excluded_histories, base_counts=counts)['pools']
    fresh = {p: rows for p, rows in fresh.items() if rows}
    assert all(map_family(r['events']) not in excluded_maps for p, rows in fresh.items() if p.endswith('_32') for r in rows)
    active = {policy: {**{p: rows for p, rows in source.items() if p.startswith('train_')}, **fresh}
              for policy, source in [('reuse', original), ('half_new', broader)]}
    evaluation = {**fresh}
    evaluation.update({'seen_'+p: rows for p, rows in original.items() if p.startswith('train_')})
    evaluation.update({'reference_'+p: rows for p, rows in original.items() if not p.startswith('train_')})
    evaluation = {p: rows for p, rows in evaluation.items() if not p.startswith('train_')}
    return dict(seed=seed, selected_blocks=sorted(blocks), original=original, active=active, evaluation=evaluation)


def decision(cells):
    pairs = []
    for seed in sorted({c['seed'] for c in cells}):
        end = max(c['step'] for c in cells if c['seed'] == seed)
        def get(policy, pool):
            return next(c['metrics'] for c in cells if (c['seed'], c['policy'], c['step'], c['pool']) == (seed, policy, end, pool))
        fit = {p: prior.fit({str(m): get(p, f'train_{m}') for m in prior.LOADS}) for p in POLICIES}
        retained = {p: prior.fit({str(m): get(p, f'seen_train_{m}') for m in prior.LOADS}) for p in POLICIES}
        x, y = [get(p, 'test_32') for p in POLICIES]
        gain = y['groups']['all']['accuracy']-x['groups']['all']['accuracy']
        pairs.append(dict(seed=seed, active_training_fit=fit, original_training_retained=retained,
                          test32_accuracy_gain=gain, test32_ce_delta=y['groups']['all']['ce']-x['groups']['all']['ce'],
                          test32_map_delta=y['complete_maps']['correct']-x['complete_maps']['correct'],
                          exposure_benefit=gain >= .005 and all(fit.values()),
                          repair=gain >= .005 and all(fit.values()) and retained['half_new']))
    return dict(pairs=pairs, all_seed_active_fit=all(all(p['active_training_fit'].values()) for p in pairs),
                all_seed_exposure_benefit=all(p['exposure_benefit'] for p in pairs),
                all_seed_repair=all(p['repair'] for p in pairs))


def extra_scores(model, payload, out, case, policy, step, scores):
    checkpoint = out/f"{case['seed']}_adam_{step}.pt"
    for pool, rows in payload['evaluation'].items():
        cell = training.evaluate(model, rows)
        cell.update(seed=case['seed'], arm='adam', policy=policy, step=step, pool=pool,
                    checkpoint=checkpoint.relative_to(ROOT).as_posix())
        scores.append(cell)


def execute(out, cases, payload, ratio=48, interval=32, fixture=False):
    assert not out.exists(), 'Preserve evidence'
    files = {p: replay.digest(ROOT/p) for p in prior.dependencies() | set(SOURCES)}
    for case in cases:
        files[case['checkpoint']] = replay.digest(ROOT/case['checkpoint'])
        if 'historical' in case:
            for key in ('final_checkpoint', 'trace'): files[case['historical'][key]] = replay.digest(ROOT/case['historical'][key])
    if not fixture:
        expected, original = inputs()
        assert cases == expected and payload['original'] == original and (ratio, interval) == (48, 32)
        assert payload == build_data(original, DATA_SEED)
        for p in (PROTOCOL, 'results/map_exposure_preflight.json', 'results/map_exposure_profiles.json',
                  'results/full_pool_fitting_confirmation/study/report.json', 'results/full_pool_fitting_confirmation/study/audit.json'):
            files[p] = replay.digest(ROOT/p)
        pre = json.loads((ROOT/'results/map_exposure_preflight.json').read_text())
        assert pre['passed'] and pre['sources'] == {p: files[p] for p in SOURCES}
        assert json.loads((ROOT/'results/map_exposure_profiles.json').read_text())['sources'] == pre['sources']
    replay.check_files(files); out.mkdir(parents=True)
    torch.save(payload, out/'data.pt')
    replay.write(out/'lock.json', dict(files=files, cases=cases, ratio=ratio, interval=interval, fixture=fixture,
                 data_sha256=replay.digest(out/'data.pt'), torch=torch.__version__))
    runs = []; scores = []; started = time.perf_counter()
    for case in cases:
        for policy in POLICIES:
            folder = out/policy; folder.mkdir(exist_ok=True)
            local = []
            run = prior.train_adam(case, payload['active'][policy], folder, local, case['closures'], ratio, interval)
            run.update(policy=policy, stop_reason='fixed_matched_continuation_budget')
            for cell in local: cell['policy'] = policy
            scores.extend(local)
            # These are reporting probes only; all training settings were locked.
            for step, cp in [(0, run['snapshots'][0]['checkpoint']), (run['optimizer_calls'], run['final_checkpoint'])]:
                with torch.random.fork_rng(devices=[]): model = training.previous.load_model(ROOT/cp)
                extra_scores(model, payload, folder, case, policy, step, scores)
            runs.append(run)
            replay.write(out/'progress.json', dict(completed_runs=len(runs), total_runs=2*len(cases)))
    torch.save(scores, out/'scores.pt')
    replay.check_files(files)
    report = dict(status='complete', runs=runs, decision=decision(scores),
        cells=[{k: v for k, v in c.items() if k not in ('logits', 'labels', 'repeat')} for c in scores],
        evaluation_targets=sum(c['labels'].numel() for c in scores),
        optimizer_calls=sum(r['optimizer_calls'] for r in runs), gradient_evaluations=sum(r['optimizer_calls'] for r in runs),
        answer_targets=sum(r['answer_targets'] for r in runs), histories=sum(r['histories'] for r in runs),
        input_events=sum(r['input_events'] for r in runs), seconds=time.perf_counter()-started,
        device='cpu', dtype='float64', threads=1, workers=1, peak_allocated_bytes=None, peak_reserved_bytes=None,
        selection='none_fixed_budget', fresh_test=not fixture, historical_seeds=True,
        lock_sha256=replay.digest(out/'lock.json'), scores_sha256=replay.digest(out/'scores.pt'))
    replay.write(out/'report.json', report)
    print(json.dumps(dict(status='complete', decision=report['decision'])), flush=True)


def profile():
    cases, original = inputs(); payload = build_data(original, DATA_SEED)
    records = []
    for policy in POLICIES:
        model, opt, start = prior.load(ROOT/cases[0]['checkpoint'])
        e, q, y, _ = prior.tables(payload['active'][policy])[32]
        before = time.perf_counter(); opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(training.forward_queries(model, e[:64], q[:64]).flatten(0,1), y[:64].flatten())
        loss.backward(); opt.step()
        state_bytes = sum(v.numel()*v.element_size() for state in opt.state.values() for v in state.values() if torch.is_tensor(v))
        records.append(dict(policy=policy, seconds=time.perf_counter()-before, batch=64, events=2048,
                            answers=2048, optimizer_state_bytes=state_bytes, optimizer_state_present=bool(opt.state),
                            peak_allocated_bytes=None, peak_reserved_bytes=None, cpu_peak_memory_measured=False))
    return dict(records=records, sources={p: replay.digest(ROOT/p) for p in SOURCES},
                scope='Two discarded complete CPU training steps; not formal exposure or speed measurement.')


if __name__ == '__main__':
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser(); parser.add_argument('--profile', action='store_true'); args = parser.parse_args()
    if args.profile:
        path = ROOT/'results/map_exposure_profiles.json'; assert not path.exists(); replay.write(path, profile())
    else:
        cases, original = inputs(); execute(OUT, cases, build_data(original, DATA_SEED))
