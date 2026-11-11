"""Fixed-length association load, reusing the audited compact reader equations."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import random
import struct
import sys
import time
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import delta_retrieval as parent

ROOT = parent.ROOT
OUT = ROOT / 'results/association_load'
LOADS = (2, 8, 16, 32)
LENGTH, KEYS, BATCH, STEPS = 32, 32, 64, 3000
DEV_SEEDS = (3313, 3319)
CONFIRM_SEEDS = (3323, 3329, 3331)
RECIPES = ('constant', 'cosine')
BASE_COUNTS = dict(train=128, validation=32, test=64)


class LoadReader(parent.Reader):
    def __init__(self, arm):
        super().__init__(arm)
        self.key = nn.Embedding(KEYS, 16, dtype=torch.float64)
        if arm == 'softmax':
            # Preserve the existing forward equation, with no trainable recency.
            del self.log_recency
            self.register_buffer('log_recency', torch.tensor(-torch.inf, dtype=torch.float64))


def pair(seed):
    torch.manual_seed(seed)
    first, second = LoadReader('softmax'), LoadReader('delta')
    second.load_state_dict({k: v for k, v in first.state_dict().items()
                            if k.startswith(('key.', 'value.', 'output.'))}, strict=False)
    assert all(sum(p.numel() for p in m.parameters()) == 615 for m in (first, second))
    return dict(softmax=first, delta=second)


def family(events):
    return min(tuple((k, (v + s) % 6) for k, v in events) for s in range(6))


def generate(seed, confirmation=False, exclude=(), base_counts=None):
    rng = random.Random(seed)
    counts = BASE_COUNTS if base_counts is None else base_counts
    seen = set(exclude)
    pools = {}
    for split in ('train', 'validation', 'test') if confirmation else ('train', 'validation'):
        for load in LOADS:
            rows = []
            queries = [i % KEYS for i in range(counts[split])]
            rng.shuffle(queries)
            for q in queries:
                for _ in range(10000):
                    keys = [q] + rng.sample([k for k in range(KEYS) if k != q], load - 1)
                    values = {k: rng.randrange(6) for k in keys}
                    events = [[k, values[k]] for k in keys for _ in range(LENGTH // load)]
                    rng.shuffle(events)
                    identity = family(events)
                    if identity not in seen:
                        break
                else:
                    raise RuntimeError('Family generation exhausted')
                seen.add(identity)
                for shift in range(6):
                    rows.append(dict(events=[[k, (v + shift) % 6] for k, v in events], query=q,
                                     label=(values[q] + shift) % 6, load=load,
                                     group='unique' if load == LENGTH else 'repeat'))
            pools[f'{split}_{load}'] = rows
    return dict(seed=seed, length=LENGTH, keys=KEYS, base_counts=counts, pools=pools)


def audit_data(payload):
    ownership, baselines = {}, {}
    for name, rows in payload['pools'].items():
        assert rows
        loads = {r['load'] for r in rows}
        assert len(loads) == 1
        load = next(iter(loads))
        count = dict(count=len(rows), constant=0, last_event=0, first_match=0)
        query_counts, labels = {}, [0] * 6
        for row in rows:
            events, q = row['events'], row['query']
            memory, hits = {}, {}
            for k, v in events:
                assert 0 <= k < KEYS and 0 <= v < 6
                if k in memory:
                    assert memory[k] == v, 'Conflicting assignment'
                memory[k] = v; hits[k] = hits.get(k, 0) + 1
            assert len(events) == LENGTH and len(memory) == load
            assert set(hits.values()) == {LENGTH // load}, 'Unequal multiplicities'
            assert memory[q] == row['label'], 'Wrong label'
            assert row['group'] == ('unique' if load == LENGTH else 'repeat')
            f = family(events)
            assert ownership.setdefault(f, name) == name, 'Family leakage'
            count['constant'] += row['label'] == 0
            count['last_event'] += events[-1][1] == row['label']
            count['first_match'] += next(v for k, v in events if k == q) == row['label']
            labels[row['label']] += 1
            query_counts[q] = query_counts.get(q, 0) + 1
        assert len(set(labels)) == 1, 'Unbalanced labels'
        assert set(query_counts) == set(range(KEYS)) and len(set(query_counts.values())) == 1, 'Unbalanced queries'
        assert count['first_match'] == len(rows) and count['constant'] * 6 == len(rows)
        baselines[name] = count
    return baselines


def dependencies(protocol=True):
    paths = ['experiments/association_load.py', 'experiments/audit_association_load.py',
             'tests/test_association_load.py', 'experiments/delta_retrieval.py',
             'experiments/audit_delta_retrieval.py']
    if protocol:
        paths.append('docs/protocols/association-load.md')
    return {p: parent.digest(ROOT / p) for p in paths}


def select(runs):
    output = {}
    for arm in parent.ARMS:
        candidates = []
        for recipe in RECIPES:
            subset = [r for r in runs if r['arm'] == arm and r['recipe'] == recipe]
            complete = len(subset) == len(DEV_SEEDS) and {r['seed'] for r in subset} == set(DEV_SEEDS)
            complete &= all(r['status'] == 'complete' for r in subset)
            fit = complete and all(r['metrics']['train_2']['all']['accuracy'] >= .99 for r in subset)
            worst = max((r['metrics'][f'train_{m}']['all']['loss']
                         for r in subset if r['status'] == 'complete' for m in LOADS), default=None)
            candidates.append(dict(recipe=recipe, low_load_fit=fit, worst_training_loss=worst))
        eligible = [c for c in candidates if c['low_load_fit']]
        chosen = min(eligible, key=lambda c: (c['worst_training_loss'], RECIPES.index(c['recipe'])))['recipe'] if eligible else None
        output[arm] = dict(selected_recipe=chosen, candidates=candidates)
    return output


def decision(runs):
    out = {}
    for arm in parent.ARMS:
        subset = [r for r in runs if r['phase'] == 'confirmation' and r['arm'] == arm]
        assert len(subset) == 3 and {r['seed'] for r in subset} == set(CONFIRM_SEEDS)
        out[arm] = dict(loads={str(m): dict(
            all_seed_training_fit=all(r['metrics'][f'train_{m}']['all']['accuracy'] >= .99 for r in subset),
            all_seed_test_reliability=all(r['metrics'][f'test_{m}']['all']['accuracy'] >= .95 for r in subset)) for m in LOADS},
            low_load_control=all(r['metrics']['train_2']['all']['accuracy'] >= .99 and
                                 r['metrics']['test_2']['all']['accuracy'] >= .95 for r in subset),
            all_seed_10_point_load_drop=all(r['metrics']['test_2']['all']['accuracy'] -
                                          r['metrics']['test_32']['all']['accuracy'] >= .10 for r in subset))
    return out


def probes(model, rows):
    e, q, y = parent.tensors(rows)
    added = torch.tensor([[next((k, v) for k, v in r['events'] if k != r['query'])] for r in rows])
    changed = torch.cat((e, added), 1)
    with torch.no_grad():
        before = torch.cat([model(a, b) for a, b in zip(e.split(128), q.split(128))])
        after = torch.cat([model(a, b) for a, b in zip(changed.split(128), q.split(128))])
        shift = None
        if model.arm == 'delta':
            state = model.prefill(e)[0]
            new_state = model.prefill(changed)[0]
            k, query = model.keys(added[:, 0, 0]), model.keys(q)
            residual = model.value(added[:, 0, 1]) - (k[:, :, None] * state).sum(1)
            shift = model.write_logit.sigmoid() * (query * k).sum(-1)[:, None] * residual
            actual = (query[:, :, None] * (new_state - state)).sum(1)
            assert torch.allclose(actual, shift, atol=1e-10, rtol=1e-10), 'Cross-write equation failed'
            assert torch.allclose(after - before, shift @ model.output.weight.T, atol=1e-10, rtol=1e-10)
    a, b = before.argmax(-1) == y, after.argmax(-1) == y
    summary = dict(count=len(rows), harmed=int((a & ~b).sum()), rescued=int((~a & b).sum()),
                   before_correct=int(a.sum()), after_correct=int(b.sum()),
                   mean_logit_shift_l2=float((after - before).norm(dim=-1).mean()),
                   after_metrics=parent.metrics(after, rows))
    if shift is not None:
        summary.update(mean_read_shift_l2=float(shift.norm(dim=-1).mean()),
                       mean_absolute_query_write_dot=float((query * k).sum(-1).abs().mean()))
    return summary, dict(before=before, after=after)


def preflight():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=False)
    prior = json.loads((ROOT / 'results/retrieval_gpu_resources/preserved.json').read_text())
    for folder in (ROOT / 'results/retrieval_gpu_resources',):
        prior.update({str(p.relative_to(ROOT)).replace('\\', '/'): parent.digest(p) for p in folder.iterdir() if p.is_file()})
    for p in ('experiments/retrieval_gpu_resources.py', 'experiments/audit_retrieval_gpu_resources.py',
              'tests/test_retrieval_gpu_resources.py', 'docs/protocols/retrieval-gpu-resources.md',
              'docs/topics/attention/retrieval-gpu-resources-results.md', 'docs/figures/retrieval-gpu-resources.svg'):
        prior[p] = parent.digest(ROOT / p)
    assert all(parent.digest(ROOT / p) == h for p, h in prior.items())
    parent.write(OUT / 'preserved.json', prior)
    spec = importlib.util.spec_from_file_location('test_association_load', ROOT / 'tests/test_association_load.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    tests = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    assert tests.wasSuccessful(), 'Preflight tests failed'
    profile_rows = generate(2026100431)['pools']['train_32'][:BATCH]
    e, q, y = parent.tensors(profile_rows)
    profiles = []
    for device in ('cpu', 'cuda'):
        assert device != 'cuda' or torch.cuda.is_available()
        if device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        for arm, model in pair(DEV_SEEDS[0]).items():
            model.to(device)
            a, b, c = e.to(device), q.to(device), y.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=.003, foreach=False)
            if device == 'cuda':
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            loss = F.cross_entropy(model(a, b), c); loss.backward(); optimizer.step()
            if device == 'cuda':
                torch.cuda.synchronize()
            record = dict(arm=arm, device=device, dtype='float64', batch=BATCH, length=LENGTH,
                          updates=1, targets=BATCH, input_events=BATCH * LENGTH,
                          elapsed_seconds=time.perf_counter() - tick,
                          parameters=sum(p.numel() for p in model.parameters()), optimizer_states=len(optimizer.state))
            if device == 'cuda':
                record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(), device_name=torch.cuda.get_device_name())
            profiles.append(record)
            del model, optimizer, loss, a, b, c
            if device == 'cuda':
                torch.cuda.empty_cache()
    parent.write(OUT / 'preflight.json', dict(passed=True, methods=tests.testsRun, profiles=profiles,
                 source_dependencies=dependencies(False), disposable_optimizer_steps=4))


def train_run(phase, seed, arm, recipe, payload, report):
    model = pair(seed)[arm]
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, foreach=False)
    data = {m: parent.tensors(payload['pools'][f'train_{m}']) for m in LOADS}
    rng, stream = torch.Generator().manual_seed(seed + 10000), hashlib.sha256()
    shared = {k: v for k, v in model.state_dict().items() if k.startswith(('key.', 'value.', 'output.'))}
    run = dict(phase=phase, seed=seed, arm=arm, recipe=recipe, status='running', metrics={}, curve=[],
               parameters=sum(p.numel() for p in model.parameters()), initial_sha256=parent.state_hash(model.state_dict()),
               shared_sha256=parent.state_hash(shared))
    report['runs'].append(run); parent.write(OUT / 'report.json', report)
    tick = time.perf_counter()
    for step in range(1, STEPS + 1):
        load = LOADS[(step - 1) % len(LOADS)]
        if recipe == 'cosine':
            optimizer.param_groups[0]['lr'] = .003 * (.1 + .9 * .5 * (1 + math.cos(math.pi * (step - 1) / (STEPS - 1))))
        e, q, y = data[load]
        idx = torch.randint(len(e), (BATCH,), generator=rng)
        stream.update(struct.pack('<I', load)); stream.update(bytes(idx.view(torch.uint8).tolist()))
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(e[idx], q[idx]), y[idx])
        assert torch.isfinite(loss), 'Nonfinite training loss'
        loss.backward(); optimizer.step()
        if step % 1000 == 0:
            run['curve'].append(dict(step=step, load=load, loss=float(loss.detach()), lr=optimizer.param_groups[0]['lr']))
            print(dict(phase=phase, seed=seed, arm=arm, recipe=recipe, step=step, loss=float(loss.detach())), flush=True)
            parent.write(OUT / 'report.json', report)
    cp = OUT / f'{phase}_{seed}_{arm}_{recipe}.pt'
    assert not cp.exists()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=STEPS), cp)
    for name, rows in payload['pools'].items():
        if not name.startswith('test_'):
            run['metrics'][name], _ = parent.evaluate(model, rows)
    run.update(status='complete', updates=STEPS, examples=STEPS * BATCH, input_events=STEPS * BATCH * LENGTH,
               examples_per_load=STEPS // len(LOADS) * BATCH, stream_sha256=stream.hexdigest(),
               elapsed_seconds=time.perf_counter() - tick, checkpoint=str(cp.relative_to(ROOT)).replace('\\', '/'),
               checkpoint_sha256=parent.digest(cp))
    parent.write(OUT / 'report.json', report)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('preflight', 'train'))
    args = parser.parse_args()
    if args.phase == 'preflight':
        preflight(); return
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    assert not (OUT / 'report.json').exists()
    checked = json.loads((OUT / 'preflight.json').read_text())
    assert checked['passed'] and checked['source_dependencies'] == dependencies(False)
    source = dependencies()
    parent.write(OUT / 'lock.json', dict(source_dependencies=source, preflight_sha256=parent.digest(OUT / 'preflight.json'),
                                       loads=LOADS, length=LENGTH, updates=STEPS, batch=BATCH))
    dev = generate(2026100431); baselines = {'development': audit_data(dev)}
    parent.write(OUT / 'development_data.json', dev)
    report = dict(status='development', source_dependencies=source, device='cpu', dtype='float64', threads=1,
                  python=sys.version, torch=torch.__version__, runs=[], baselines=baselines,
                  data_sha256={'development': parent.digest(OUT / 'development_data.json')})
    for recipe, seed in itertools.product(RECIPES, DEV_SEEDS):
        for arm in parent.ARMS if seed == DEV_SEEDS[0] else parent.ARMS[::-1]:
            train_run('development', seed, arm, recipe, dev, report)
    report['selection'] = select(report['runs'])
    parent.write(OUT / 'development_locked.json', report)
    if all(c['selected_recipe'] is not None for c in report['selection'].values()):
        blocked = {family(r['events']) for rows in dev['pools'].values() for r in rows}
        fresh = generate(2026100437, confirmation=True, exclude=blocked)
        audit_data(dict(pools={**{'dev_' + p: v for p, v in dev['pools'].items()},
                              **{'confirm_' + p: v for p, v in fresh['pools'].items()}}))
        report['baselines']['confirmation'] = audit_data(fresh)
        parent.write(OUT / 'confirmation_data.json', fresh)
        report['data_sha256']['confirmation'] = parent.digest(OUT / 'confirmation_data.json')
        for seed in CONFIRM_SEEDS:
            for arm in parent.ARMS if seed != CONFIRM_SEEDS[1] else parent.ARMS[::-1]:
                train_run('confirmation', seed, arm, report['selection'][arm]['selected_recipe'], fresh, report)
        parent.write(OUT / 'pretest.json', report)
        saved_probes = {}
        for run in [r for r in report['runs'] if r['phase'] == 'confirmation']:
            model = LoadReader(run['arm']); model.load_state_dict(torch.load(ROOT / run['checkpoint'], weights_only=True)['model'])
            run['probes'] = {}
            key = f"{run['seed']}_{run['arm']}"
            saved_probes[key] = {}
            for load in LOADS:
                name = f'test_{load}'; rows = fresh['pools'][name]
                run['metrics'][name], _ = parent.evaluate(model, rows)
                run['probes'][name], saved_probes[key][name] = probes(model, rows)
        torch.save(saved_probes, OUT / 'probe_scores.pt')
        report['probe_scores_sha256'] = parent.digest(OUT / 'probe_scores.pt')
        report['confirmation_decision'] = decision(report['runs'])
    else:
        report['confirmation_skipped'] = 'At least one arm has no eligible low-load development recipe.'
    assert dependencies() == source
    assert all(parent.digest(ROOT / p) == h for p, h in json.loads((OUT / 'preserved.json').read_text()).items())
    report['status'] = 'complete'; parent.write(OUT / 'report.json', report)
    print('Bounded training complete; independent audit required.', flush=True)


if __name__ == '__main__':
    main()
