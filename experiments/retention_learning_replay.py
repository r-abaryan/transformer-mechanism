"""Passive observations around the immutable original training loop."""
import argparse
import copy
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch
import objective_beta_pressure as pressure

parent = pressure.parent
ROOT = pressure.ROOT
OUT = ROOT/'results/retention_learning_replay'
PROBES = (0, 10, 100, 300, 1000, 2000, 3000)
RealAdam = torch.optim.Adam
digest = parent.study.parent.digest
write = parent.study.parent.write


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def summarize(cells):
    pair = {}
    result = {}
    for objective in pressure.OBJECTIVES:
        selected = [c for c in cells if c['objective'] == objective]
        gradients = {n:torch.stack([c['gradients'][n] for c in selected]).mean(0)
                     for n in selected[0]['gradients']}
        pair[objective] = gradients
        result[objective] = dict(ce=sum(c['metrics']['all']['ce'] for c in selected)/4,
            accuracy=sum(c['metrics']['all']['accuracy'] for c in selected)/4,
            alpha_gradient=float(gradients['write_logit']))
    return dict(objectives=result, comparison=pressure.compare(*pair.values()))


class Observer:
    def __init__(self, folder, seed, arm, payload, probes=PROBES):
        self.folder, self.seed, self.arm = folder, seed, arm
        self.payload, self.probes = payload, probes
        self.steps, self.snapshots = [], []
        self.model = None

    def probe(self, optimizer, step):
        model = self.model
        before = copy.deepcopy(model.state_dict())
        opt_before = copy.deepcopy(optimizer.state_dict())
        rng = torch.get_rng_state().clone()
        grads = {n:None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()}
        cells = []
        for load in pressure.LOADS:
            for objective in pressure.OBJECTIVES:
                cells.append(dict(load=load, objective=objective,
                    **pressure.evaluate(model, self.payload['pools'][f'train_{load}'], objective)))
        assert equal(before, model.state_dict()) and equal(opt_before, optimizer.state_dict())
        assert torch.equal(rng, torch.get_rng_state()), 'Probe consumed global RNG'
        assert equal(grads, {n:p.grad for n,p in model.named_parameters()}), 'Probe changed training gradients'
        path = self.folder/f'snapshot_{self.seed}_{self.arm}_{step}.pt'
        assert not path.exists()
        torch.save(dict(model=before, optimizer=opt_before, step=step, cells=cells), path)
        record = dict(step=step, beta=float(model.write_logit.sigmoid().detach()),
                      checkpoint=path.relative_to(ROOT).as_posix(), sha256=digest(path), **summarize(cells))
        self.snapshots.append(record)
        write(self.folder/f'trajectory_{self.seed}_{self.arm}.json', dict(steps=self.steps, snapshots=self.snapshots))
        print(json.dumps(dict(seed=self.seed, arm=self.arm, probe=step, beta=record['beta'])), flush=True)

    def optimizer(self, params, **kwargs):
        observer = self

        class ObservedAdam(RealAdam):
            def __init__(self, params, **kwargs):
                super().__init__(params, **kwargs)
                if 0 in observer.probes:
                    observer.probe(self, 0)

            def step(self, closure=None):
                assert closure is None
                p = observer.model.write_logit
                alpha_before, beta_before = float(p.detach()), float(p.sigmoid().detach())
                gradient = float(p.grad)
                outcome = super().step()
                s = self.state[p]
                step = int(s['step'])
                observer.steps.append(dict(step=step, load=pressure.LOADS[(step-1)%4],
                    alpha_before=alpha_before, alpha_after=float(p.detach()), beta_before=beta_before,
                    beta_after=float(p.sigmoid().detach()), gradient=gradient,
                    exp_avg=float(s['exp_avg']), exp_avg_sq=float(s['exp_avg_sq'])))
                if step in observer.probes:
                    observer.probe(self, step)
                return outcome

        return ObservedAdam(params, **kwargs)


def replay_run(folder, seed, arm, payload, report, saved, probes=PROBES, steps=3000):
    observer = Observer(folder, seed, arm, payload, probes)
    original_pair = parent.study.pair

    def pair(seed):
        models = original_pair(seed)
        observer.model = models['delta']
        return models

    with patch.object(parent, 'OUT', folder), patch.object(parent, 'STEPS', steps), \
         patch.object(parent.study, 'pair', pair), patch.object(torch.optim, 'Adam', observer.optimizer):
        parent.train_run('development', seed, arm, payload, report, saved)
    write(folder/f'trajectory_{seed}_{arm}.json', dict(steps=observer.steps, snapshots=observer.snapshots))
    return observer


def protected_files():
    lock = json.loads((pressure.OUT/'lock.json').read_text())
    files = dict(lock['files'])
    for name in ('lock.json', 'report.json', 'scores.pt', 'audit.json'):
        p = pressure.OUT/name
        files[p.relative_to(ROOT).as_posix()] = digest(p)
    for name in ('profiles.json', 'scores.pt'):
        p = parent.OUT/name
        files[p.relative_to(ROOT).as_posix()] = digest(p)
    return files


def check_files(files):
    for name, sha in files.items():
        assert digest(ROOT/name) == sha, f'Changed frozen evidence: {name}'


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not OUT.exists(), 'Preserve completed and failed attempts'
    preflight_path = ROOT/'results/retention_learning_replay_preflight.json'
    preflight = json.loads(preflight_path.read_text())
    assert preflight['passed']
    for name, sha in preflight['sources'].items():
        assert digest(ROOT/name) == sha
    files = protected_files()
    for name in (*preflight['sources'], 'docs/protocols/retention-learning-replay.md',
                 preflight_path.relative_to(ROOT).as_posix()):
        files[name] = digest(ROOT/name)
    check_files(files)
    old = json.loads((parent.OUT/'report.json').read_text())
    payload = json.loads((parent.OUT/'development_data.json').read_text())
    assert len(old['runs']) == 4
    OUT.mkdir()
    write(OUT/'lock.json', dict(files=files, probes=PROBES, updates_per_run=3000,
                              batch=64, seeds=[3461,3463], arms=list(parent.ARMS)))
    report = dict(status='running', runs=[], reproduction=True, independent_seeds=2,
                  device='cpu', dtype='float64', threads=1, probes=PROBES,
                  peak_cuda_allocated_bytes=None, peak_cuda_reserved_bytes=None,
                  test_access=False, endpoint_checks=[])
    saved = {}
    started = time.perf_counter()
    for historical in old['runs']:
        seed, arm = historical['seed'], historical['arm']
        replay_run(OUT, seed, arm, payload, report, saved)
        item = report['runs'][-1]
        before = torch.load(ROOT/historical['checkpoint'], weights_only=True)
        after = torch.load(ROOT/item['checkpoint'], weights_only=True)
        assert equal(before, after), 'Reproduction endpoint mismatch; stop without tuning'
        for field in ('initial_sha256', 'history_stream_sha256', 'potential_query_draws_sha256',
                      'actual_queries_labels_sha256', 'exposure', 'curve', 'metrics', 'beta'):
            assert equal(item[field], historical[field]), f'Reproduction mismatch: {field}'
        report['endpoint_checks'].append(dict(seed=seed, arm=arm, model_optimizer_exact=True,
                                             streams_exposure_metrics_exact=True))
        write(OUT/'report.json', report)
    torch.save(saved, OUT/'final_scores.pt')
    report.update(status='complete', total_updates=12000, examples=768000,
                  input_events=24576000, elapsed_seconds=time.perf_counter()-started,
                  final_scores_sha256=digest(OUT/'final_scores.pt'), lock_sha256=digest(OUT/'lock.json'))
    check_files(files)
    write(OUT/'report.json', report)
    print('Reproduction complete, all four endpoints exact. Run independent audit.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        import importlib.util
        import unittest
        path = ROOT/'tests/test_retention_learning_replay.py'
        spec = importlib.util.spec_from_file_location('test_retention_learning_replay',path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
        names = ('experiments/retention_learning_replay.py','experiments/audit_retention_learning_replay.py',
                 'tests/test_retention_learning_replay.py')
        record = dict(passed=result.wasSuccessful(), methods=result.testsRun,
                      sources={n:digest(ROOT/n) for n in names})
        write(ROOT/'results/retention_learning_replay_preflight.json', record)
        assert record['passed']
    else:
        run()
