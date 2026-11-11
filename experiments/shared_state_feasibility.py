"""Fixed-key/head shared-memory witnesses; no training or solver retries."""
import argparse
import csv
import hashlib
import importlib.util
import json
import time
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog
import torch
import query_feedback_continuation as historical

ROOT = historical.ROOT
OUT = ROOT / 'results/shared_state_feasibility'
PROTOCOL = 'docs/protocols/shared-state-feasibility.md'
SOURCES = ('experiments/shared_state_feasibility.py',
           'experiments/audit_shared_state_feasibility.py', 'tests/test_shared_state_feasibility.py')
MARGIN = 1e-7
OPTIONS = dict(presolve=True, primal_feasibility_tolerance=1e-9,
               dual_feasibility_tolerance=1e-9, maxiter=10000, time_limit=5.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def versions():
    return dict(numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__)


def check_files(files):
    for name, expected in files.items():
        assert digest(ROOT / name) == expected, name


def inputs():
    report = json.loads((historical.OUT / 'report.json').read_text())
    data = historical.parent.parent.OUT / 'development_data.json'
    return [dict(seed=seed, checkpoint=next(c['checkpoint'] for c in report['cells']
            if c['seed'] == seed and c['step'] == 12000 and c['pool'] == 'train_32'),
            data=data.relative_to(ROOT).as_posix(), pool='train_32') for seed in (3461, 3463)]


def labels(rows):
    keys, ys, repeats = [], [], []
    for row in rows:
        mapping = dict(row['events']); ordered = sorted(mapping)
        assert all(mapping[k] == v for k, v in row['events']), 'Unstable values'
        assert mapping[row['query']] == row['label'], 'Assigned label'
        keys.append(ordered); ys.append([mapping[k] for k in ordered])
        repeats.append([sum(k == q for k, v in row['events']) > 1 for q in ordered])
    return torch.tensor(keys), torch.tensor(ys), torch.tensor(repeats)


def decimal_logits(q, s, w, b, precision=60):
    with localcontext() as ctx:
        ctx.prec = precision
        D = Decimal.from_float
        result = []; guards = []
        sd = [[D(float(x)) for x in row] for row in s]
        wd = [[D(float(x)) for x in row] for row in w]
        bd = [D(float(x)) for x in b]
        for vector in q:
            qd = [D(float(x)) for x in vector]
            reads = [sum((qd[d] * sd[d][j] for d in range(len(qd))), Decimal(0))
                     for j in range(len(wd[0]))]
            magnitudes = [sum((abs(qd[d] * sd[d][j]) for d in range(len(qd))), Decimal(0))
                          for j in range(len(wd[0]))]
            result.append([sum((v * a for v, a in zip(row, reads)), bias)
                           for row, bias in zip(wd, bd)])
            guards.append([Decimal('1e-50') * (1 + abs(bias) + sum(
                (abs(v) * a for v, a in zip(row, magnitudes)), Decimal(0)))
                for row, bias in zip(wd, bd)])
        return result, guards


def dual_certificate(q, w, b, y, candidate):
    """Exact check on primitive stored q,W,b, not rounded LP coefficients."""
    F = lambda x: Fraction.from_float(float(x))
    lam = [F(x) for x in candidate]
    if len(lam) != len(y) * (len(b) - 1) + 1 or any(x < 0 for x in lam):
        return False, None
    if sum(lam) != 1:
        return False, None
    sums = [[Fraction(0) for j in w[0]] for d in q[0]]
    bound = lam[-1]; n = 0
    for vector, target in zip(q, y):
        for c in range(len(b)):
            if c == target: continue
            a = lam[n]; n += 1
            if not a: continue
            bound += a * (F(b[target]) - F(b[c]))
            for d, v in enumerate(vector):
                for j in range(len(w[0])):
                    sums[d][j] += a * F(v) * (F(w[target][j]) - F(w[c][j]))
    return all(v == 0 for row in sums for v in row) and bound <= 0, float(bound)


def solve(q, w, b, y):
    q = np.asarray(q, dtype=np.float64); w = np.asarray(w, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64); y = np.asarray(y, dtype=np.int64)
    competitors = np.array([[c for c in range(len(b)) if c != target] for target in y])
    coefficients = (q[:, None, :, None] * (w[y, None, :] - w[competitors])[:, :, None, :]).reshape(-1, q.shape[1] * w.shape[1])
    a = np.column_stack((-coefficients, np.ones(len(coefficients))))
    a = np.vstack((a, np.r_[np.zeros(coefficients.shape[1]), 1.]))
    rhs = np.r_[(b[y, None] - b[competitors]).reshape(-1), 1.]
    cost = np.r_[np.zeros(coefficients.shape[1]), -1.]
    start = time.perf_counter()
    result = linprog(cost, A_ub=a, b_ub=rhs, bounds=[(None, None)] * len(cost),
                     method='highs-ds', options=OPTIONS)
    elapsed = time.perf_counter() - start
    s = None if result.x is None else result.x[:-1].reshape(q.shape[1], w.shape[1])
    dual = None if result.ineqlin.marginals is None else -result.ineqlin.marginals
    residual = None if dual is None else float(np.max(np.abs(a.T @ dual + cost)))
    return s, dual, dict(solver_status=int(result.status), iterations=int(result.nit),
        seconds=elapsed, solver_margin=None if result.x is None else float(result.x[-1]),
        approximate_dual_residual=residual)


def metrics(z, y, repeat):
    records = {}
    for name in ('all', 'unique', 'repeat'):
        scope = torch.ones_like(y, dtype=torch.bool) if name == 'all' else repeat == (name == 'repeat')
        n = int(scope.sum())
        records[name] = dict(count=n, accuracy=float((z.argmax(-1)[scope] == y[scope]).double().mean()) if n else None,
            ce=float(torch.nn.functional.cross_entropy(z[scope], y[scope])) if n else None)
    records['complete_maps'] = int((z.argmax(-1) == y).all(-1).sum())
    return records


def summary(records):
    result = []
    for seed in sorted({r['seed'] for r in records}):
        for group in ('all', 'native_complete', 'native_incomplete'):
            part = [r for r in records if r['seed'] == seed and
                (group == 'all' or r['native_complete'] == (group == 'native_complete'))]
            result.append(dict(seed=seed, group=group, histories=len(part),
                positive=sum(r['outcome'] == 'positive' for r in part),
                certified_nonpositive=sum(r['outcome'] == 'certified_nonpositive' for r in part),
                unresolved=sum(r['outcome'] == 'unresolved' for r in part),
                native_complete=sum(r['native_complete'] for r in part),
                native_correct=sum(r['native_correct'] for r in part),
                queries=sum(r['queries'] for r in part),
                oracle_scored_queries=sum(r['queries'] for r in part if r['oracle_correct'] is not None),
                oracle_correct=sum(r['oracle_correct'] for r in part if r['oracle_correct'] is not None)))
    return result


def evaluate(model, rows, seed):
    before = {k: v.clone() for k, v in model.state_dict().items()}; rng = torch.get_rng_state().clone()
    ids, ys, repeat = labels(rows); w = model.output.weight.detach(); b = model.output.bias.detach()
    with torch.no_grad():
        e = torch.tensor([r['events'] for r in rows]); native_s = model.prefill(e)[0]
        q = model.keys(ids); native_z = model.read((native_s[:, None].expand(-1, ids.shape[1], -1, -1).reshape(-1, 16, 8),), ids.flatten()).reshape(*ys.shape, 6)
    states = torch.full_like(native_s, float('nan')); oracle_z = torch.full_like(native_z, float('nan'))
    duals = []; records = []
    for i in range(len(rows)):
        qt = q[i].detach().tolist(); wt = w.tolist(); bt = b.tolist(); yt = ys[i].tolist()
        s, dual, record = solve(qt, wt, bt, yt)
        outcome = 'unresolved'; oracle_min = None; lower = None; bound = None; oracle_norm = None
        oracle_correct = None; oracle_ce = None; oracle_ties = None
        if s is not None and np.isfinite(s).all():
            states[i] = torch.from_numpy(s)
            with torch.no_grad():
                oracle_z[i] = model.read((states[i:i+1].expand(len(yt), -1, -1),), ids[i])
            dec, guard = decimal_logits(qt, s.tolist(), wt, bt)
            expected = torch.tensor([[float(x) for x in row] for row in dec], dtype=torch.float64)
            difference = float((expected - oracle_z[i]).abs().max())
            assert difference <= 1e-9 * (1 + float(expected.abs().max())), 'Scalar/native oracle logits'
            lower = float(min(dec[j][target] - dec[j][c] - guard[j][target] - guard[j][c]
                              for j, target in enumerate(yt) for c in range(6) if c != target))
            oracle_min = min(float(oracle_z[i,j,target] - oracle_z[i,j,c])
                             for j, target in enumerate(yt) for c in range(6) if c != target)
            oracle_norm = float(states[i].norm())
            oracle_correct = int((oracle_z[i].argmax(-1) == ys[i]).sum())
            oracle_ce = float(torch.nn.functional.cross_entropy(oracle_z[i], ys[i]))
            gaps = [min(float(oracle_z[i,j,target] - oracle_z[i,j,c]) for c in range(6) if c != target)
                    for j,target in enumerate(yt)]
            oracle_ties = sum(gap == 0 for gap in gaps)
            if lower > MARGIN and oracle_min > MARGIN: outcome = 'positive'
        if outcome != 'positive' and dual is not None:
            exact, bound = dual_certificate(qt, wt, bt, yt, dual)
            if exact and bound <= 0: outcome = 'certified_nonpositive'
        native_norm = float(native_s[i].norm())
        correct = int((native_z[i].argmax(-1) == ys[i]).sum())
        native_gaps = [min(float(native_z[i,j,target] - native_z[i,j,c]) for c in range(6) if c != target)
                       for j,target in enumerate(yt)]
        record.update(seed=seed, history=i, queries=len(yt), native_correct=correct,
            native_complete=correct == len(yt), outcome=outcome, oracle_min_margin=oracle_min,
            native_ce=float(torch.nn.functional.cross_entropy(native_z[i], ys[i])),
            native_min_margin=min(native_gaps), native_target_ties=sum(gap == 0 for gap in native_gaps),
            oracle_correct=oracle_correct, oracle_complete=None if oracle_correct is None else oracle_correct == len(yt),
            oracle_ce=oracle_ce, oracle_target_ties=oracle_ties,
            decimal_guarded_margin=lower, exact_dual_bound=bound, native_state_norm=native_norm,
            oracle_state_norm=oracle_norm, oracle_to_native_norm=None if oracle_norm is None or native_norm == 0 else oracle_norm/native_norm)
        records.append(record); duals.append(None if dual is None else torch.from_numpy(dual.copy()))
        if (i + 1) % 128 == 0: print(f'seed {seed}: {i+1}/{len(rows)} histories', flush=True)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert torch.equal(rng, torch.get_rng_state()), 'Evaluation RNG changed'
    return dict(seed=seed, queries=ids, labels=ys, repeat=repeat, normalized_queries=q.detach(),
        native_states=native_s.detach(), native_logits=native_z.detach(), states=states,
        oracle_logits=oracle_z, duals=duals, records=records, native_metrics=metrics(native_z, ys, repeat))


def execute(out, cases, fixture=False):
    assert not out.exists(), 'Preserve evidence'
    files = {p: digest(ROOT/p) for p in (*SOURCES, PROTOCOL, 'requirements-analysis.txt',
        'experiments/delta_retrieval.py', 'experiments/association_load.py',
        'experiments/query_gradient_variability.py')}
    for case in cases:
        for name in ('checkpoint', 'data'): files[case[name]] = digest(ROOT/case[name])
    if not fixture:
        ready = ROOT/'results/shared_state_feasibility_preflight.json'
        preflight = json.loads(ready.read_text())
        assert preflight['passed'] and preflight['sources'] == {p:files[p] for p in SOURCES}
        assert preflight['versions'] == versions()
        files[ready.relative_to(ROOT).as_posix()] = digest(ready)
        for folder in ('query_feedback_continuation', 'memory_first_moment_reset'):
            report = ROOT/'results'/folder/'report.json'; audit = report.with_name('audit.json')
            a = json.loads(audit.read_text()); assert a['passed'] and a['report_sha256'] == digest(report)
            for p in (report, audit): files[p.relative_to(ROOT).as_posix()] = digest(p)
    check_files(files); out.mkdir()
    write(out/'lock.json', dict(files=files, cases=cases, fixture=fixture, versions=versions(),
        method='highs-ds', options=OPTIONS, strict_margin=MARGIN, training_updates=0))
    bundles = []; records = []; total_seconds = time.perf_counter()
    for case in cases:
        rows = json.loads((ROOT/case['data']).read_text())['pools'][case['pool']]
        if not fixture:
            assert len(rows) == 768 and all(len(r['events']) == 32 and len(dict(r['events'])) == 32 for r in rows)
        model = historical.parent.previous.load_model(ROOT/case['checkpoint'])
        result = evaluate(model, rows, case['seed']); bundles.append(result); records.extend(result['records'])
    torch.save(bundles, out/'witnesses.pt')
    with (out/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    report = dict(status='complete', fixture=fixture, training_updates=0, optimizer_calls=0,
        device='cpu', dtype='float64', versions=versions(), histories=len(records),
        native_answers=sum(b['labels'].numel() for b in bundles),
        oracle_answer_outputs=sum(int(torch.isfinite(b['oracle_logits']).all(-1).sum()) for b in bundles),
        summary=summary(records), native_metrics=[dict(seed=b['seed'],metrics=b['native_metrics']) for b in bundles],
        seconds=time.perf_counter()-total_seconds, lock_sha256=digest(out/'lock.json'),
        witnesses_sha256=digest(out/'witnesses.pt'), metrics_sha256=digest(out/'metrics.csv'))
    check_files(files); write(out/'report.json', report)
    print(json.dumps(report['summary']), flush=True)


def preflight():
    spec = importlib.util.spec_from_file_location('test_shared_state_feasibility', ROOT/SOURCES[-1])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    path = ROOT/'results/shared_state_feasibility_preflight.json'
    assert not path.exists()
    write(path, dict(passed=result.wasSuccessful(), methods=result.testsRun,
        sources={p:digest(ROOT/p) for p in SOURCES}, versions=versions()))
    assert result.wasSuccessful()


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight', action='store_true')
    if parser.parse_args().preflight: preflight()
    else: execute(OUT, inputs())
