"""Follow fixed examples through saved fitting trajectories, with no training."""
import argparse
import csv
import json
import time

import torch
from torch.nn import functional as F
import fitting_representation as prior

ROOT = prior.ROOT
OUT = ROOT / 'results/fitting_trajectory'
SOURCES = ('experiments/fitting_trajectory.py', 'experiments/audit_fitting_trajectory.py',
           'tests/test_fitting_trajectory.py')
PROTOCOL = 'docs/protocols/fitting-trajectory.md'
SPLITS = ('train', 'validation', 'test')
GROUPS = ('all', 'unique', 'repeat')
TOL = 1e-10


def inputs():
    report = json.loads((prior.previous.OUT / 'report.json').read_text())
    runs = [dict(seed=r['seed'], arm=r['arm'], snapshots=r['snapshots']) for r in report['runs']]
    return runs, torch.load(prior.previous.OUT / 'pools.pt', weights_only=True)


@torch.no_grad()
def coefficients(reader, events, queries):
    keys = reader.keys(events[..., 0])
    basis = F.one_hot(events[..., 1], 6).to(torch.float64)
    state = keys.new_zeros(len(events), 16, 6)
    beta = reader.write_logit.sigmoid()
    for t in range(events.shape[1]):
        key = keys[:, t]
        old = torch.einsum('bk,bkj->bj', key, state)
        state += beta * key[:, :, None] * (basis[:, t] - old)[:, None, :]
    return torch.einsum('bmk,bkj->bmj', reader.keys(queries), state)


def margins(values, labels):
    target = values.gather(-1, labels[..., None]).squeeze(-1)
    others = values.clone().scatter_(-1, labels[..., None], -torch.inf).max(-1).values
    raw = target - others
    norm = (values - values.mean(-1, keepdim=True)).norm(dim=-1)
    # Zero centered vectors have zero margins. Clamping is only a zero guard.
    unit = torch.where(norm > 0, raw / norm.clamp_min(torch.finfo(values.dtype).tiny), 0.)
    return raw, unit


def distribution(values):
    if not values.numel():
        return dict(count=0, mean=None, q05=None, median=None, q95=None)
    q = torch.quantile(values, torch.tensor([.05, .5, .95], dtype=values.dtype))
    return dict(count=values.numel(), mean=float(values.mean()), q05=float(q[0]),
                median=float(q[1]), q95=float(q[2]))


def summarize_cell(c, logits, labels, repeat, initial):
    cm, cu = margins(c, labels)
    zm, zu = margins(logits, labels)
    nc = logits.argmax(-1) == labels
    cc = c.argmax(-1) == labels
    initial_nc = initial['logits'].argmax(-1) == labels
    initial_cc = initial['coefficients'].argmax(-1) == labels
    _, initial_cu = margins(initial['coefficients'], labels)
    _, initial_zu = margins(initial['logits'], labels)
    groups = {}
    for group in GROUPS:
        mask = torch.ones_like(repeat) if group == 'all' else repeat == (group == 'repeat')
        n = int(mask.sum())
        counts = dict(both_correct=int((mask & nc & cc).sum()),
                      coefficient_only=int((mask & ~nc & cc).sum()),
                      native_only=int((mask & nc & ~cc).sum()),
                      neither=int((mask & ~nc & ~cc).sum()),
                      native_rescues=int((mask & nc & ~initial_nc).sum()),
                      native_harms=int((mask & ~nc & initial_nc).sum()),
                      coefficient_rescues=int((mask & cc & ~initial_cc).sum()),
                      coefficient_harms=int((mask & ~cc & initial_cc).sum()))
        cohorts = {}
        for name, selected in [('initial_correct', initial_nc), ('initial_wrong', ~initial_nc)]:
            take = mask & selected
            cohorts[name] = dict(coefficient_change=distribution((cu-initial_cu)[take]),
                                 native_change=distribution((zu-initial_zu)[take]))
        groups[group] = dict(count=n, coefficient_accuracy=float(cc[mask].double().mean()) if n else None,
                             coefficient_raw_margin=distribution(cm[mask]),
                             coefficient_unit_margin=distribution(cu[mask]),
                             native_raw_margin=distribution(zm[mask]),
                             native_unit_margin=distribution(zu[mask]),
                             coefficient_change=distribution((cu-initial_cu)[mask]),
                             native_change=distribution((zu-initial_zu)[mask]),
                             zero_coefficient_vectors=int((mask & (c-c.mean(-1, keepdim=True)).norm(dim=-1).eq(0)).sum()),
                             **counts, cohorts=cohorts)
    return dict(native=prior.training.reduce_scores(logits, labels, repeat), groups=groups)


def decisions(cells):
    pairs = []
    for seed in sorted({c['seed'] for c in cells}):
        contrasts = {}
        for split in SPLITS:
            final = {}
            for arm in prior.ARMS:
                subset = [c for c in cells if c['seed'] == seed and c['arm'] == arm and c['pool'] == f'{split}_32']
                final[arm] = max(subset, key=lambda c: c['stage'])
            a, b = (final[arm]['summary']['groups']['all'] for arm in prior.ARMS)
            contrasts[split] = dict(coefficient_unit_margin=a['coefficient_unit_margin']['mean']-b['coefficient_unit_margin']['mean'],
                                    coefficient_accuracy=a['coefficient_accuracy']-b['coefficient_accuracy'],
                                    native_unit_margin=a['native_unit_margin']['mean']-b['native_unit_margin']['mean'])
        pairs.append(dict(seed=seed, contrasts=contrasts))
    broad = all(p['contrasts'][split]['coefficient_unit_margin'] > TOL for p in pairs for split in SPLITS)
    concentrated = all(p['contrasts']['train']['coefficient_unit_margin'] > TOL and
                       all(p['contrasts'][split]['coefficient_unit_margin'] <= TOL for split in ('validation', 'test'))
                       for p in pairs)
    return dict(pairs=pairs, broad_target_margin_signature=broad,
                training_only_target_margin_signature=concentrated,
                verdict='broad' if broad else 'training_only' if concentrated else 'mixed_or_unresolved')


def csv_rows(cells):
    for cell in cells:
        for group in GROUPS:
            g = cell['summary']['groups'][group]
            native = cell['summary']['native']['groups'][group]
            yield dict(seed=cell['seed'], arm=cell['arm'], stage=cell['stage'], gradients=cell['gradients'],
                       pool=cell['pool'], group=group, count=g['count'], native_accuracy=native['accuracy'],
                       native_ce=native['ce'], coefficient_accuracy=g['coefficient_accuracy'],
                       coefficient_unit_margin=g['coefficient_unit_margin']['mean'],
                       native_unit_margin=g['native_unit_margin']['mean'],
                       coefficient_only=g['coefficient_only'], native_only=g['native_only'],
                       native_rescues=g['native_rescues'], native_harms=g['native_harms'])


@torch.no_grad()
def execute(out, runs, pools, fixture=False):
    assert not out.exists(), 'Preserve evidence'
    files = {p: prior.replay.digest(ROOT/p) for p in prior.previous.prior.dependencies() | set(SOURCES)}
    for run in runs:
        for snap in run['snapshots']:
            files[snap['checkpoint']] = prior.replay.digest(ROOT/snap['checkpoint'])
    if not fixture:
        assert {r['seed'] for r in runs} == set(prior.previous.SEEDS)
        assert len(runs) == 6 and all(len(r['snapshots']) == 5 for r in runs)
        for p in [PROTOCOL, 'results/fitting_trajectory_preflight.json',
                  'results/fitting_direction/report.json', 'results/fitting_direction/audit.json',
                  'results/fitting_direction/pools.pt', 'results/fitting_direction/scores.pt']:
            files[p] = prior.replay.digest(ROOT/p)
        pre = json.loads((ROOT/'results/fitting_trajectory_preflight.json').read_text())
        assert pre['passed'] and pre['sources'] == {p: files[p] for p in SOURCES}
    prior.replay.check_files(files)
    out.mkdir(parents=True)
    torch.save(pools, out/'pools.pt')
    prior.replay.write(out/'lock.json', dict(files=files, runs=runs, fixture=fixture,
                       pools_sha256=prior.replay.digest(out/'pools.pt'), torch=torch.__version__))
    started = time.perf_counter()
    cells = []; scores = []; maximum = 0.
    for run in runs:
        initial = {}
        for stage, snap in enumerate(run['snapshots']):
            reader = prior.model(snap['checkpoint'])
            effective = reader.output.weight @ reader.value.weight.T
            for split in SPLITS:
                for load in prior.LOADS:
                    pool = f'{split}_{load}'
                    e, q, labels, repeat = prior.training.replay.pressure.prepare(pools[pool], 'all_queries')
                    cs = []; zs = []
                    for i in range(0, len(e), 64):
                        c = coefficients(reader, e[i:i+64], q[i:i+64])
                        z = prior.training.forward_queries(reader, e[i:i+64], q[i:i+64])
                        err = float((z-(c @ effective.T+reader.output.bias)).abs().max())
                        assert err < 1e-9, 'Factorization'
                        maximum = max(maximum, err); cs.append(c); zs.append(z)
                    c, z = torch.cat(cs), torch.cat(zs)
                    raw = dict(coefficients=c, logits=z, labels=labels, repeat=repeat)
                    if stage == 0: initial[pool] = raw
                    metadata = dict(seed=run['seed'], arm=run['arm'], stage=stage,
                                    gradients=snap['step'], pool=pool, checkpoint=snap['checkpoint'])
                    cells.append(dict(**metadata, summary=summarize_cell(c, z, labels, repeat, initial[pool])))
                    scores.append(dict(**metadata, **raw))
            print(json.dumps(dict(seed=run['seed'], arm=run['arm'], stage=stage, gradients=snap['step'])), flush=True)
    torch.save(scores, out/'scores.pt')
    rows = list(csv_rows(cells))
    with (out/'trajectory.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    prior.replay.check_files(files)
    prior.replay.write(out/'report.json', dict(status='complete', cells=cells, decision=decisions(cells),
                       evaluation_targets=sum(c['labels'].numel() for c in scores), checkpoints=sum(len(r['snapshots']) for r in runs),
                       optimizer_calls=0, gradient_calls=0, seconds=time.perf_counter()-started,
                       device='cpu', dtype='float64', threads=1, max_factorization_error=maximum,
                       lock_sha256=prior.replay.digest(out/'lock.json'), scores_sha256=prior.replay.digest(out/'scores.pt'),
                       csv_sha256=prior.replay.digest(out/'trajectory.csv')))


if __name__ == '__main__':
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser(); parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if args.audit:
        from audit_fitting_trajectory import audit
        print(json.dumps(audit(OUT)), flush=True)
    else:
        execute(OUT, *inputs())
