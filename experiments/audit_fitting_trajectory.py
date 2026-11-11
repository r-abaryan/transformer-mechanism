"""Separate matrix recurrence, row labels, margins and trajectory decisions."""
import csv
import json
import math

import torch
import fitting_trajectory as study


@torch.no_grad()
def matrix_outputs(reader, events, queries):
    raw = reader.key.weight[events[..., 0]]
    k = raw / raw.square().sum(-1, keepdim=True).clamp_min(1e-24).sqrt()
    rawq = reader.key.weight[queries]
    q = rawq / rawq.square().sum(-1, keepdim=True).clamp_min(1e-24).sqrt()
    beta = reader.write_logit.sigmoid()
    identity = torch.eye(16, dtype=torch.float64)
    state = k.new_zeros(len(events), 16, 8)
    basis_state = k.new_zeros(len(events), 16, 6)
    for t in range(events.shape[1]):
        key = k[:, t]
        transition = identity - beta * key[:, :, None] * key[:, None, :]
        value = reader.value.weight[events[:, t, 1]]
        onehot = torch.eye(6, dtype=torch.float64)[events[:, t, 1]]
        state = transition @ state + beta * key[:, :, None] * value[:, None, :]
        basis_state = transition @ basis_state + beta * key[:, :, None] * onehot[:, None, :]
    c = q @ basis_state
    z = (q @ state) @ reader.output.weight.T + reader.output.bias
    return c, z


def independent_margins(values, labels):
    top, indices = values.topk(2, dim=-1)
    competitor = torch.where(indices[..., 0] == labels, top[..., 1], top[..., 0])
    raw = values.gather(-1, labels[..., None]).squeeze(-1)-competitor
    centered = values-values.sum(-1, keepdim=True)/6
    norm = centered.square().sum(-1).sqrt()
    unit = torch.zeros_like(raw)
    nz = norm > 0
    unit[nz] = raw[nz]/norm[nz]
    return raw, unit


def stats(values):
    n = values.numel()
    if not n: return dict(count=0, mean=None, q05=None, median=None, q95=None)
    ordered = values.sort().values
    def quantile(p):
        rank = p*(n-1); low = math.floor(rank); high = math.ceil(rank)
        return float(ordered[low]+(rank-low)*(ordered[high]-ordered[low]))
    return dict(count=n, mean=float(values.sum()/n), q05=quantile(.05), median=quantile(.5), q95=quantile(.95))


def independent_summary(c, z, y, repeat, initial):
    cm, cu = independent_margins(c, y); zm, zu = independent_margins(z, y)
    cc = c.argmax(-1).eq(y); nc = z.argmax(-1).eq(y)
    icc = initial['coefficients'].argmax(-1).eq(y); inc = initial['logits'].argmax(-1).eq(y)
    _, icu = independent_margins(initial['coefficients'], y)
    _, izu = independent_margins(initial['logits'], y)
    native = study.prior.reference.metrics(z, y, repeat)
    groups = {}
    for group in ('all', 'unique', 'repeat'):
        mask = torch.full_like(repeat, True) if group == 'all' else repeat.eq(group == 'repeat')
        n = int(mask.sum()); count = lambda x: int(x[mask].sum())
        cohorts = {}
        for name, selected in [('initial_correct', inc), ('initial_wrong', ~inc)]:
            take = mask & selected
            cohorts[name] = dict(coefficient_change=stats((cu-icu)[take]), native_change=stats((zu-izu)[take]))
        groups[group] = dict(count=n, coefficient_accuracy=count(cc)/n if n else None,
            coefficient_raw_margin=stats(cm[mask]), coefficient_unit_margin=stats(cu[mask]),
            native_raw_margin=stats(zm[mask]), native_unit_margin=stats(zu[mask]),
            coefficient_change=stats((cu-icu)[mask]), native_change=stats((zu-izu)[mask]),
            zero_coefficient_vectors=count((c-c.sum(-1, keepdim=True)/6).square().sum(-1).eq(0)),
            both_correct=count(nc & cc), coefficient_only=count(~nc & cc), native_only=count(nc & ~cc),
            neither=count(~nc & ~cc), native_rescues=count(nc & ~inc), native_harms=count(~nc & inc),
            coefficient_rescues=count(cc & ~icc), coefficient_harms=count(~cc & icc), cohorts=cohorts)
    return dict(native=native, groups=groups)


@torch.no_grad()
def audit(out, persist=True):
    lock = json.loads((out/'lock.json').read_text()); report = json.loads((out/'report.json').read_text())
    digest = study.prior.replay.digest
    study.prior.replay.check_files(lock['files'])
    assert report['status'] == 'complete' and lock['torch'] == torch.__version__
    for name, field, owner in [('lock.json', 'lock_sha256', report), ('pools.pt', 'pools_sha256', lock),
                               ('scores.pt', 'scores_sha256', report), ('trajectory.csv', 'csv_sha256', report)]:
        assert digest(out/name) == owner[field], 'Artifact binding'
    runs = lock['runs']; pools = torch.load(out/'pools.pt', weights_only=True)
    scores = torch.load(out/'scores.pt', weights_only=True)
    expected = {(r['seed'], r['arm'], i, f'{split}_{load}') for r in runs for i in range(len(r['snapshots']))
                for split in study.SPLITS for load in study.prior.LOADS}
    ids = [(c['seed'], c['arm'], c['stage'], c['pool']) for c in scores]
    assert len(ids) == len(expected) and set(ids) == expected, 'Missing cell'
    assert len(report['cells']) == len(scores)
    if not lock['fixture']:
        original_runs, original_pools = study.inputs()
        assert runs == original_runs and pools == original_pools
        historical = torch.load(study.prior.previous.OUT/'scores.pt', weights_only=True)
    # Shared starts are checked from actual parameter values, not file names.
    for seed in {r['seed'] for r in runs}:
        pair = [r for r in runs if r['seed'] == seed]; assert len(pair) == 2
        a, b = [study.prior.model(r['snapshots'][0]['checkpoint']).state_dict() for r in pair]
        assert all(torch.equal(a[k], b[k]) for k in a), 'Unmatched start'
    maximum_c = maximum_z = 0.; targets = 0; independently_reduced = []
    for run in runs:
        initial = {}
        for stage, snap in enumerate(run['snapshots']):
            reader = study.prior.model(snap['checkpoint'])
            for split in study.SPLITS:
                for load in study.prior.LOADS:
                    pool = f'{split}_{load}'; identity = (run['seed'], run['arm'], stage, pool)
                    idx = ids.index(identity); cell = scores[idx]; saved = report['cells'][idx]
                    assert cell['checkpoint'] == snap['checkpoint'] and cell['gradients'] == snap['step'], 'Checkpoint lineage'
                    e, q, y, repeat = study.prior.reference.independent.inputs(pools[pool], 'all_queries')
                    assert torch.equal(cell['labels'], y) and torch.equal(cell['repeat'], repeat), 'Independent labels'
                    cs = []; zs = []
                    for i in range(0, len(e), 64):
                        c, z = matrix_outputs(reader, e[i:i+64], q[i:i+64]); cs.append(c); zs.append(z)
                    c, z = torch.cat(cs), torch.cat(zs)
                    maximum_c = max(maximum_c, study.prior.reference.independent.close(cell['coefficients'], c))
                    maximum_z = max(maximum_z, study.prior.reference.independent.close(cell['logits'], z))
                    assert torch.equal(c.argmax(-1), cell['coefficients'].argmax(-1)), 'Coefficient ranking'
                    assert torch.equal(z.argmax(-1), cell['logits'].argmax(-1)), 'Native ranking'
                    raw = dict(coefficients=c, logits=z)
                    if stage == 0: initial[pool] = raw
                    summary = independent_summary(c, z, y, repeat, initial[pool])
                    metadata = {k: cell[k] for k in ('seed', 'arm', 'stage', 'gradients', 'pool', 'checkpoint')}
                    want = dict(**metadata, summary=summary)
                    study.prior.compare(saved, want)
                    independently_reduced.append(want); targets += y.numel()
                    if not lock['fixture']:
                        old = [a for a in historical if (a['seed'], a['arm'], a['step'], a['pool']) ==
                               (run['seed'], run['arm'], snap['step'], pool)]
                        if old:
                            assert len(old) == 1 and torch.equal(old[0]['logits'], cell['logits']), 'Historical native control'
                            study.prior.compare(summary['native'], old[0]['metrics'])
            print(json.dumps(dict(audited_seed=run['seed'], arm=run['arm'], stage=stage)), flush=True)
    # Independent primary signs, using independently reduced matrix outputs.
    pairs = []
    for seed in sorted({r['seed'] for r in runs}):
        contrasts = {}
        for split in study.SPLITS:
            selected = []
            for arm in study.prior.ARMS:
                items = [c for c in independently_reduced if c['seed'] == seed and c['arm'] == arm and c['pool'] == f'{split}_32']
                selected.append(max(items, key=lambda c: c['stage'])['summary']['groups']['all'])
            a, b = selected
            contrasts[split] = dict(coefficient_unit_margin=a['coefficient_unit_margin']['mean']-b['coefficient_unit_margin']['mean'],
                                    coefficient_accuracy=a['coefficient_accuracy']-b['coefficient_accuracy'],
                                    native_unit_margin=a['native_unit_margin']['mean']-b['native_unit_margin']['mean'])
        pairs.append(dict(seed=seed, contrasts=contrasts))
    broad = all(p['contrasts'][s]['coefficient_unit_margin'] > 1e-10 for p in pairs for s in study.SPLITS)
    only = all(p['contrasts']['train']['coefficient_unit_margin'] > 1e-10 and
               p['contrasts']['validation']['coefficient_unit_margin'] <= 1e-10 and
               p['contrasts']['test']['coefficient_unit_margin'] <= 1e-10 for p in pairs)
    decision = dict(pairs=pairs, broad_target_margin_signature=broad, training_only_target_margin_signature=only,
                    verdict='broad' if broad else 'training_only' if only else 'mixed_or_unresolved')
    study.prior.compare(report['decision'], decision)
    with (out/'trajectory.csv').open(newline='', encoding='utf-8') as f: actual_rows = list(csv.DictReader(f))
    expected_rows = list(study.csv_rows(report['cells']))
    assert len(actual_rows) == len(expected_rows)
    for actual, want in zip(actual_rows, expected_rows):
        assert actual == {k: '' if v is None else str(v) for k, v in want.items()}, 'CSV mismatch'
    assert report['evaluation_targets'] == targets and report['checkpoints'] == sum(len(r['snapshots']) for r in runs)
    assert report['optimizer_calls'] == report['gradient_calls'] == 0
    study.prior.replay.check_files(lock['files'])
    result = dict(passed=True, cells=len(scores), csv_rows=len(actual_rows), evaluation_targets=targets,
                  max_coefficient_error=maximum_c, max_logit_error=maximum_z, protected_files=len(lock['files']),
                  report_sha256=digest(out/'report.json'), decision=decision)
    if persist:
        assert not (out/'audit.json').exists(), 'Preserve audit'
        study.prior.replay.write(out/'audit.json', result)
    return result


if __name__ == '__main__':
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    print(json.dumps(audit(study.OUT)), flush=True)
