"""Frozen first-token encoder/cross-memory visibility diagnostic; no training."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from copy_marker_model import make_model
from copy_rope_model import attention, rotate

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'results/copy_source_visibility'
MODES = ('E0M0', 'E0M1', 'E1M0', 'E1M1', 'short', 'aligned')
POOLS = ('train_6', 'validation_6', 'test_6', 'test_8', 'test_10',
         'uniform_6', 'uniform_8', 'uniform_10')
SEEDS = (3163, 3167, 3169)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def forward(model, rows, mode, perturb=0.):
    if mode not in MODES:
        raise ValueError(mode)
    content = rows[:, :6] if mode == 'short' else rows
    source = torch.cat((content, torch.full((len(rows), 1), 2, dtype=torch.long)), 1)
    base = model.base
    enc, dec = base.encoder, base.decoder
    n = content.shape[1]
    ep = torch.arange(n+1)
    dp = torch.zeros(1, dtype=torch.long)
    excluded = (ep >= 6) & (ep < n)
    encoder_mask = excluded.expand(len(rows), -1) if mode.startswith('E0') else torch.zeros_like(source, dtype=torch.bool)
    memory_mask = excluded.expand(len(rows), -1) if mode.endswith('M0') else torch.zeros_like(source, dtype=torch.bool)
    s = base.source(source) * math.sqrt(8)
    if perturb:
        # Test-only input perturbation; formal calls always use zero.
        s = s + perturb * excluded[None, :, None] * torch.arange(1, 9, dtype=s.dtype)
    t = base.target(torch.ones(len(rows), 1, dtype=torch.long)) * math.sqrt(8)
    memory = enc.norm1(s + attention(enc.self_attn, s, s, s, encoder_mask, ep, ep))
    memory = enc.norm2(memory + enc.linear2(F.relu(enc.linear1(memory))))
    x = dec.norm1(t + attention(dec.self_attn, t, t, t, torch.zeros(len(rows), 1, dtype=torch.bool), dp, dp, True))
    m = dec.multihead_attn
    h, d = m.num_heads, m.embed_dim // m.num_heads
    q, k, v = [F.linear(a, w, b).reshape(len(rows), a.shape[1], h, d).transpose(1, 2)
               for a, w, b in zip((x, memory, memory), m.in_proj_weight.chunk(3), m.in_proj_bias.chunk(3))]
    q, k = rotate(q, dp), rotate(k, ep)
    scores = q @ k.transpose(-1, -2) / math.sqrt(d)
    weights = scores.masked_fill(memory_mask[:, None, None], -torch.inf).softmax(-1)
    if mode == 'aligned':
        weights = torch.zeros_like(weights)
        weights[..., 0] = 1.
    cross = (weights @ v).transpose(1, 2).reshape_as(x)
    x = dec.norm2(x + m.out_proj(cross))
    z = base.output(dec.norm3(x + dec.linear2(F.relu(dec.linear1(x)))))[:, 0]
    return dict(logits=z, q=q, k=k, v=v, scores=scores, weights=weights)


def metrics(rows, stored):
    z, weights = stored['logits'], stored['weights']
    labels = torch.tensor([r[0] for r in rows])
    good = z.argmax(-1) == labels
    loss = F.cross_entropy(z, labels, reduction='none')
    others = z.clone().scatter(1, labels[:, None], -torch.inf)
    margin = z.gather(1, labels[:, None])[:, 0] - others.max(-1).values
    distinct = torch.tensor([len(set(r)) == len(r) for r in rows])
    result = {}
    for g, mask in (('all', torch.ones(len(rows), dtype=torch.bool)), ('distinct', distinct), ('repeated', ~distinct)):
        count = int(mask.sum())
        result[g] = dict(count=count, accuracy=float(good[mask].double().mean()) if count else None,
                         ce=float(loss[mask].mean()) if count else None,
                         margin=float(margin[mask].mean()) if count else None,
                         target_mass=float(weights[mask, :, 0, 0].mean()) if count else None)
    return result


def evaluate(model, rows, mode):
    parts = []
    with torch.no_grad():
        for batch in torch.tensor(rows).split(64):
            parts.append(forward(model, batch, mode))
    stored = {key: torch.cat([p[key] for p in parts]) for key in parts[0]}
    return metrics(rows, stored), stored


def contrasts(cells, evidence, pools):
    index = {(c['seed'], c['pool'], c['mode']): c['metrics'] for c in cells}
    results = []
    formulas = {'encoder_native': {'E0M1': 1, 'E1M1': -1},
                'encoder_restricted': {'E0M0': 1, 'E1M0': -1},
                'competitor_native': {'E1M0': 1, 'E1M1': -1},
                'competitor_restricted': {'E0M0': 1, 'E0M1': -1},
                'joint': {'E0M0': 1, 'E1M1': -1}, 'marker': {'short': 1, 'E0M0': -1},
                'interaction': {'E1M1': 1, 'E1M0': -1, 'E0M1': -1, 'E0M0': 1}}
    for seed, pool in sorted({(c['seed'], c['pool']) for c in cells}):
        rows = pools[pool]
        label = torch.tensor([r[0] for r in rows])
        correct = {m: evidence[f'{seed}/{pool}/{m}']['logits'].argmax(-1) == label for m in MODES}
        for group in ('all', 'distinct', 'repeated'):
            if not index[seed, pool, 'E1M1'][group]['count']:
                continue
            mask = torch.tensor([group == 'all' or (len(set(r)) == len(r)) == (group == 'distinct') for r in rows])
            effects = {name: {metric: sum(coef * index[seed, pool, mode][group][metric] for mode, coef in formula.items())
                              for metric in ('accuracy', 'margin')} for name, formula in formulas.items()}
            results.append(dict(seed=seed, pool=pool, group=group, effects=effects,
                                recovery={name: effects[name]['accuracy'] >= .1 and effects[name]['margin'] > 0
                                          for name in ('encoder_native', 'competitor_native', 'joint')},
                                full_joint=index[seed, pool, 'E0M0'][group]['accuracy'] == 1.,
                                marker_flag=abs(effects['marker']['accuracy']) > .01,
                                transitions={m: dict(rescues=int((correct[m] & ~correct['E1M1'] & mask).sum()),
                                                     harms=int((~correct[m] & correct['E1M1'] & mask).sum())) for m in MODES}))
    primary = [r for r in results if r['pool'] in ('test_8', 'test_10') and r['group'] != 'all']
    return dict(groups=results, primary_groups=len(primary),
                all_group_recovery={k: bool(primary) and all(r['recovery'][k] for r in primary)
                                    for k in ('encoder_native', 'competitor_native', 'joint')},
                full_joint=bool(primary) and all(r['full_joint'] for r in primary),
                marker_flags=sum(r['marker_flag'] for r in primary))


def produce(directory):
    directory = Path(directory)
    assert not (directory / 'report.json').exists()
    lock = json.loads((directory / 'lock.json').read_text())
    assert all(digest(ROOT / p) == h for p, h in lock['protected'].items())
    evidence, cells = {}, []
    started = time.perf_counter()
    for run in lock['runs']:
        model = make_model('rope').eval()
        assert digest(ROOT / run['checkpoint']) == run['checkpoint_sha256']
        model.load_state_dict(torch.load(ROOT / run['checkpoint'], weights_only=True)['model'])
        for pool, rows in lock['pools'].items():
            for mode in MODES:
                met, saved = evaluate(model, rows, mode)
                cells.append(dict(seed=run['seed'], pool=pool, mode=mode, metrics=met))
                evidence[f"{run['seed']}/{pool}/{mode}"] = saved
        print(json.dumps(dict(seed=run['seed'], cells=len(cells))), flush=True)
    assert len(cells) == len(lock['runs']) * len(lock['pools']) * len(MODES)
    torch.save(evidence, directory / 'evidence.pt')
    report = dict(status='complete', cells=cells, decision=contrasts(cells, evidence, lock['pools']),
                  outputs=sum(len(v) for v in lock['pools'].values())*len(lock['runs'])*len(MODES),
                  training_updates=0, device='cpu', dtype='float64', threads=1,
                  seconds=time.perf_counter()-started, lock_sha256=digest(directory / 'lock.json'),
                  evidence_sha256=digest(directory / 'evidence.pt'))
    assert all(digest(ROOT / p) == h for p, h in lock['protected'].items())
    write(directory / 'report.json', report)
    return report


def freeze():
    assert not (DIRECTORY / 'lock.json').exists()
    preflight = json.loads((DIRECTORY / 'preflight.json').read_text())
    assert preflight['passed']
    oldpath = ROOT / 'results/copy_marker_training/report.json'
    old = json.loads(oldpath.read_text())
    runs = [r for r in old['runs'] if r['arm'] == 'rope']
    assert tuple(r['seed'] for r in runs) == SEEDS
    protected = [ROOT / p for p in ('docs/protocols/copy-source-visibility.md',
                 'experiments/copy_source_visibility.py', 'experiments/audit_copy_source_visibility.py',
                 'tests/test_copy_source_visibility.py', 'experiments/copy_marker_model.py',
                 'experiments/copy_rope_model.py', 'experiments/encoder_decoder_contract.py',
                 'experiments/encoder_decoder_copy.py', 'experiments/audit_kv_cache_equivalence.py',
                 'results/copy_source_visibility/preflight.json')]
    for name in ('copy_marker_training', 'copy_marker_alignment'):
        p = ROOT / 'results' / name
        audit = json.loads((p / 'audit.json').read_text())
        assert audit['passed'] and audit['report_sha256'] == digest(p / 'report.json')
        protected.extend((p / 'report.json', p / 'audit.json'))
    alignment = ROOT / 'results/copy_marker_alignment/evidence.pt'
    aligned_report = json.loads((alignment.parent / 'report.json').read_text())
    assert digest(alignment) == aligned_report['evidence_sha256']
    protected.append(alignment)
    for run in runs:
        assert digest(ROOT / run['checkpoint']) == run['checkpoint_sha256']
        protected.append(ROOT / run['checkpoint'])
    pools = {p: old['evaluation_pools'][p] for p in POOLS}
    assert sum(map(len, pools.values())) == 728
    write(DIRECTORY / 'lock.json', dict(runs=[{k: r[k] for k in ('seed', 'checkpoint', 'checkpoint_sha256')} for r in runs],
          pools=pools, modes=list(MODES), archive=str(alignment.relative_to(ROOT)),
          protected={p.relative_to(ROOT).as_posix(): digest(p) for p in protected}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'run'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    freeze() if args.action == 'freeze' else produce(DIRECTORY)
