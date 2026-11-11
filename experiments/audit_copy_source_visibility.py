"""Independent layer/scalar checks for the frozen source-visibility diagnostic."""
import json
import math
from pathlib import Path

import torch

from copy_source_visibility import ROOT, DIRECTORY, MODES, digest, write
from copy_marker_model import make_model


def reference(model, rows, mode, source_positions=None):
    base = model.base
    data = [r[:6] if mode == 'short' else list(r) for r in rows]
    n = len(data[0])
    source = torch.tensor([r + [2] for r in data])
    enc, dec = base.encoder, base.decoder

    def norm(x, layer):
        centered = x-x.mean(-1, keepdim=True)
        return centered/(centered.square().mean(-1, keepdim=True)+layer.eps).sqrt()*layer.weight+layer.bias

    def ff(layer, x):
        return (x@layer.linear1.weight.T+layer.linear1.bias).clamp_min(0)@layer.linear2.weight.T+layer.linear2.bias

    def rotated(x, positions=None):
        pieces = []
        positions = torch.arange(x.shape[-2], dtype=x.dtype) if positions is None else torch.tensor(positions, dtype=x.dtype)
        for i in range(x.shape[-1]//2):
            angle = positions * 10000.**(-2*i/x.shape[-1])
            a, b = x[..., 2*i], x[..., 2*i+1]
            pieces.extend((a*angle.cos()-b*angle.sin(), a*angle.sin()+b*angle.cos()))
        return torch.stack(pieces, -1)

    def att(layer, x, mem, blocked, force=False, qp=None, kp=None):
        h, d = layer.num_heads, layer.embed_dim//layer.num_heads
        projections = []
        for i, value in enumerate((x, mem, mem)):
            w = layer.in_proj_weight[i*layer.embed_dim:(i+1)*layer.embed_dim]
            b = layer.in_proj_bias[i*layer.embed_dim:(i+1)*layer.embed_dim]
            p = (value@w.T+b).reshape(len(rows), value.shape[1], h, d).permute(0, 2, 1, 3)
            projections.append(rotated(p, qp if i == 0 else kp) if i < 2 else p)
        q, k, v = projections
        scores = (q@k.transpose(-1, -2))/math.sqrt(d)
        mask = torch.tensor(blocked, dtype=torch.bool)[None, None, None]
        s = scores.masked_fill(mask, -torch.inf)
        ex = (s-s.max(-1, keepdim=True).values).exp()
        weights = ex/ex.sum(-1, keepdim=True)
        if force:
            weights = torch.tensor([1.]+[0.]*(len(blocked)-1), dtype=x.dtype).expand_as(weights)
            mixed = v[:, :, :1]
        else:
            mixed = weights@v
        out = mixed.permute(0, 2, 1, 3).reshape_as(x)@layer.out_proj.weight.T+layer.out_proj.bias
        return out, dict(q=q, k=k, v=v, scores=scores, weights=weights)

    eblock = [mode in ('E0M0', 'E0M1') and 6 <= j < n for j in range(n+1)]
    mblock = [mode in ('E0M0', 'E1M0') and 6 <= j < n for j in range(n+1)]
    s = base.source(source)*math.sqrt(8)
    memory = norm(s+att(enc.self_attn, s, s, eblock, qp=source_positions, kp=source_positions)[0], enc.norm1)
    memory = norm(memory+ff(enc, memory), enc.norm2)
    t = base.target(torch.tensor([[1] for _ in rows]))*math.sqrt(8)
    x = norm(t+att(dec.self_attn, t, t, [False])[0], dec.norm1)
    cross, tensors = att(dec.multihead_attn, x, memory, mblock, mode == 'aligned', kp=source_positions)
    x = norm(x+cross, dec.norm2)
    x = norm(x+ff(dec, x), dec.norm3)
    return dict(logits=(x@base.output.weight.T)[:, 0], **tensors)


def scalar_metrics(rows, tensors):
    logits, weights = tensors['logits'].tolist(), tensors['weights'].tolist()
    result = {}
    for group in ('all', 'distinct', 'repeated'):
        ids = [i for i, r in enumerate(rows) if group == 'all' or (len(set(r)) == len(r)) == (group == 'distinct')]
        values = dict(count=len(ids), accuracy=None, ce=None, margin=None, target_mass=None)
        if ids:
            ce, margins, masses, correct = [], [], [], []
            for i in ids:
                z, label = logits[i], rows[i][0]
                top = max(range(len(z)), key=z.__getitem__)
                correct.append(top == label)
                maximum = max(z)
                ce.append(maximum+math.log(math.fsum(math.exp(v-maximum) for v in z))-z[label])
                margins.append(z[label]-max(v for j, v in enumerate(z) if j != label))
                masses.append(math.fsum(head[0][0] for head in weights[i])/len(weights[i]))
            values.update(accuracy=sum(correct)/len(ids), ce=math.fsum(ce)/len(ids),
                          margin=math.fsum(margins)/len(ids), target_mass=math.fsum(masses)/len(ids))
        result[group] = values
    return result


def close(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max([close(a[k], b[k]) for k in a]+[0.])
    if isinstance(a, list):
        assert len(a) == len(b)
        return max([close(x, y) for x, y in zip(a, b)]+[0.])
    if isinstance(a, float):
        error = abs(a-b)
        assert math.isfinite(error) and error <= 1e-9, (a, b)
        return error
    assert a == b, (a, b)
    return 0.


def independent_contrasts(cells, stored, pools):
    index = {(c['seed'], c['pool'], c['mode']): c['metrics'] for c in cells}
    groups = []
    for seed, pool in sorted({(c['seed'], c['pool']) for c in cells}):
        rows = pools[pool]
        predictions = {m: stored[f'{seed}/{pool}/{m}']['logits'].argmax(-1).tolist() for m in MODES}
        for g in ('all', 'distinct', 'repeated'):
            ids = [i for i, r in enumerate(rows) if g == 'all' or (len(set(r)) == len(r)) == (g == 'distinct')]
            if not ids:
                continue
            effects = {k: {} for k in ('encoder_native', 'encoder_restricted', 'competitor_native', 'competitor_restricted', 'joint', 'marker', 'interaction')}
            for met in ('accuracy', 'margin'):
                a, b, c, d, s = [index[seed, pool, m][g][met] for m in ('E0M0', 'E0M1', 'E1M0', 'E1M1', 'short')]
                for k, value in zip(effects, (b-d, a-c, c-d, a-b, a-d, s-a, d-c-b+a)):
                    effects[k][met] = value
            transitions = {}
            for m in MODES:
                rescued = harmed = 0
                for i in ids:
                    before = predictions['E1M1'][i] == rows[i][0]
                    after = predictions[m][i] == rows[i][0]
                    rescued += after and not before
                    harmed += before and not after
                transitions[m] = dict(rescues=rescued, harms=harmed)
            groups.append(dict(seed=seed, pool=pool, group=g, effects=effects, transitions=transitions,
                               recovery={k: effects[k]['accuracy'] >= .1 and effects[k]['margin'] > 0 for k in ('encoder_native', 'competitor_native', 'joint')},
                               full_joint=index[seed, pool, 'E0M0'][g]['accuracy'] == 1,
                               marker_flag=abs(effects['marker']['accuracy']) > .01))
    primary = [g for g in groups if g['pool'] in ('test_8', 'test_10') and g['group'] in ('distinct', 'repeated')]
    return dict(groups=groups, primary_groups=len(primary),
                all_group_recovery={k: bool(primary) and all(g['recovery'][k] for g in primary) for k in ('encoder_native', 'competitor_native', 'joint')},
                full_joint=bool(primary) and all(g['full_joint'] for g in primary),
                marker_flags=sum(g['marker_flag'] for g in primary))


def audit(directory, save=True):
    directory = Path(directory)
    if save:
        assert not (directory / 'audit.json').exists()
    lock = json.loads((directory / 'lock.json').read_text())
    report = json.loads((directory / 'report.json').read_text())
    assert report['status'] == 'complete' and report['training_updates'] == 0
    assert report['lock_sha256'] == digest(directory / 'lock.json')
    assert report['evidence_sha256'] == digest(directory / 'evidence.pt')
    assert lock['modes'] == list(MODES)
    assert all(digest(ROOT / p) == h for p, h in lock['protected'].items())
    stored = torch.load(directory / 'evidence.pt', weights_only=True)
    expected = {(r['seed'], p, m) for r in lock['runs'] for p in lock['pools'] for m in MODES}
    assert len(report['cells']) == len(expected)
    assert {(c['seed'], c['pool'], c['mode']) for c in report['cells']} == expected
    assert set(stored) == {f'{s}/{p}/{m}' for s, p, m in expected}
    index = {(c['seed'], c['pool'], c['mode']): c for c in report['cells']}
    archive = torch.load(ROOT / lock['archive'], weights_only=True) if lock.get('archive') else None
    maximum = metric_error = 0.
    cells, recomputed = [], {}
    with torch.no_grad():
        for run in lock['runs']:
            assert digest(ROOT / run['checkpoint']) == run['checkpoint_sha256']
            model = make_model('rope').eval()
            model.load_state_dict(torch.load(ROOT / run['checkpoint'], weights_only=True)['model'])
            for pool, rows in lock['pools'].items():
                for mode in MODES:
                    key = f"{run['seed']}/{pool}/{mode}"
                    parts = [reference(model, rows[i:i+64], mode) for i in range(0, len(rows), 64)]
                    ref = {k: torch.cat([v[k] for v in parts]) for k in parts[0]}
                    assert ref.keys() == stored[key].keys()
                    for k, tensor in ref.items():
                        assert tensor.shape == stored[key][k].shape
                        error = float((tensor-stored[key][k]).abs().max())
                        assert error <= 1e-9, (key, k, error)
                        maximum = max(maximum, error)
                    assert torch.equal(ref['logits'].argmax(-1), stored[key]['logits'].argmax(-1))
                    if archive is not None and mode in ('E1M1', 'aligned'):
                        archived = archive[run['seed'], pool, 'native' if mode == 'E1M1' else 'aligned']['logits'][:, 0]
                        error = float((ref['logits']-archived).abs().max())
                        assert error <= 1e-9
                        maximum = max(maximum, error)
                    met = scalar_metrics(rows, ref)
                    metric_error = max(metric_error, close(met, index[run['seed'], pool, mode]['metrics']))
                    cells.append(dict(seed=run['seed'], pool=pool, mode=mode, metrics=met))
                    recomputed[key] = ref
    decision = independent_contrasts(cells, recomputed, lock['pools'])
    metric_error = max(metric_error, close(decision, report['decision']))
    outputs = sum(len(lock['pools'][p]) for _, p, _ in expected)
    assert report['outputs'] == outputs
    if archive is not None:
        assert len(expected) == 144 and outputs == 13104 and decision['primary_groups'] == 9
    assert all(digest(ROOT / p) == h for p, h in lock['protected'].items())
    result = dict(passed=True, cells=len(expected), outputs=outputs, max_tensor_error=maximum,
                  max_metric_error=metric_error, argmax_differences=0, native_archive_checked=archive is not None,
                  protected_files=len(lock['protected']), report_sha256=digest(directory / 'report.json'))
    if save:
        write(directory / 'audit.json', result)
    return result


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    print(json.dumps(audit(DIRECTORY), indent=2))
