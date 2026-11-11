"""Bounded zero-update context interventions on saved WikiText models."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from wikitext2_training import ROOT, configure, model_new, load_stream, angles
from mqa_decoding import chunk

SEEDS = (2183, 2203, 2213)
CONDITIONS = ('original', 'recent_only', 'earlier_shuffle', 'recent_shuffle')
OUT = ROOT / 'results/language_context_use'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def make_inputs(stream):
    n = (len(stream) - 1) // 64
    starts = torch.arange(n) * 64
    ids = stream[starts[:, None] + torch.arange(64)]
    generators = [torch.Generator().manual_seed(s) for s in (202610041, 202610042)]
    return dict(ids=ids, targets=stream[starts + 64], starts=starts,
                earlier_order=torch.stack([torch.randperm(32, generator=generators[0]) for _ in starts]),
                recent_order=torch.stack([torch.randperm(31, generator=generators[1]) for _ in starts]))


def transform(inputs, condition, first, last):
    x = inputs['ids'][first:last].clone()
    if condition == 'recent_only':
        return x[:, 32:], 32
    if condition == 'earlier_shuffle':
        x[:, :32] = x[:, :32].gather(1, inputs['earlier_order'][first:last])
    elif condition == 'recent_shuffle':
        x[:, 32:63] = x[:, 32:63].gather(1, inputs['recent_order'][first:last])
    else:
        assert condition == 'original'
    return x, 0


def logits(model, ids, offset):
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    c, s = angles(64, device, dtype)
    cache = [tuple(torch.empty(len(ids), model.groups, ids.shape[1], 8, device=device, dtype=dtype)
                   for _ in range(2)) for _ in model.blocks]
    return chunk(model, ids.to(device), cache, 0, c[offset:], s[offset:])[:, -1]


def masks(targets):
    return dict(all=torch.ones_like(targets, dtype=torch.bool), ordinary=targets > 1,
                unknown=targets == 0, row_end=targets == 1)


def metrics(cell, base=None):
    result = {}
    for name, mask in masks(cell['targets']).items():
        count = int(mask.sum())
        row = dict(count=count, ce=None, perplexity=None, accuracy=None)
        if count:
            loss = cell['losses'][mask].double()
            correct = cell['predictions'][mask] == cell['targets'][mask]
            ce = float(loss.mean())
            row.update(ce=ce, perplexity=math.exp(ce), accuracy=float(correct.double().mean()))
            if base is not None:
                change = loss - base['losses'][mask].double()
                before = base['predictions'][mask] == cell['targets'][mask]
                delta = float(change.mean())
                row.update(delta_ce=delta, perplexity_ratio=math.exp(delta),
                           loss_improved=int((change < 0).sum()), loss_worsened=int((change > 0).sum()),
                           loss_equal=int((change == 0).sum()), rescues=int((~before & correct).sum()),
                           harms=int((before & ~correct).sum()))
        result[name] = row
    return result


@torch.inference_mode()
def evaluate(model, inputs, condition, batch=8):
    losses, predictions, fixtures = [], [], []
    for first in range(0, len(inputs['targets']), batch):
        ids, offset = transform(inputs, condition, first, first + batch)
        scores = logits(model, ids, offset)
        target = inputs['targets'][first:first + batch].to(scores.device)
        losses.append(F.cross_entropy(scores, target, reduction='none').cpu())
        predictions.append(scores.argmax(-1).cpu())
        if first == 0:
            fixtures.append(dict(first=0, logits=scores[:2].cpu()))
    cell = dict(losses=torch.cat(losses), predictions=torch.cat(predictions),
                targets=inputs['targets'].clone(), fixtures=fixtures)
    assert bool(torch.isfinite(cell['losses']).all())
    return cell


def main():
    configure()
    assert torch.cuda.is_available() and not OUT.exists(), 'Preserve prior attempts.'
    parent_path = ROOT / 'results/recurrent_language.json'
    parent = json.loads(parent_path.read_text())
    audit_path = ROOT / 'results/recurrent_language_audit.json'
    audit = json.loads(audit_path.read_text())
    assert parent['phase'] == 'complete' and audit['passed']
    assert audit['report_sha256'] == digest(parent_path)
    for path, sha in parent['provenance'].items():
        assert digest(ROOT / path) == sha, path
    runs = [r for r in parent['runs'] if r['arm'] == 'transformer']
    assert {r['seed'] for r in runs} == set(SEEDS)
    encoding = json.loads((ROOT / 'results/wikitext2_encoding.json').read_text())
    stream = load_stream('validation')
    inputs = make_inputs(stream)
    assert len(inputs['targets']) == 3400 and len(stream) - 1 == 217645
    from audit_language_context_use import verify_inputs
    verify_inputs(inputs, stream)
    protected = dict(parent['provenance'])
    for path in (parent_path, audit_path, ROOT / encoding['splits']['validation']['path']):
        protected[str(path.relative_to(ROOT))] = digest(path)
    for run in runs:
        for key in ('checkpoint', 'scores'):
            assert digest(ROOT / run[key]) == run[key + '_sha256']
            protected[run[key]] = run[key + '_sha256']
    for path in ('experiments/language_context_use.py', 'experiments/audit_language_context_use.py',
                 'tests/test_language_context_use.py', 'docs/protocols/language-context-use.md',
                 'results/language_context_use_preflight.json', 'experiments/mqa_decoding.py',
                 'experiments/audit_kv_cache_equivalence.py'):
        protected[path] = digest(ROOT / path)
    OUT.mkdir()
    torch.save(inputs, OUT / 'inputs.pt')
    write(OUT / 'lock.json', dict(files=protected, inputs_sha256=digest(OUT / 'inputs.pt')))
    report = dict(phase='running', seeds=list(SEEDS), conditions=list(CONDITIONS), targets_per_cell=3400,
                  training_updates=0, vocabulary=encoding['vocabulary'], torch=torch.__version__,
                  device=torch.cuda.get_device_name(0), dtype='float32', batch=8, tf32=False,
                  lock_sha256=digest(OUT / 'lock.json'), cells=[], archived_baseline_checks=[])
    write(OUT / 'report.json', report)
    saved = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for seed in SEEDS:
        run = next(r for r in runs if r['seed'] == seed)
        model = model_new('mqa', encoding['vocabulary']).float().cuda().eval()
        model.load_state_dict(torch.load(ROOT / run['checkpoint'], map_location='cpu', weights_only=True)['model'])
        cells = {c: evaluate(model, inputs, c) for c in CONDITIONS}
        old = torch.load(ROOT / run['scores'], map_location='cpu', weights_only=True)['validation']
        indices = inputs['starts'] + 63
        assert torch.equal(old['targets'][indices], inputs['targets'])
        err = float((old['losses'][indices].double() - cells['original']['losses'].double()).abs().max())
        assert err <= 1e-4
        report['archived_baseline_checks'].append(dict(seed=seed, max_ce_error=err,
            argmax_differences=int((old['predictions'][indices] != cells['original']['predictions']).sum())))
        for condition, cell in cells.items():
            saved.append(dict(seed=seed, condition=condition, **cell))
            report['cells'].append(dict(seed=seed, condition=condition, metrics=metrics(cell, cells['original'])))
        del model
        print(json.dumps(dict(seed=seed, metrics=[c for c in report['cells'] if c['seed'] == seed])))
    torch.cuda.synchronize()
    report.update(phase='complete', elapsed_seconds=time.perf_counter()-started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), predictions=3*4*3400)
    report['changed_windows'] = {c: int((transform(inputs, c, 0, 3400)[0] != inputs['ids']).any(1).sum())
                                 for c in ('earlier_shuffle', 'recent_shuffle')}
    torch.save(saved, OUT / 'scores.pt')
    report['scores_sha256'] = digest(OUT / 'scores.pt')
    write(OUT / 'report.json', report)


if __name__ == '__main__':
    main()
