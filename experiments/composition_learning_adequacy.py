"""Frozen GPU feasibility study; archived graph task and sources stay unchanged."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import gc
import hashlib
import json
import math
import platform
import time

import torch
from torch import nn
from torch.nn import functional as F
from recurrent_composition import ROOT, ARMS, expand, metrics
from recurrent_retrieval import write
from mqa_training import hash_state
from position_distractors import digest
from wikitext2_training import configure

SEEDS = (2267, 2273, 2281)
EXAMPLES = 3_072_000
EVAL_BATCH = 512
PREFIX = 'composition_learning_adequacy'
PROTOCOL = ROOT/'docs/protocols/composition-learning-adequacy.md'
DATA = ROOT/'results/recurrent_composition_data.json'


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(64)
        self.q = nn.Linear(64, 64)
        self.k = nn.Linear(64, 32)
        self.v = nn.Linear(64, 32)
        self.out = nn.Linear(64, 64)
        self.norm2 = nn.LayerNorm(64)
        self.up = nn.Linear(64, 128)
        self.down = nn.Linear(128, 64)


class LookupTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(15, 64)
        self.blocks = nn.ModuleList([Block(), Block()])
        self.norm = nn.LayerNorm(64)
        self.output = nn.Linear(64, 15, bias=False)

    def forward(self, ids):
        x = self.embedding(ids)
        b, n, _ = x.shape
        angles = (torch.arange(n, device=x.device, dtype=torch.float64)+4)[:, None]
        angles = angles*10000.**(-torch.arange(0, 32, 2, device=x.device, dtype=torch.float64)/32)
        c, s = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        mask = torch.ones(n, n, device=x.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            z = block.norm1(x)
            q = block.q(z).view(b, n, 2, 32).transpose(1, 2)
            k = block.k(z).view(b, n, 1, 32).transpose(1, 2)
            v = block.v(z).view(b, n, 1, 32).transpose(1, 2)
            rotated = []
            for a in (q, k):
                even, odd = a[..., 0::2], a[..., 1::2]
                rotated.append(torch.stack((even*c-odd*s, even*s+odd*c), -1).flatten(-2))
            q, k = rotated
            weights = (q@k.transpose(-1, -2)/math.sqrt(32)).masked_fill(mask, -torch.inf).softmax(-1)
            # Broadcasting shares the one KV head across the two query heads.
            h = (weights@v).transpose(1, 2).reshape(b, n, 64)
            x = x+block.out(h)
            x = x+block.down(F.relu(block.up(block.norm2(x))))
        return self.output(self.norm(x))


class LookupGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(15, 64)
        self.gru = nn.GRU(64, 110, batch_first=True)
        self.output = nn.Linear(110, 15, bias=False)

    def forward(self, ids):
        states, _ = self.gru(self.embedding(ids))
        return self.output(states)


def pair(seed):
    torch.manual_seed(seed)
    transformer = LookupTransformer()
    torch.manual_seed(seed)
    gru = LookupGRU()
    with torch.no_grad():
        gru.embedding.weight.copy_(transformer.embedding.weight)
    assert [sum(p.numel() for p in m.parameters()) for m in (transformer, gru)] == [60672, 60690]
    return dict(transformer=transformer, gru=gru)


def evaluate(model, pool):
    x, y = pool
    device = next(model.parameters()).device
    with torch.no_grad():
        scores = torch.cat([model(c.to(device))[:, -1].cpu() for c in x.split(EVAL_BATCH)])
    return metrics(scores.double(), x, y), scores


def decision(runs):
    one_fit, one_transfer, two_fit, two_transfer = {}, {}, {}, {}
    for arm in ARMS:
        rows = [r for r in runs if r['arm'] == arm]
        completed = len(rows) == len(SEEDS) and all(r['status'] == 'trained' for r in rows)
        one_fit[arm] = completed and all(r['metrics'][f'{split}_{n}']['one_step']['accuracy'] >= .99
            for r in rows for split in ('train', 'validation') for n in (4, 6))
        one_transfer[arm] = one_fit[arm] and all(r['metrics'][f'test_{n}']['one_step']['accuracy'] >= (.90 if n == 8 else .99)
            for r in rows for n in (4, 6, 8))
        two_fit[arm] = completed and all(r['metrics'][f'{split}_{n}'][g]['accuracy'] >= .99
            for r in rows for split in ('train', 'validation') for n in (4, 6)
            for g in ('two_step', 'two_forward', 'two_reverse'))
        two_transfer[arm] = two_fit[arm] and all(r['metrics'][f'test_{n}'][g]['accuracy'] >= (.90 if n == 8 else .99)
            for r in rows for n in (4, 6, 8) for g in ('two_forward', 'two_reverse'))
    return dict(one_step_fit=one_fit, one_step_transfer=one_transfer,
                composition_comparison_eligible=all(one_fit.values()),
                two_step_fit=two_fit, two_step_transfer=two_transfer)


def provenance():
    return {str(p.relative_to(ROOT)):digest(p) for p in [*sorted((ROOT/'experiments').glob('*.py')), PROTOCOL]}


def profile():
    configure()
    path = ROOT/f'results/{PREFIX}_profile.json'
    assert not path.exists(), 'Preserve hardware profiling; no automatic overwrite.'
    assert torch.cuda.is_available()
    report = dict(provenance=provenance(), input_sha256=digest(DATA),
                  torch=torch.__version__, python=platform.python_version(), platform=platform.platform(),
                  device=torch.cuda.get_device_name(), total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                  cap_bytes=6*1024**3, cases=[])
    for batch in (256, 512, 1024):
        for arm in ARMS:
            for length in (11, 15, 19):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                case = dict(batch=batch, arm=arm, tokens=length, status='running')
                model = optimizer = x = y = loss = None
                try:
                    model = pair(2026100202)[arm].cuda()
                    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                    g = torch.Generator().manual_seed(2026100202)
                    x = torch.randint(15, (batch, length), generator=g).cuda()
                    y = torch.randint(15, (batch,), generator=g).cuda()
                    tick = time.perf_counter()
                    for _ in range(3):
                        optimizer.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(model(x)[:, -1], y)
                        loss.backward()
                        optimizer.step()
                    torch.cuda.synchronize()
                    case.update(status='completed', elapsed_seconds=time.perf_counter()-tick)
                except torch.cuda.OutOfMemoryError as e:
                    case.update(status='oom', error=str(e))
                case.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved())
                report['cases'].append(case)
                del model, optimizer, x, y, loss
                write(path, report)
    candidates = [b for b in (256, 512, 1024) if all(c['status'] == 'completed' and
        max(c['peak_allocated_bytes'], c['peak_reserved_bytes']) < report['cap_bytes']
        for c in report['cases'] if c['batch'] == b)]
    report['selected_batch'] = max(candidates) if candidates else None
    if candidates:
        batch = report['selected_batch']
        assert EXAMPLES % batch == 0
        report.update(examples_per_run=EXAMPLES, updates_per_run=EXAMPLES//batch,
                      effective_batch=batch, accumulation_steps=1, phase='frozen_before_task_preflight')
    else:
        report['phase'] = 'blocked_before_task_training'
    write(path, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','provenance')}), flush=True)


def main():
    configure()
    rp = ROOT/f'results/{PREFIX}.json'
    pp = ROOT/f'results/{PREFIX}_profile.json'
    assert not rp.exists(), 'Preserve attempts; no automatic restart.'
    frozen = json.loads(pp.read_text())
    assert frozen['phase'] == 'frozen_before_task_preflight'
    assert frozen['provenance'] == provenance() and frozen['input_sha256'] == digest(DATA)
    batch, updates = frozen['selected_batch'], frozen['updates_per_run']
    marks = (updates//6, updates//3, 2*updates//3, updates)
    assert len(set(marks)) == 4 and batch % 2 == 0
    payload = json.loads(DATA.read_text())
    data = expand(payload)
    from audit_recurrent_composition import audit_inputs, baselines
    from audit_composition_learning_adequacy import preflight
    historical = {str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'results').glob('recurrent_*.json')}
    for directory in (ROOT/'results').glob('recurrent_*_checkpoints'):
        for p in directory.glob('*.pt'):
            historical[str(p.relative_to(ROOT))] = digest(p)
    report = dict(phase='preflight', profile=str(pp.relative_to(ROOT)), profile_sha256=digest(pp),
                  provenance=provenance(), historical=historical, input_sha256=digest(DATA), seeds=list(SEEDS),
                  effective_batch=batch, updates_per_run=updates, examples_per_run=EXAMPLES,
                  input_checks=audit_inputs(payload, data), baselines=baselines(data),
                  preflight=preflight(data), runs=[])
    write(rp, report)
    directory = ROOT/f'results/{PREFIX}_checkpoints'
    directory.mkdir(exist_ok=True)
    groups = {n:[torch.where(data[f'train_{n}'][0][:, -1] == h)[0] for h in (13, 14)] for n in (4, 6)}
    gpu_data = {n:tuple(t.cuda() for t in data[f'train_{n}']) for n in (4, 6)}
    for index, seed in enumerate(SEEDS):
        models = pair(seed)
        for arm in ARMS[::1 if index % 2 == 0 else -1]:
            model = models[arm]
            run = dict(seed=seed, arm=arm, parameters=sum(p.numel() for p in model.parameters()),
                       initial_sha256=hash_state(model.state_dict()),
                       embedding_sha256=hash_state({'embedding':model.embedding.weight}),
                       status='running', curve=[], metrics={})
            report['runs'].append(run)
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            generator = torch.Generator().manual_seed(seed+1000)
            stream = hashlib.sha256()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            for step in range(1, updates+1):
                n = 4 if step % 2 else 6
                x, y = gpu_data[n]
                idx = torch.cat([g[torch.randint(len(g), (batch//2,), generator=generator)] for g in groups[n]])
                stream.update(json.dumps(idx.tolist()).encode())
                idx = idx.cuda()
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x[idx])[:, -1], y[idx])
                finite = bool(torch.isfinite(loss))
                if finite:
                    loss.backward()
                    gradients = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
                    finite = gradients.numel() == run['parameters'] and bool(torch.isfinite(gradients).all())
                    del gradients
                if not finite:
                    run.update(status='failed', failure_step=step, reason='Nonfinite loss or missing/nonfinite gradient')
                    break
                optimizer.step()
                if step in marks:
                    row = dict(step=step, minibatch_loss=loss.item(),
                        pools={key:evaluate(model, pool)[0] for key, pool in data.items() if not key.startswith('test_')})
                    run['curve'].append(row)
                    print(json.dumps(dict(seed=seed, arm=arm, step=step, minibatch_loss=loss.item(),
                        validation6_one=row['pools']['validation_6']['one_step']['accuracy'],
                        validation6_two=row['pools']['validation_6']['two_step']['accuracy'])), flush=True)
            torch.cuda.synchronize()
            if run['status'] != 'failed':
                run.update(status='trained', metrics=run['curve'][-1]['pools'].copy())
            cp = directory/f'{seed}_{arm}.pt'
            torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            optimizer=optimizer.state_dict(), step=step), cp)
            run.update(step=step, elapsed_seconds=time.perf_counter()-tick, stream_sha256=stream.hexdigest(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                checkpoint=str(cp.relative_to(ROOT)), checkpoint_sha256=digest(cp))
            model.cpu()
            del optimizer, loss, idx
            report['phase'] = 'training'
            write(rp, report)
    report['phase'] = 'all_attempts_finished_before_test'
    write(ROOT/f'results/{PREFIX}_pretest.json', report)
    for run in report['runs']:
        if run['status'] != 'trained':
            continue
        model = pair(run['seed'])[run['arm']].cuda()
        model.load_state_dict(torch.load(ROOT/run['checkpoint'], map_location='cpu', weights_only=True)['model'])
        scores = {}
        for key, pool in data.items():
            metric, scores[key] = evaluate(model, pool)
            if key in run['metrics']:
                assert metric == run['metrics'][key]
            run['metrics'][key] = metric
        sp = directory/f"{run['seed']}_{run['arm']}_scores.pt"
        torch.save(scores, sp)
        run.update(scores=str(sp.relative_to(ROOT)), scores_sha256=digest(sp))
        del model
        write(rp, report)
    report.update(phase='complete', decision=decision(report['runs']))
    write(rp, report)
    print(json.dumps(report['decision']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    profile() if args.profile else main()
