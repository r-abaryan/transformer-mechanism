"""Frozen matched retrieval comparison; reuse prior Transformer and datasets."""
import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from mqa_training import model_new, hash_state
from query_exposure import expand_rows, measure
from position_distractors import digest

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2131, 2143, 2161)
ARMS = ('transformer', 'gru')


class RetrievalGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(19, 16)
        self.gru = nn.GRU(16, 27, batch_first=True)
        self.output = nn.Linear(27, 19, bias=False)
        self.double()

    def forward(self, ids):
        hidden, _ = self.gru(self.embedding(ids))
        return self.output(hidden)


def pair(seed):
    torch.manual_seed(seed)
    transformer = model_new('mqa')
    torch.manual_seed(seed)
    gru = RetrievalGRU()
    with torch.no_grad():
        gru.embedding.weight.copy_(transformer.embedding.weight)
    assert [sum(p.numel() for p in m.parameters()) for m in (transformer, gru)] == [4544, 4462]
    return dict(transformer=transformer, gru=gru)


def datasets():
    payload = json.loads((ROOT/'results/mqa_training_data.json').read_text())
    data = {}
    for key, rows in payload['datasets'].items():
        expanded = expand_rows(rows)
        data[key] = (torch.tensor([x for x, _ in expanded]), torch.tensor([y for _, y in expanded]))
    return payload, data


def evaluate(model, data, forward=None):
    x, y = data
    with torch.no_grad():
        scores = torch.cat([(model(c) if forward is None else forward(model, c))[:, -1] for c in x.split(256)])
    return measure(scores, y, x), scores


def decision(runs):
    reliability = {}
    fit = {}
    for arm in ARMS:
        rows = [r for r in runs if r['arm'] == arm]
        fit[arm] = len(rows) == 3 and all(r['status'] == 'trained' and all(
            r['metrics'][f'{split}_{n}'][f'{kind}_accuracy'] >= .99
            for split in ('train', 'validation') for n in (3, 4) for kind in ('repeated', 'unique')) for r in rows)
        reliability[arm] = fit[arm] and all(all(
            r['metrics'][f'test_{n}'][f'{kind}_accuracy'] >= (.90 if n == 6 else .99)
            for n in (3, 4, 6) for kind in ('repeated', 'unique')) for r in rows)
    gains = []
    for seed in SEEDS:
        rows = {r['arm']: r for r in runs if r['seed'] == seed}
        if any(r['status'] != 'trained' for r in rows.values()):
            return dict(reliability=reliability, fit=fit, consistent_advantage=None, gains=[])
        means = {arm: sum(rows[arm]['metrics']['test_6'][f'{kind}_accuracy'] for kind in ('repeated', 'unique'))/2 for arm in ARMS}
        gains.append(dict(seed=seed, **means, gru_minus_transformer=means['gru']-means['transformer']))
    winner = 'gru' if all(g['gru_minus_transformer'] >= .10 for g in gains) else 'transformer' if all(g['gru_minus_transformer'] <= -.10 for g in gains) else None
    return dict(reliability=reliability, fit=fit, consistent_advantage=winner, gains=gains)


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rp = ROOT/'results/recurrent_retrieval.json'
    assert not rp.exists(), 'Preserve existing attempts; no automatic restart.'
    from audit_recurrent_retrieval import preflight, baselines
    payload, data = datasets()
    provenance = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT/'experiments').glob('*.py')}
    for name in ('docs/protocols/recurrent-retrieval.md', 'results/mqa_training_data.json'):
        provenance[name] = digest(ROOT/name)
    report = dict(phase='preflight', torch=torch.__version__, provenance=provenance,
                  preflight=preflight(payload, data), baselines=baselines(data), runs=[])
    write(rp, report)
    directory = ROOT/'results/recurrent_retrieval_checkpoints'
    directory.mkdir(exist_ok=True)
    for index, seed in enumerate(SEEDS):
        models = pair(seed)
        for arm in ARMS[::1 if index % 2 == 0 else -1]:
            model = models[arm]
            run = dict(seed=seed, arm=arm, parameters=sum(p.numel() for p in model.parameters()),
                       initial_sha256=hash_state(model.state_dict()),
                       embedding_sha256=hash_state({'embedding': model.embedding.weight}),
                       status='running', curve=[], metrics={})
            report['runs'].append(run)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            generator = torch.Generator().manual_seed(seed+1000)
            stream = hashlib.sha256()
            tick = time.perf_counter()
            for step in range(1, 6001):
                n = 3 if step % 2 else 4
                x, y = data[f'train_{n}']
                idx = torch.randint(len(x), (64,), generator=generator)
                stream.update(json.dumps(idx.tolist()).encode())
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x[idx])[:, -1], y[idx])
                finite = bool(torch.isfinite(loss))
                if finite:
                    loss.backward()
                    finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                if not finite:
                    run.update(status='failed', failure_step=step)
                    break
                optimizer.step()
                if step % 1000 == 0:
                    run['curve'].append(dict(step=step, minibatch_loss=loss.item()))
                    print(json.dumps(dict(seed=seed, arm=arm, **run['curve'][-1])), flush=True)
            if run['status'] != 'failed':
                run['status'] = 'trained'
            cp = directory/f'{seed}_{arm}.pt'
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step), cp)
            run.update(step=step, elapsed_seconds=time.perf_counter()-tick, stream_sha256=stream.hexdigest(),
                       checkpoint=str(cp.relative_to(ROOT)), checkpoint_sha256=digest(cp))
            if run['status'] == 'trained':
                for key in data:
                    if not key.startswith('test_'):
                        run['metrics'][key], _ = evaluate(model, data[key])
            report['phase'] = 'training'
            write(rp, report)
    report['phase'] = 'all_attempts_finished_before_test'
    write(ROOT/'results/recurrent_retrieval_pretest.json', report)
    for run in report['runs']:
        if run['status'] != 'trained':
            continue
        model = pair(run['seed'])[run['arm']]
        model.load_state_dict(torch.load(ROOT/run['checkpoint'], weights_only=True)['model'])
        scores = {}
        for key in data:
            metric, scores[key] = evaluate(model, data[key])
            if key in run['metrics']:
                assert metric == run['metrics'][key]
            run['metrics'][key] = metric
        sp = directory/f"{run['seed']}_{run['arm']}_scores.pt"
        torch.save(scores, sp)
        run.update(scores=str(sp.relative_to(ROOT)), scores_sha256=digest(sp))
        write(rp, report)
    report.update(phase='complete', decision=decision(report['runs']))
    write(rp, report)
    print(json.dumps(report['decision']), flush=True)


if __name__ == '__main__':
    main()
