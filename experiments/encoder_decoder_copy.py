"""Frozen copy-task learning and free-running generation on the audited model."""
import hashlib
import json
import random
import time

import torch
from torch.nn import functional as F
from encoder_decoder_contract import ROOT, Model
from position_distractors import digest
from recurrent_retrieval import write
from mqa_training import hash_state

SEEDS = (2371, 2377, 2381)
COUNTS = {'train_3':128, 'train_4':256, 'validation_3':64, 'validation_4':64,
          'test_3':64, 'test_4':64, 'test_6':128, 'test_8':128}
PREFIX = 'encoder_decoder_copy'


def make_data():
    rng = random.Random(2026100203)
    seen, pools = set(), {}
    for name, count in COUNTS.items():
        length = int(name.rsplit('_', 1)[1])
        rows = []
        for distinct in (True, False):
            selected = 0
            for attempt in range(100000):
                row = tuple(rng.sample(range(3, 11), length)) if distinct else tuple(rng.choices(range(3, 11), k=length))
                if (len(set(row)) == length) != distinct or row in seen:
                    continue
                seen.add(row)
                rows.append(list(row))
                selected += 1
                if selected == count//2:
                    break
            assert selected == count//2, 'Frozen proposal budget exhausted.'
        pools[name] = rows
    return dict(generator_seed=2026100203, pools=pools)


def inputs(source):
    labels = torch.cat((source, torch.full((len(source), 1), 2, dtype=torch.long)), 1)
    shifted = torch.cat((torch.ones(len(source), 1, dtype=torch.long), source), 1)
    return shifted, labels


def logits(model, source, prefix):
    s, t = model.features(source, prefix)
    return model(s, t, torch.zeros_like(source, dtype=torch.bool), torch.zeros_like(prefix, dtype=torch.bool))


def generate(next_logits, source, limit):
    prefix = torch.ones(len(source), 1, dtype=torch.long)
    results = [[] for _ in source]
    finished = torch.zeros(len(source), dtype=torch.bool)
    for _ in range(limit):
        chosen = next_logits(source, prefix).argmax(-1)
        for i, token in enumerate(chosen.tolist()):
            if not bool(finished[i]):
                results[i].append(token)
        finished = finished | (chosen == 2)
        prefix = torch.cat((prefix, chosen[:, None]), 1)
        if bool(finished.all()):
            break
    return results


def metrics(scores, source, generated):
    _, labels = inputs(source)
    token_loss = F.cross_entropy(scores.flatten(0, 1), labels.flatten(), reduction='none').reshape_as(labels)
    correct = scores.argmax(-1) == labels
    exact = torch.tensor([row == x+[2] for row, x in zip(generated, source.tolist())])
    n = source.shape[1]
    eos = {name:torch.tensor([bool(row and row[-1] == 2) and compare(len(row), n+1) for row in generated])
           for name, compare in (('early_eos', lambda a,b:a<b), ('on_time_eos', lambda a,b:a==b), ('late_eos', lambda a,b:a>b))}
    eos['missing_eos'] = torch.tensor([not row or row[-1] != 2 for row in generated])
    distinct = torch.tensor([len(set(row)) == n for row in source.tolist()])
    out = {}
    for name, mask in (('all',torch.ones(len(source),dtype=torch.bool)), ('distinct',distinct), ('repeated',~distinct)):
        assert bool(mask.any())
        row = dict(sequences=int(mask.sum()), targets=int(mask.sum())*(n+1),
                   exact_accuracy=exact[mask].double().mean().item(),
                   token_accuracy=correct[mask].double().mean().item(),
                   content_accuracy=correct[mask,:n].double().mean().item(), loss=token_loss[mask].mean().item())
        row.update({k:v[mask].double().mean().item() for k,v in eos.items()})
        out[name] = row
    return out


def evaluate(model, source):
    scores, generated = [], []
    with torch.no_grad():
        for x in source.split(64):
            shifted, _ = inputs(x)
            scores.append(logits(model, x, shifted))
            generated.extend(generate(lambda a,b:logits(model,a,b)[:, -1], x, x.shape[1]+3))
    scores = torch.cat(scores)
    return metrics(scores, source, generated), dict(logits=scores, generated=generated)


def decision(runs):
    ready = len(runs) == 3 and all(r['status'] == 'trained' for r in runs)
    short = ready and all(r['metrics'][f'{split}_{n}'][g]['exact_accuracy'] >= .95
        for r in runs for split in ('train','validation') for n in (3,4) for g in ('distinct','repeated'))
    held = short and all(r['metrics'][f'test_{n}'][g]['exact_accuracy'] >= .95
        for r in runs for n in (3,4) for g in ('distinct','repeated'))
    transfer = {str(n):short and all(r['metrics'][f'test_{n}'][g]['exact_accuracy'] >= .90
        for r in runs for g in ('distinct','repeated')) for n in (6,8)}
    return dict(short_fit=short, short_held_out=held, length_transfer=transfer)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rp, ip = ROOT/f'results/{PREFIX}.json', ROOT/f'results/{PREFIX}_data.json'
    assert not rp.exists() and not ip.exists(), 'Preserve attempts; no restart.'
    payload = make_data()
    write(ip, payload)
    data = {k:torch.tensor(v) for k,v in payload['pools'].items()}
    from audit_encoder_decoder_copy import audit_inputs, baselines, controller_checks
    files = [*sorted((ROOT/'experiments').glob('*.py')), ROOT/'docs/protocols/encoder-decoder-copy.md']
    report = dict(phase='preflight', device='cpu', dtype='float64', torch=torch.__version__,
                  seeds=list(SEEDS), updates=6000, batch=64, sequences_per_run=384000, targets_per_run=1728000,
                  gpu_memory='not used or measured', input_sha256=digest(ip),
                  provenance={str(p.relative_to(ROOT)):digest(p) for p in files},
                  historical={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'results').glob('*.json') if not p.name.startswith(PREFIX)},
                  input_checks=audit_inputs(payload), baselines=baselines(payload), controller_checks=controller_checks(), runs=[])
    write(rp, report)
    directory = ROOT/f'results/{PREFIX}_checkpoints'
    directory.mkdir(exist_ok=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = Model()
        run = dict(seed=seed, parameters=sum(p.numel() for p in model.parameters()), status='running',
                   initial_sha256=hash_state(model.state_dict()), curve=[], metrics={})
        report['runs'].append(run)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        generator = torch.Generator().manual_seed(seed+1000)
        stream = hashlib.sha256()
        tick = time.perf_counter()
        for step in range(1, 6001):
            n = 3 if step % 2 else 4
            pool = data[f'train_{n}']
            half = len(pool)//2
            idx = torch.cat((torch.randint(half,(32,),generator=generator), torch.randint(half,(32,),generator=generator)+half))
            stream.update(json.dumps(idx.tolist()).encode())
            x = pool[idx]
            shifted, labels = inputs(x)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(logits(model, x, shifted).flatten(0,1), labels.flatten())
            finite = bool(torch.isfinite(loss))
            if finite:
                loss.backward()
                finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            if not finite:
                run.update(status='failed', failure_step=step)
                break
            optimizer.step()
            if step % 1000 == 0:
                row = dict(step=step, minibatch_loss=loss.item())
                if step in (1000,3000,6000):
                    row['pools'] = {key:evaluate(model,x)[0] for key,x in data.items() if not key.startswith('test_')}
                run['curve'].append(row)
                progress = dict(seed=seed, step=step, minibatch_loss=loss.item())
                if 'pools' in row:
                    progress['train4_exact'] = row['pools']['train_4']['all']['exact_accuracy']
                    progress['validation4_exact'] = row['pools']['validation_4']['all']['exact_accuracy']
                print(json.dumps(progress), flush=True)
        if run['status'] != 'failed':
            run.update(status='trained', metrics=run['curve'][-1]['pools'].copy())
        cp = directory/f'{seed}.pt'
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step), cp)
        run.update(step=step, elapsed_seconds=time.perf_counter()-tick, stream_sha256=stream.hexdigest(),
                   checkpoint=str(cp.relative_to(ROOT)), checkpoint_sha256=digest(cp))
        report['phase'] = 'training'
        write(rp, report)
    report['phase'] = 'all_attempts_finished_before_test'
    write(ROOT/f'results/{PREFIX}_pretest.json', report)
    for run in report['runs']:
        if run['status'] != 'trained':continue
        model = Model()
        model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        details = {}
        for key,x in data.items():
            metric,details[key] = evaluate(model,x)
            if key in run['metrics']:assert metric == run['metrics'][key]
            run['metrics'][key] = metric
        sp = directory/f"{run['seed']}_scores.pt"
        torch.save(details,sp)
        run.update(scores=str(sp.relative_to(ROOT)), scores_sha256=digest(sp))
        write(rp,report)
    report.update(phase='complete',decision=decision(report['runs']))
    write(rp,report)
    print(json.dumps(report['decision']),flush=True)


if __name__ == '__main__':main()
