"""Predeclared first-token retrieval experiment; no architecture changes."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch
from torch.nn import functional as F
from verify_decoder import TinyDecoder

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def tensors(rows):
    return torch.tensor([list(r) + [4] for r in rows]), torch.tensor([r[0] for r in rows])


def evaluate(model, data):
    model.eval()
    x, y = data
    with torch.no_grad():
        logits = torch.cat([model(chunk)[:, -1] for chunk in x.split(128)])
        result = dict(accuracy=(logits.argmax(-1) == y).double().mean().item(),
                      loss=F.cross_entropy(logits, y).item())
    model.train()
    return result


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = list(itertools.product(range(4), repeat=6))
    random.Random(20260929).shuffle(rows)
    splits = dict(train=rows[:2048], validation=rows[2048:2560], test=rows[2560:3072])
    rng = random.Random(20260930)
    longer = set()
    while len(longer) < 512:
        longer.add(tuple(rng.randrange(4) for _ in range(10)))
    splits['longer'] = sorted(longer)
    assert not set(splits['train']) & set(splits['test'])
    assert not set(splits['train']) & set(splits['validation'])
    assert not set(splits['test']) & set(splits['validation'])
    data = {name: tensors(values) for name, values in splits.items()}
    majority = max(range(4), key=lambda i: sum(r[0] == i for r in splits['train']))
    baselines = {name: dict(majority_accuracy=sum(r[0] == majority for r in values)/len(values),
                            last_symbol_accuracy=sum(r[0] == r[-1] for r in values)/len(values),
                            oracle_accuracy=1.0, uniform_expected_accuracy=0.25,
                            uniform_loss=math.log(4),
                            target_counts=[sum(r[0] == i for r in values) for i in range(4)])
                 for name, values in splits.items()}
    report = dict(protocol_sha256=hashlib.sha256((ROOT/'docs/first-token-protocol.md').read_bytes()).hexdigest(),
                  source_sha256={name: hashlib.sha256((ROOT/'experiments'/name).read_bytes()).hexdigest()
                                 for name in ('train_first_token.py', 'verify_decoder.py')},
                  torch=torch.__version__, device='cpu', dtype='float64', steps=600,
                  learning_rate=0.1, batch_size=32,
                  split_hashes={name: digest(values) for name, values in splits.items()},
                  baselines=baselines, runs=[])
    checkpoint_dir = ROOT/'results/first_token_checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    for seed in (11, 29, 47, 73, 101):
        torch.manual_seed(seed)
        model = TinyDecoder(vocabulary=5, width=16, heads=2, hidden=32).double()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        generator = torch.Generator().manual_seed(seed+1000)
        curve = [dict(step=0, **evaluate(model, data['validation']))]
        started = time.perf_counter()
        for step in range(1, 601):
            indices = torch.randint(2048, (32,), generator=generator)
            x, y = data['train']
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x[indices])[:, -1], y[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at seed {seed}, step {step}')
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                curve.append(dict(step=step, **evaluate(model, data['validation'])))
        seconds = time.perf_counter()-started
        checkpoint = checkpoint_dir/f'seed_{seed}.pt'
        torch.save(model.state_dict(), checkpoint)
        run = dict(seed=seed, parameters=sum(p.numel() for p in model.parameters()),
                   training_seconds_including_validation=seconds, examples_processed=600*32,
                   tokens_processed=600*32*7, validation_curve=curve,
                   metrics={name: evaluate(model, tensors_) for name, tensors_ in data.items()},
                   checkpoint=str(checkpoint.relative_to(ROOT)),
                   checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        report['runs'].append(run)
        (ROOT/'results/first_token_training.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(seed=seed, seconds=seconds, metrics=run['metrics'])), flush=True)
    report['summary'] = {name: dict(
        mean_accuracy=statistics.mean(r['metrics'][name]['accuracy'] for r in report['runs']),
        sd_accuracy=statistics.stdev(r['metrics'][name]['accuracy'] for r in report['runs']),
        mean_loss=statistics.mean(r['metrics'][name]['loss'] for r in report['runs']),
        all_seeds_above_95=all(r['metrics'][name]['accuracy'] >= .95 for r in report['runs']))
        for name in splits}
    (ROOT/'results/first_token_training.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
