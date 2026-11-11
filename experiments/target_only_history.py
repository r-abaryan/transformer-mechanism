"""Fixed-checkpoint, query-conditioned removal of competing writes."""
import argparse
import csv
import importlib.util
import json
import unittest

import torch
import feedback_loss_decomposition as previous
import readout_adam_transport as adam

training = adam.training
ROOT = previous.ROOT
OUT = ROOT / 'results/target_only_history'
SOURCES = ('experiments/target_only_history.py', 'experiments/audit_target_only_history.py',
           'tests/test_target_only_history.py')
PROTOCOL = 'docs/protocols/target-only-history.md'


def filtered_queries(rows):
    events, queries, labels, repeat = training.replay.pressure.prepare(rows, 'all_queries')
    selected = []; counts = []
    for row, qs in zip(rows, queries.tolist()):
        for q in qs:
            writes = [e for e in row['events'] if e[0] == q]
            assert writes and len({v for k, v in writes}) == 1, 'Stable target values required'
            selected.append(writes); counts.append(len(writes))
    return selected, queries, labels, repeat, torch.tensor(counts).reshape(labels.shape)


def transitions(original, filtered, labels, repeat):
    a = original.argmax(-1) == labels; b = filtered.argmax(-1) == labels
    groups = {}
    for name in ('all', 'unique', 'repeat'):
        scope = torch.ones_like(a) if name == 'all' else repeat == (name == 'repeat')
        groups[name] = dict(count=int(scope.sum()), correct_both=int((scope & a & b).sum()),
            wrong_both=int((scope & ~a & ~b).sum()), rescued=int((scope & ~a & b).sum()),
            newly_wrong=int((scope & a & ~b).sum()))
    return groups


def evaluate(model, rows, batch=256):
    before = {k: v.clone() for k, v in model.state_dict().items()}; rng = torch.get_rng_state().clone()
    selected, queries, labels, repeat, counts = filtered_queries(rows)
    q = queries.flatten(); output = model.output.weight.new_empty(len(selected), 6)
    # There are no padding writes. Each group has its actual target-only length.
    with torch.no_grad():
        for n in sorted(set(counts.flatten().tolist())):
            ids = torch.where(counts.flatten() == n)[0]
            for start in range(0, len(ids), batch):
                chosen = ids[start:start + batch]
                e = torch.tensor([selected[int(i)] for i in chosen])
                output[chosen] = model(e, q[chosen])
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()), 'Weights changed'
    assert torch.equal(rng, torch.get_rng_state()), 'RNG changed'
    z = output.reshape(*labels.shape, 6)
    return dict(logits=z, labels=labels, repeat=repeat, target_writes=counts,
                metrics=training.reduce_scores(z, labels, repeat))


def inputs():
    pairs, _, _ = previous.inputs()
    return [(cell, rows) for seed in (3461, 3463) for step in (3000, 12000)
            for a, b, rows in pairs if a['seed'] == seed for cell in ((a if step == 3000 else b),)]


def preflight():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('test_target_only_history', ROOT / SOURCES[-1])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    record = dict(passed=result.wasSuccessful(), methods=result.testsRun,
                  sources={n: training.replay.digest(ROOT / n) for n in SOURCES})
    name = 'target_only_history_preflight' + ('' if record['passed'] else '_failed') + '.json'
    path = ROOT / 'results' / name; assert not path.exists()
    training.replay.write(path, record); assert record['passed']


def run():
    torch.set_num_threads(1); assert not OUT.exists(), 'Preserve evidence'
    ready_path = ROOT / 'results/target_only_history_preflight.json'
    ready = json.loads(ready_path.read_text())
    assert ready['passed'] and ready['sources'] == {n: training.replay.digest(ROOT / n) for n in SOURCES}
    files = dict(json.loads((adam.OUT / 'lock.json').read_text())['files'])
    old_audit = json.loads((adam.OUT / 'audit.json').read_text())
    assert old_audit['passed'] and old_audit['report_sha256'] == training.replay.digest(adam.OUT / 'report.json')
    for name in ('report.json', 'audit.json', 'metrics.csv', 'lock.json'):
        p = adam.OUT / name; files[p.relative_to(ROOT).as_posix()] = training.replay.digest(p)
    for name in (*SOURCES, PROTOCOL, ready_path.relative_to(ROOT).as_posix()):
        files[name] = training.replay.digest(ROOT / name)
    training.replay.check_files(files); OUT.mkdir()
    training.replay.write(OUT / 'lock.json', dict(files=files, seeds=[3461,3463], steps=[3000,12000],
        splits=['train','validation','test'], loads=[2,8,16,32], answers=311808, cells=48,
        intervention='keep only writes whose key equals this query; no padding', tolerance=1e-9))
    cells = []; scores = []
    for source, rows in inputs():
        model = training.previous.load_model(ROOT / source['checkpoint'])
        filtered = evaluate(model, rows)
        assert torch.equal(filtered['labels'], source['labels']) and torch.equal(filtered['repeat'], source['repeat'])
        metadata = {k: source[k] for k in ('seed','step','pool','checkpoint')}
        counts = filtered['target_writes']
        assert counts.unique().tolist() == [32 // int(source['pool'].split('_')[1])]
        scores.append(dict(**metadata, **filtered))
        cells.append(dict(**metadata, original=source['metrics'], filtered=filtered['metrics'],
            transitions=transitions(source['logits'], filtered['logits'], filtered['labels'], filtered['repeat']),
            target_writes=int(counts[0,0]), retained_events=int(counts.sum()),
            beta=float(model.write_logit.sigmoid().detach())))
        print(f"Evaluated {source['seed']} step {source['step']} {source['pool']}", flush=True)
    answers = sum(c['labels'].numel() for c in scores)
    assert len(cells) == 48 and answers == 2*2*(768+192+384)*sum((2,8,16,32)) == 311808
    assert sum(c['retained_events'] for c in cells) == 2*2*(768+192+384)*4*32 == 688128
    torch.save(scores, OUT / 'scores.pt')
    report = dict(status='complete', training_updates=0, optimizer_calls=0, device='cpu', dtype='float64',
        cells=cells, cells_count=48, evaluated_answers=answers, retained_events=688128,
        original_scores_reused=True, scores_sha256=training.replay.digest(OUT/'scores.pt'),
        lock_sha256=training.replay.digest(OUT/'lock.json'))
    training.replay.check_files(files); training.replay.write(OUT/'report.json', report)
    fields = ('seed','step','pool','group','count','original_accuracy','filtered_accuracy','original_ce',
              'filtered_ce','correct_both','wrong_both','rescued','newly_wrong')
    with (OUT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for c in cells:
            for group, t in c['transitions'].items():
                d = {k:c[k] for k in ('seed','step','pool')}; d.update(group=group, **t)
                for endpoint in ('original','filtered'):
                    for metric in ('accuracy','ce'): d[f'{endpoint}_{metric}'] = c[endpoint]['groups'][group][metric]
                writer.writerow(d)
    print(json.dumps({k:v for k,v in report.items() if k!='cells'}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--preflight',action='store_true')
    if p.parse_args().preflight: preflight()
    else: run()
