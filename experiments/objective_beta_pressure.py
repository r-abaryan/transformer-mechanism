"""Exact training-objective gradients at four immutable retention endpoints."""
import argparse
import json
import math
import time

import torch
from torch.nn import functional as F
import association_load as model_source
import delta_query_coverage as parent

ROOT = parent.ROOT
OUT = ROOT/'results/objective_beta_pressure'
LOADS = (2, 8, 16, 32)
OBJECTIVES = ('assigned', 'all_queries')


def prepare(rows, objective):
    events = torch.tensor([row['events'] for row in rows])
    if objective == 'assigned':
        queries = torch.tensor([[row['query']] for row in rows])
        labels = torch.tensor([[row['label']] for row in rows])
    else:
        assert objective == 'all_queries'
        queries, labels = parent.query_tables(rows)
    repeat = (events[:, None, :, 0] == queries[:, :, None]).sum(-1) > 1
    return events, queries, labels, repeat


def reduce_logits(logits, labels, repeat):
    loss = F.cross_entropy(logits.reshape(-1, 6), labels.flatten(), reduction='none')
    correct = logits.argmax(-1).flatten() == labels.flatten()
    result = {}
    for name, mask in dict(all=torch.ones_like(correct), unique=~repeat.flatten(), repeat=repeat.flatten()).items():
        count = int(mask.sum())
        result[name] = dict(count=count, ce=float(loss[mask].mean().detach()) if count else None,
                            accuracy=float(correct[mask].double().mean()) if count else None)
    return result


def evaluate(model, rows, objective, batch=64):
    events, queries, labels, repeat = prepare(rows, objective)
    params = dict(model.named_parameters())
    gradients = {name:torch.zeros_like(p) for name,p in params.items()}
    outputs = []
    before = model_source.parent.state_hash(model.state_dict())
    for first in range(0, len(rows), batch):
        e, q, y = events[first:first+batch], queries[first:first+batch], labels[first:first+batch]
        state = model.prefill(e)
        logits = torch.stack([model.read(state, q[:,j]) for j in range(q.shape[1])], 1)
        loss = F.cross_entropy(logits.reshape(-1, 6), y.flatten(), reduction='sum') / labels.numel()
        derivative = torch.autograd.grad(loss, tuple(params.values()))
        for (name,_), grad in zip(params.items(), derivative):
            gradients[name] += grad.detach()
        outputs.append(logits.detach())
    assert model_source.parent.state_hash(model.state_dict()) == before
    logits = torch.cat(outputs)
    return dict(logits=logits, labels=labels, repeat=repeat, gradients=gradients,
                metrics=reduce_logits(logits, labels, repeat))


def pressure(gradient):
    return 'lower' if gradient > 1e-7 else 'higher' if gradient < -1e-7 else 'unresolved'


def describe(cell, beta):
    alpha_gradient = float(cell['gradients']['write_logit'])
    factor = beta*(1-beta)
    return dict(metrics=cell['metrics'], beta=beta, sigmoid_factor=factor,
                alpha_gradient=alpha_gradient, beta_gradient=alpha_gradient/factor,
                pressure=pressure(alpha_gradient),
                block_norms={name:float(g.norm()) for name,g in cell['gradients'].items()})


def compare(a, b):
    cosines = {}
    for name in (*a.keys(), 'whole_model'):
        x = torch.cat([t.flatten() for t in a.values()]) if name == 'whole_model' else a[name].flatten()
        y = torch.cat([t.flatten() for t in b.values()]) if name == 'whole_model' else b[name].flatten()
        nx, ny = float(x.norm()), float(y.norm())
        cosines[name] = float(x@y/(nx*ny)) if min(nx,ny) > 1e-12 else None
    pa, pb = pressure(float(a['write_logit'])), pressure(float(b['write_logit']))
    return dict(assigned_pressure=pa, all_query_pressure=pb,
                opposite_resolved_signs={pa,pb} == {'higher','lower'}, gradient_cosines=cosines)


def run():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    assert not OUT.exists(), 'Preserve completed and failed attempts.'
    report_path = parent.OUT/'report.json'
    old = json.loads(report_path.read_text()); audit_path = parent.OUT/'independent_audit.json'
    audit = json.loads(audit_path.read_text())
    assert old['status'] == 'complete' and audit['passed'] and audit['report_sha256'] == parent.study.parent.digest(report_path)
    protected = dict(old['source_dependencies'])
    for name,sha in protected.items(): assert parent.study.parent.digest(ROOT/name) == sha
    data_path = parent.OUT/'development_data.json'
    assert parent.study.parent.digest(data_path) == old['data_sha256']['development']
    for path in (report_path, audit_path, data_path):
        protected[path.relative_to(ROOT).as_posix()] = parent.study.parent.digest(path)
    for run in old['runs']:
        assert parent.study.parent.digest(ROOT/run['checkpoint']) == run['checkpoint_sha256']
        protected[run['checkpoint']] = run['checkpoint_sha256']
    frozen = ('experiments/objective_beta_pressure.py','experiments/audit_objective_beta_pressure.py',
              'tests/test_objective_beta_pressure.py','docs/protocols/objective-beta-pressure.md',
              'results/objective_beta_pressure_preflight.json')
    preflight = json.loads((ROOT/frozen[-1]).read_text()); assert preflight['passed']
    assert preflight['source_hashes'] == {name:parent.study.parent.digest(ROOT/name) for name in frozen[:3]}
    protected.update({name:parent.study.parent.digest(ROOT/name) for name in frozen})
    data = json.loads(data_path.read_text())
    from audit_objective_beta_pressure import check_rows
    for m in LOADS:
        rows = data['pools'][f'train_{m}']; assert len(rows) == 768
        check_rows(rows)
    OUT.mkdir(); parent.study.parent.write(OUT/'lock.json', dict(files=protected, objectives=OBJECTIVES, loads=LOADS))
    report = dict(status='running', torch=torch.__version__, device='cpu', dtype='float64', batch=64,
                  threads=1, training_updates=0, cells=[], comparisons=[], pooled=[])
    saved = []; started = time.perf_counter()
    for run in old['runs']:
        model = model_source.LoadReader('delta').eval()
        model.load_state_dict(torch.load(ROOT/run['checkpoint'], map_location='cpu', weights_only=True)['model'])
        beta = float(model.write_logit.sigmoid().detach())
        by_objective = {o:[] for o in OBJECTIVES}
        for m in LOADS:
            pair = {}
            for objective in OBJECTIVES:
                cell = evaluate(model, data['pools'][f'train_{m}'], objective)
                expected = run['metrics'][f'train_{m}'] if objective == 'assigned' else run['metrics'][f'all_train_{m}']['all_queries']
                assert abs(cell['metrics']['all']['ce'] - expected['loss']) <= 1e-9
                assert abs(cell['metrics']['all']['accuracy'] - expected['accuracy']) <= 1e-12
                record = dict(seed=run['seed'], arm=run['arm'], load=m, objective=objective)
                saved.append(dict(**record, **cell)); report['cells'].append(dict(**record, **describe(cell, beta)))
                by_objective[objective].append(cell); pair[objective] = cell['gradients']
            report['comparisons'].append(dict(seed=run['seed'], arm=run['arm'], load=m, **compare(*pair.values())))
        pooled = {o:{name:sum(c['gradients'][name] for c in cells)/len(LOADS)
                     for name in cells[0]['gradients']} for o,cells in by_objective.items()}
        report['pooled'].append(dict(seed=run['seed'], arm=run['arm'], beta=beta,
            objectives={o:dict(ce=sum(c['metrics']['all']['ce'] for c in by_objective[o])/len(LOADS),
                              alpha_gradient=float(g['write_logit']), beta_gradient=float(g['write_logit'])/(beta*(1-beta)),
                              pressure=pressure(float(g['write_logit'])),block_norms={n:float(t.norm()) for n,t in g.items()})
                        for o,g in pooled.items()}, comparison=compare(*pooled.values())))
        parent.study.parent.write(OUT/'report.json', report)
        print(json.dumps(report['pooled'][-1]), flush=True)
    assert len(saved) == len(old['runs'])*len(LOADS)*len(OBJECTIVES) == 32
    count = sum(c['labels'].numel() for c in saved); assert count == 190464
    torch.save(saved, OUT/'scores.pt')
    report.update(status='complete', cells_count=len(saved), predictions=count, elapsed_seconds=time.perf_counter()-started,
                  scores_sha256=parent.study.parent.digest(OUT/'scores.pt'), lock_sha256=parent.study.parent.digest(OUT/'lock.json'))
    for name,sha in protected.items():assert parent.study.parent.digest(ROOT/name) == sha
    parent.study.parent.write(OUT/'report.json', report)


if __name__ == '__main__':
    run()
