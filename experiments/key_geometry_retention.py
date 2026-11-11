"""All-query retention under fixed, query-independent key geometry interventions."""
import argparse
import csv
import importlib.util
import itertools
import json
import unittest

import torch
from torch.nn import functional as F
import query_write_order as previous

prior = previous.prior
study = prior.study
ROOT = study.ROOT
OUT = ROOT / 'results/key_geometry_retention'
VARIANTS = ('native', 'padded', 'orthogonal')


def dependencies():
    paths = ('experiments/key_geometry_retention.py', 'tests/test_key_geometry_retention.py',
             'docs/protocols/key-geometry-retention.md', *previous.dependencies())
    return {p: study.parent.digest(ROOT / p) for p in paths}


def key_table(model, variant):
    assert variant in VARIANTS, 'Undeclared geometry'
    keys = model.keys(torch.arange(32))
    if variant == 'padded':
        keys = F.pad(keys, (0, 16))
    elif variant == 'orthogonal':
        keys = torch.eye(32, dtype=keys.dtype)
    return keys


def query_rows(rows):
    expanded = []
    for row in rows:
        previous.validate(row)
        mapping = dict(row['events'])
        assert set(mapping) == set(range(32)), 'Expected all 32 known keys'
        expanded.extend(dict(row, query=q, label=mapping[q]) for q in range(32))
    return expanded


def evaluate(model, rows, variant):
    assert model.arm == 'delta'
    query_rows(rows)
    before = study.parent.state_hash(model.state_dict())
    e = torch.tensor([r['events'] for r in rows])
    with torch.no_grad():
        keys = key_table(model, variant); k = keys[e[..., 0]]; v = model.value(e[..., 1])
        # Parent prefill fixes width at 16. This diagnostic alone allocates the declared width.
        state = v.new_zeros(len(rows), keys.shape[1], v.shape[-1])
        beta = model.write_logit.sigmoid()
        for t in range(e.shape[1]):
            old = (k[:, t, :, None] * state).sum(1)
            state = state + beta * k[:, t, :, None] * (v[:, t] - old)[:, None, :]
        # One state per history, shared by all queries. No query participates in prefill.
        reads = keys @ state
        scores = model.output(reads)
        direct_error = float((reads - beta * model.value(torch.tensor([[dict(r['events'])[q] for q in range(32)] for r in rows]))).abs().max())
    assert before == study.parent.state_hash(model.state_dict())
    return scores, dict(state_shape=list(state.shape[1:]), state_bytes_per_history=state[0].numel()*state.element_size(),
                        direct_read_error=direct_error)


def reference(model, rows, variant, return_reads=False):
    e = torch.tensor([r['events'] for r in rows])
    raw = model.key.weight
    keys = raw / raw.square().sum(-1, keepdim=True).clamp_min(1e-24).sqrt()
    if variant == 'padded': keys = torch.cat((keys, torch.zeros_like(keys)), dim=1)
    elif variant == 'orthogonal': keys = torch.diag(torch.ones(32, dtype=raw.dtype))
    else: assert variant == 'native'
    with torch.no_grad():
        state = raw.new_zeros(len(rows), keys.shape[1], model.value.weight.shape[1]); eye = torch.eye(keys.shape[1], dtype=raw.dtype)
        beta = torch.sigmoid(model.write_logit)
        for t in range(32):
            kt = keys[e[:, t, 0]]; vt = model.value.weight[e[:, t, 1]]
            state = (eye - beta * kt[:, :, None] * kt[:, None, :]) @ state + beta * kt[:, :, None] * vt[:, None, :]
        reads = torch.einsum('qd,bdv->bqv', keys, state)
        scores = reads @ model.output.weight.T + model.output.bias
        return (scores, reads) if return_reads else scores


def summarize(scores, rows, independent=False):
    expanded = query_rows(rows)
    reducer = prior.reduce_scores if independent else study.parent.metrics
    metrics = reducer(scores.reshape(-1, 6), expanded)['all']
    if independent:
        correct_contexts = 0
        for logits, row in zip(scores.tolist(), rows):
            mapping = dict(row['events'])
            correct_contexts += all(max(range(6), key=lambda i: logits[q][i]) == mapping[q] for q in range(32))
    else:
        labels = torch.tensor([r['label'] for r in expanded]).reshape(len(rows), 32)
        correct_contexts = int((scores.argmax(-1) == labels).all(1).sum())
    return dict(all_queries=metrics, complete_maps=dict(count=len(rows), correct=correct_contexts, accuracy=correct_contexts/len(rows)))


def audit_cell(checkpoint, rows, variant, scores, metrics):
    model = prior.load(checkpoint, 'delta'); ref = reference(model, rows, variant)
    error = prior.check_scores(scores.reshape(-1,6), ref.reshape(-1,6))
    independent = summarize(ref, rows, independent=True)
    prior.check_metrics({'all':metrics['all_queries']}, {'all':independent['all_queries']})
    assert metrics['complete_maps'] == independent['complete_maps'], 'Wrong complete-map metric'
    return error, ref


def check_coverage(cells, seeds):
    keys = [(c['seed'], c['variant']) for c in cells]
    expected = set(itertools.product(seeds, VARIANTS))
    assert len(keys) == len(expected) and set(keys) == expected, 'Missing or duplicate geometry cells'


def run():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    _, data, chosen = prior.parent_evidence()
    old = json.loads((previous.OUT/'report.json').read_text()); audit = json.loads((previous.OUT/'independent_audit.json').read_text())
    assert audit['passed'] and audit['report_sha256'] == study.parent.digest(previous.OUT/'report.json')
    assert old['source_dependencies'] == previous.dependencies()
    chosen = [c for c in chosen if c['arm']=='delta']; rows = data['pools']['test_32']
    assert len(chosen)==3 and len(rows)==384 and not OUT.exists(), 'Preserve existing evidence'
    source = dependencies(); OUT.mkdir()
    protected = json.loads((previous.OUT/'preserved.json').read_text())
    for p in previous.OUT.rglob('*'):
        if p.is_file(): protected[p.relative_to(ROOT).as_posix()] = study.parent.digest(p)
    for p in ('docs/topics/attention/query-write-order-results.md','docs/topics/attention/query-write-order-guide.md','docs/assessments/association-retention-assessment.md'):
        protected[p] = study.parent.digest(ROOT/p)
    assert all(study.parent.digest(ROOT/p)==h for p,h in protected.items())
    study.parent.write(OUT/'preserved.json', protected)
    spec=importlib.util.spec_from_file_location('test_key_geometry_retention',ROOT/'tests/test_key_geometry_retention.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    study.parent.write(OUT/'preflight.json',dict(passed=result.wasSuccessful(),methods=result.testsRun,source_dependencies=source))
    assert result.wasSuccessful(), 'Preflight failed; no saved-model evaluation'
    study.parent.write(OUT/'lock.json',dict(source_dependencies=source,seeds=study.CONFIRM_SEEDS,variants=VARIANTS,
        expected_cells=3*3,expected_predictions=3*3*384*32,training_updates=0,
        data_sha256=study.parent.digest(study.OUT/'confirmation_data.json'),
        parent_report_sha256=study.parent.digest(study.OUT/'report.json')))
    cells,saved=[],{}
    for item in chosen:
        cp=ROOT/item['checkpoint'];assert study.parent.digest(cp)==item['checkpoint_sha256']
        model=prior.load(cp,'delta');native=None
        for variant in VARIANTS:
            scores,state=evaluate(model,rows,variant);metrics=summarize(scores,rows)
            error,_=audit_cell(cp,rows,variant,scores,metrics)
            if variant=='native':
                native=scores
                e,q,_=study.parent.tensors(rows)
                with torch.no_grad(): original=model(e,q)
                prior.check_scores(scores[torch.arange(len(rows)),q],original)
                prior.check_metrics({'all':study.parent.metrics(original,rows)['all']},{'all':item['metrics']['test_32']['all']})
            change=float((scores-native).abs().max())
            if variant=='padded':assert change<1e-9, 'Padding changed results'
            if variant=='orthogonal':assert state['direct_read_error']<1e-9, 'Orthogonal direct-read identity failed'
            saved[f"{item['seed']}_{variant}"]=scores
            cells.append(dict(seed=item['seed'],variant=variant,checkpoint=item['checkpoint'],checkpoint_sha256=item['checkpoint_sha256'],
                metrics=metrics,state=state,beta=float(model.write_logit.sigmoid().detach()),max_logit_error=error,native_max_logit_change=change,
                vs_native=prior.paired_counts(native.reshape(-1,6),scores.reshape(-1,6),query_rows(rows))))
        print(f"Evaluated all-query geometries for seed {item['seed']}",flush=True)
    check_coverage(cells,study.CONFIRM_SEEDS)
    assert dependencies()==source and all(study.parent.digest(ROOT/p)==h for p,h in protected.items())
    torch.save(saved,OUT/'scores.pt')
    study.parent.write(OUT/'report.json',dict(status='complete',training_updates=0,device='cpu',dtype='float64',threads=1,cells=cells,
        predictions=110592,source_dependencies=source,scores_sha256=study.parent.digest(OUT/'scores.pt'),protected_files_unchanged=len(protected),
        criteria=dict(padding_invariant=all(c['native_max_logit_change']<1e-9 for c in cells if c['variant']=='padded'),
            orthogonal_all_maps_correct=all(c['metrics']['complete_maps']['accuracy']==1 for c in cells if c['variant']=='orthogonal')),
        scope='Fixed known-symbol geometry intervention, all 32 queries share one final state, no reordering or training. Orthogonal state doubles width.'))


def audit():
    torch.set_num_threads(1)
    _,data,chosen=prior.parent_evidence();rows=data['pools']['test_32']
    r=json.loads((OUT/'report.json').read_text());lock=json.loads((OUT/'lock.json').read_text())
    assert r['source_dependencies']==lock['source_dependencies']==dependencies()
    assert lock['data_sha256']==study.parent.digest(study.OUT/'confirmation_data.json') and lock['parent_report_sha256']==study.parent.digest(study.OUT/'report.json')
    assert r['status']=='complete' and r['scores_sha256']==study.parent.digest(OUT/'scores.pt') and r['training_updates']==0
    scores=torch.load(OUT/'scores.pt',weights_only=True);check_coverage(r['cells'],study.CONFIRM_SEEDS)
    assert len(scores)==len(r['cells'])==9 and r['predictions']==110592
    records,errors=[],[]
    chosen={c['seed']:c for c in chosen if c['arm']=='delta'}
    for c in r['cells']:
        item=chosen[c['seed']];assert c['checkpoint']==item['checkpoint'] and c['checkpoint_sha256']==study.parent.digest(ROOT/c['checkpoint'])
        actual=scores[f"{c['seed']}_{c['variant']}"];assert actual.shape==(384,32,6)
        error,ref=audit_cell(ROOT/c['checkpoint'],rows,c['variant'],actual,c['metrics']);assert error==c['max_logit_error'];errors.append(error)
        native=scores[f"{c['seed']}_native"]
        assert previous.independent_pairs(native.reshape(-1,6),ref.reshape(-1,6),query_rows(rows))==c['vs_native']
        assert c['native_max_logit_change']==float((actual-native).abs().max())
        model=prior.load(ROOT/c['checkpoint'],'delta');assert c['beta']==float(model.write_logit.sigmoid().detach())
        if c['variant']=='native':
            e,q,_=study.parent.tensors(rows)
            with torch.no_grad(): expected=study.parent.reference(model,e,q)
            prior.check_scores(ref[torch.arange(len(rows)),q],expected)
            prior.check_metrics({'all':prior.reduce_scores(expected,rows)['all']},{'all':item['metrics']['test_32']['all']})
        width=16 if c['variant']=='native' else 32
        assert c['state']['state_shape']==[width,8] and c['state']['state_bytes_per_history']==width*8*8
        with torch.no_grad():
            labels=torch.tensor([[dict(row['events'])[q] for q in range(32)] for row in rows])
            direct_reads=model.write_logit.sigmoid()*model.value(labels)
            direct=model.output(direct_reads)
            _,ref_reads=reference(model,rows,c['variant'],return_reads=True)
            read_error=float((ref_reads-direct_reads).abs().max())
        assert abs(read_error-c['state']['direct_read_error'])<1e-9
        if c['variant']=='orthogonal':
            assert read_error<1e-9
            prior.check_scores(ref.reshape(-1,6),direct.reshape(-1,6))
        if c['variant']=='padded':assert c['native_max_logit_change']<1e-9
        records.append(dict(seed=c['seed'],variant=c['variant'],**c['metrics']['all_queries'],complete_maps=c['metrics']['complete_maps']['correct'],
            complete_map_accuracy=c['metrics']['complete_maps']['accuracy'],state_bytes=c['state']['state_bytes_per_history'],rescued=c['vs_native']['rescued'],harmed=c['vs_native']['harmed']))
    assert r['criteria']==dict(padding_invariant=True,orthogonal_all_maps_correct=all(c['metrics']['complete_maps']['correct']==384 for c in r['cells'] if c['variant']=='orthogonal'))
    protected=json.loads((OUT/'preserved.json').read_text());assert all(study.parent.digest(ROOT/p)==h for p,h in protected.items())
    assert not (OUT/'independent_audit.json').exists()
    with (OUT/'metrics.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    study.parent.write(OUT/'independent_audit.json',dict(passed=True,report_sha256=study.parent.digest(OUT/'report.json'),source_dependencies=dependencies(),
        cells=9,predictions=110592,max_logit_error=max(errors),argmax_disagreements=0,protected_files_unchanged=len(protected),
        scope='Independent matrix transitions, Python loss/complete-map/paired reductions, parent baseline and direct orthogonal identity. Shared data/weights/PyTorch.'))
    print('Key-geometry retention audit passes.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('run','audit'));args=p.parse_args()
    run() if args.phase=='run' else audit()
