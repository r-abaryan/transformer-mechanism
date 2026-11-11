"""Single checkpoint intervention: remove this softmax reader's recency term."""
import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import torch
import delta_retrieval as parent
from audit_delta_retrieval import reduce_scores, check_metrics

OUT=parent.ROOT/'results/recency_ablation'
SEEDS=parent.CONFIRM_SEEDS
POOLS=('train_4','train_8','validation_4','validation_8','test_4','test_8','test_16','test_32')
MODES=('native','no_recency')


def without_recency(model):
    if model.arm!='softmax':
        raise ValueError('This diagnostic applies only to the softmax reader.')
    changed=copy.deepcopy(model)
    with torch.no_grad():changed.log_recency.fill_(-torch.inf)
    # exp(-infinity)=0 removes only the bias coefficient. Never train this copy.
    return changed


def check_scores(actual,reference):
    error=(actual-reference).abs().max().item()
    changes=int((actual.argmax(-1)!=reference.argmax(-1)).sum())
    if error>=1e-9 or changes:
        raise AssertionError(f'Intervention equations disagree: max error {error}, argmax changes {changes}')
    return error


def evaluate(model,rows,mode):
    original_hash=parent.state_hash(model.state_dict())
    changed=model if mode=='native' else without_recency(model)
    e,q,_=parent.tensors(rows)
    with torch.no_grad():
        actual=torch.cat([changed(a,b) for a,b in zip(e.split(128),q.split(128))])
        independent=torch.cat([parent.reference(model,a,b,wrong=mode=='no_recency') for a,b in zip(e.split(128),q.split(128))])
    error=check_scores(actual,independent)
    metrics=parent.metrics(actual,rows)
    check_metrics(metrics,reduce_scores(independent,rows))
    assert original_hash==parent.state_hash(model.state_dict())
    return metrics,actual,error


def paired_counts(native,modified,rows):
    labels=torch.tensor([r['label'] for r in rows]);a=native.argmax(-1)==labels;b=modified.argmax(-1)==labels
    result={}
    for group in parent.GROUPS:
        m=torch.tensor([r['group']==group for r in rows])
        result[group]=dict(count=int(m.sum()),rescued=int((~a&b&m).sum()),harmed=int((a&~b&m).sum()),
                           both_correct=int((a&b&m).sum()),both_wrong=int((~a&~b&m).sum()))
    return result


def decision(cells):
    changed=[c for c in cells if c['mode']=='no_recency']
    expected={(s,p) for s in SEEDS for p in POOLS}
    if {(c['seed'],c['pool']) for c in changed}!=expected or len(changed)!=len(expected):
        raise AssertionError('Missing or duplicate diagnostic cells')
    long=[c for c in changed if c['pool'] in ('test_16','test_32')]
    short=[c for c in changed if c['pool'] not in ('test_16','test_32')]
    costs=[c for c in cells if c['mode']=='native']
    lookup={(c['seed'],c['pool']):c for c in costs}
    return dict(long_unique_recovery=all(c['metrics']['unique']['accuracy']>=.95 for c in long),
                short_unique_preservation=all(c['metrics']['unique']['accuracy']>=.95 for c in short),
                repeat_cost_detected=any(c['metrics']['repeat']['accuracy']<lookup[c['seed'],c['pool']]['metrics']['repeat']['accuracy']-.02 for c in changed))


def dependencies():
    paths=('experiments/recency_ablation.py','tests/test_recency_ablation.py','docs/protocols/recency-ablation.md',
           'experiments/delta_retrieval.py','experiments/audit_delta_retrieval.py')
    return {p:parent.digest(parent.ROOT/p) for p in paths}


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    report_path=parent.OUT/'report.json';old=json.loads(report_path.read_text())
    assert old['status']=='complete' and old['source_dependencies']==parent.dependencies()
    old_audit=json.loads((parent.OUT/'independent_audit.json').read_text())
    assert old_audit['passed'] and old_audit['report_sha256']==parent.digest(report_path)
    assert old_audit['audit_source_sha256']==parent.digest(parent.ROOT/'experiments/audit_delta_retrieval.py')
    data_path=parent.OUT/'confirmation_data.json';data=json.loads(data_path.read_text())
    assert old['data_sha256']['confirmation']==parent.digest(data_path)
    assert set(data['pools'])==set(POOLS)
    parent.audit_data(data)
    selected=[r for r in old['runs'] if r['phase']=='confirmation' and r['arm']=='softmax']
    assert {r['seed'] for r in selected}==set(SEEDS) and len(selected)==len(SEEDS)
    prior={str(p.relative_to(parent.ROOT)).replace('\\','/'):parent.digest(p) for p in parent.OUT.iterdir() if p.is_file()}
    for p,h in json.loads((parent.OUT/'prior_hashes.json').read_text()).items():
        assert parent.digest(parent.ROOT/p)==h;prior[p]=h
    prior.update(old['source_dependencies'])
    prior['experiments/audit_delta_retrieval.py']=old_audit['audit_source_sha256']
    source=dependencies()
    OUT.mkdir(exist_ok=False)
    parent.write(OUT/'preserved.json',prior)
    # Exercise the actual evaluation/reducer entry with initialized saved weights.
    spec=importlib.util.spec_from_file_location('test_recency_ablation',parent.ROOT/'tests/test_recency_ablation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tested=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if not tested.wasSuccessful():
        parent.write(OUT/'preflight.json',dict(passed=False,methods=tested.testsRun,source_dependencies=source))
        raise RuntimeError('Preflight failure; no saved-model evaluation authorized after failure.')
    parent.write(OUT/'preflight.json',dict(passed=True,methods=tested.testsRun,source_dependencies=source))
    cells=[];pairs=[];scores={};max_error=0.
    for run in selected:
        cp=parent.ROOT/run['checkpoint'];assert parent.digest(cp)==run['checkpoint_sha256']
        saved=torch.load(cp,weights_only=True);assert saved['step']==3000
        model=parent.Reader('softmax');model.load_state_dict(saved['model'])
        for pool,rows in data['pools'].items():
            results={}
            for mode in MODES:
                metrics,logits,error=evaluate(model,rows,mode);max_error=max(max_error,error)
                if mode=='native':check_metrics(run['metrics'][pool],metrics)
                key=f'{run["seed"]}_{pool}_{mode}';scores[key]=logits
                results[mode]=logits
                cells.append(dict(seed=run['seed'],pool=pool,mode=mode,metrics=metrics,max_logit_error=error,
                                  checkpoint=run['checkpoint'],checkpoint_sha256=run['checkpoint_sha256']))
            pairs.append(dict(seed=run['seed'],pool=pool,groups=paired_counts(results['native'],results['no_recency'],rows)))
    expected=len(SEEDS)*len(POOLS)*len(MODES);assert len(cells)==expected==48
    expected_predictions=len(SEEDS)*len(MODES)*sum(len(v) for v in data['pools'].values())
    assert expected_predictions==20736
    assert all(parent.digest(parent.ROOT/p)==h for p,h in prior.items())
    assert dependencies()==source
    sp=OUT/'scores.pt';torch.save(scores,sp)
    parent.write(OUT/'report.json',dict(status='complete',training_updates=0,device='cpu',dtype='float64',threads=1,
        source_dependencies=source,parent_report_sha256=parent.digest(report_path),parent_data_sha256=parent.digest(data_path),
        preflight_sha256=parent.digest(OUT/'preflight.json'),cells=cells,paired_outcomes=pairs,expected_cells=expected,
        predictions=expected_predictions,max_logit_error=max_error,argmax_audit_changes=0,
        native_metrics_match_archive=True,prior_files_unchanged=len(prior),scores_sha256=parent.digest(sp),
        decision=decision(cells),scope='Posthoc checkpoint intervention on opened pools; neither fresh confirmation nor a trained repair.'))
    print(json.dumps(dict(cells=len(cells),predictions=expected_predictions,max_logit_error=max_error,
                          prior_files_unchanged=len(prior),decision=decision(cells))),flush=True)


if __name__=='__main__':main()
