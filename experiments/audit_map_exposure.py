"""Independent labels, data scope, every Adam gradient/update and evaluations."""
import json

import torch
import map_exposure as study
import audit_full_pool_fitting as engine_audit


def canonical_map(row):
    memory = {}
    for key, value in row['events']:
        assert key not in memory or memory[key] == value
        memory[key] = value
    return min(tuple((k, (memory[k]+shift) % 6) for k in sorted(memory)) for shift in range(6))


def audit_data(payload):
    original = payload['original']; a = payload['active']['reuse']; b = payload['active']['half_new']
    for pool, rows in original.items(): engine_audit.reference.independent.check_rows(rows)
    for policy in study.POLICIES:
        for rows in payload['active'][policy].values(): engine_audit.reference.independent.check_rows(rows)
    selected = set(payload['selected_blocks']); assert len(selected)*2 == len(original['train_32'])//6
    changed = []
    for m in (2, 8, 16): assert a[f'train_{m}'] == b[f'train_{m}'] == original[f'train_{m}'], 'Lower-load data changed'
    assert a['train_32'] == original['train_32']
    old_maps = {canonical_map(row) for rows in original.values() for row in rows if row['load'] == 32}
    added_maps = set()
    for block in range(len(a['train_32'])//6):
        for shift in range(6):
            x, y = [p['train_32'][block*6+shift] for p in (a,b)]
            assert x['query'] == y['query'] and [k for k,v in x['events']] == [k for k,v in y['events']], 'Changed key/query order'
            if block not in selected: assert x == y, 'Changed unselected block'
            else:
                assert x['label'] != y['label'], 'Missing disagreeing assigned-query control'
                assert canonical_map(y) not in old_maps, 'Old mapping reused as new'
                added_maps.add(canonical_map(y))
                base = b['train_32'][block*6]
                assert y['events'] == [[k,(v+shift)%6] for k,v in base['events']], 'Broken cyclic block'
                assert y['label'] == (base['label']+shift)%6
                changed.append(block*6+shift)
    assert len(added_maps) == len(selected), 'Duplicate added mapping'
    overlaps = {}
    for split in ('validation','test'):
        for m in study.prior.LOADS:
            pool = f'{split}_{m}'
            assert a[pool] == b[pool] == payload['evaluation'][pool]
            engine_audit.reference.independent.check_rows(a[pool])
            trainmaps = {canonical_map(r) for r in original[f'train_{m}']}
            overlaps[pool] = sum(canonical_map(r) in trainmaps for r in a[pool])
            if m == 32: assert all(canonical_map(r) not in old_maps | added_maps for r in a[pool]), 'High-load evaluation map leakage'
    assert len(changed)*2 == len(a['train_32'])
    for p, rows in original.items():
        target = 'seen_'+p if p.startswith('train_') else 'reference_'+p
        assert payload['evaluation'][target] == rows, 'Wrong retention/reference pool'
    return dict(changed_histories=len(changed), added_map_families=len(added_maps),
                heldout_semantic_map_overlap_with_original_training=overlaps)


def audit(out, persist=True):
    lock = json.loads((out/'lock.json').read_text()); report = json.loads((out/'report.json').read_text())
    study.replay.check_files(lock['files']); assert report['status'] == 'complete' and lock['torch'] == torch.__version__
    for name, expected in [('lock.json', report['lock_sha256']), ('data.pt', lock['data_sha256']), ('scores.pt', report['scores_sha256'])]:
        assert study.replay.digest(out/name) == expected, 'Artifact binding'
    payload = torch.load(out/'data.pt', weights_only=True); data_check = audit_data(payload)
    if not lock['fixture']:
        cases, original = study.inputs()
        assert lock['cases'] == cases and payload == study.build_data(original, study.DATA_SEED)
        assert (lock['ratio'],lock['interval']) == (48,32)
    scores = torch.load(out/'scores.pt', weights_only=True)
    expected = set(); checks = []; native_controls = []
    for case in lock['cases']:
        pair = [r for r in report['runs'] if r['seed'] == case['seed']]
        assert [r['policy'] for r in pair] == list(study.POLICIES), 'Missing or duplicate run'
        for run in pair:
            policy = run['policy']; pool = payload['active'][policy]
            assert run['trace_sha256'] == study.replay.digest(study.ROOT/run['trace'])
            check = engine_audit.check_adam(case, run, pool, lock, case['closures'])
            checks.append(dict(seed=case['seed'],policy=policy,**check))
            for snap in run['snapshots']:
                assert snap['sha256'] == study.replay.digest(study.ROOT/snap['checkpoint'])
                assert snap['fit'] == study.prior.fit(snap['training'])
                for m in study.prior.LOADS: expected.add((case['seed'],policy,snap['step'],f'train_{m}'))
            for step in (0,run['optimizer_calls']):
                for p in payload['evaluation']: expected.add((case['seed'],policy,step,p))
            if policy == 'reuse' and 'historical' in case:
                previous = case['historical']
                actual = torch.load(study.ROOT/run['final_checkpoint'], weights_only=True)
                old = torch.load(study.ROOT/previous['final_checkpoint'], weights_only=True)
                assert study.replay.equal(actual['model'],old['model']) and study.replay.equal(actual['optimizer'],old['optimizer']), 'Historical endpoint control'
                new_trace = torch.load(study.ROOT/run['trace'], weights_only=True)
                old_trace = torch.load(study.ROOT/previous['trace'], weights_only=True)
                assert study.replay.equal(new_trace,old_trace), 'Historical trace control'
                native_controls.append(dict(seed=case['seed'],historical_exact=True))
        for field in ('optimizer_calls','histories','input_events','answer_targets'):
            assert pair[0][field] == pair[1][field], 'Unmatched exposure'
        x,y=[torch.load(study.ROOT/r['trace'],weights_only=True) for r in pair]
        assert len(x)==len(y)
        for a,b in zip(x,y):
            assert a['load']==b['load'] and torch.equal(a['indices'],b['indices']) and torch.equal(a['draws'],b['draws']), 'Unmatched sampling'
    ids=[(c['seed'],c['policy'],c['step'],c['pool']) for c in scores]
    assert len(ids)==len(expected) and set(ids)==expected, 'Missing score cell'
    assert len(report['cells'])==len(scores)
    maximum=0.;targets=0
    for cell, record in zip(scores,report['cells']):
        engine_audit.compare(record,{k:v for k,v in cell.items() if k not in ('logits','labels','repeat')})
        rows=payload['active'][cell['policy']][cell['pool']] if cell['pool'].startswith('train_') else payload['evaluation'][cell['pool']]
        with torch.random.fork_rng(devices=[]):checked=engine_audit.reference.audit_cell(study.ROOT/cell['checkpoint'],rows,cell)
        maximum=max(maximum,checked['max_logit_error']);targets+=checked['targets']
    engine_audit.compare(report['decision'],study.decision(scores))
    # Recount final fitting/gain criteria from predictions rather than report accuracy.
    for case in lock['cases']:
        seed=case['seed'];end=case['closures']*lock['ratio'];p=next(p for p in report['decision']['pairs'] if p['seed']==seed)
        def score(policy,pool):return next(c for c in scores if (c['seed'],c['policy'],c['step'],c['pool'])==(seed,policy,end,pool))
        for policy in study.POLICIES:
            for prefix,field in [('train_','active_training_fit'),('seen_train_','original_training_retained')]:
                fit=all(int((score(policy,prefix+str(m))['logits'].argmax(-1)==score(policy,prefix+str(m))['labels']).sum())/score(policy,prefix+str(m))['labels'].numel()>=.99 for m in study.prior.LOADS)
                assert p[field][policy]==fit, 'Training fit decision'
        x,y=[score(policy,'test_32') for policy in study.POLICIES]
        gain=float((y['logits'].argmax(-1)==y['labels']).double().mean()-(x['logits'].argmax(-1)==x['labels']).double().mean())
        assert abs(gain-p['test32_accuracy_gain'])<1e-12
        benefit=gain>=.005 and all(p['active_training_fit'].values())
        assert p['exposure_benefit']==benefit and p['repair']==(benefit and p['original_training_retained']['half_new'])
    for name in ('optimizer_calls','histories','input_events','answer_targets'):
        assert report[name]==sum(r[name] for r in report['runs']), 'Exposure count'
    assert report['gradient_evaluations']==report['optimizer_calls'] and report['evaluation_targets']==targets
    study.replay.check_files(lock['files'])
    result=dict(passed=True,cells=len(scores),evaluation_targets=targets,optimizer_calls=report['optimizer_calls'],
                max_gradient_error=max(c['max_gradient_error'] for c in checks),max_adam_error=max(c['max_update_error'] for c in checks),
                max_logit_error=maximum,protected_files=len(lock['files']),native_controls=native_controls,data=data_check,
                report_sha256=study.replay.digest(out/'report.json'))
    if persist:
        assert not (out/'audit.json').exists();study.replay.write(out/'audit.json',result)
    return result


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    print(json.dumps(audit(study.OUT)),flush=True)
