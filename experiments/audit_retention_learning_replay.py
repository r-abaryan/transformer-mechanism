"""Independent scalar Adam reconstruction and saved-state objective checks."""
import json
import math

import torch
import retention_learning_replay as study
import audit_objective_beta_pressure as gradient_audit


def check_steps(records, initial_alpha):
    alpha, first, second = initial_alpha, 0., 0.
    errors, mismatches = [], []
    for t, row in enumerate(records, 1):
        assert row['step'] == t and row['load'] == (2,8,16,32)[(t-1)%4]
        g = row['gradient']
        first = .9*first+.1*g
        second = .999*second+.001*g*g
        update = -.003*(first/(1-.9**t))/(math.sqrt(second/(1-.999**t))+1e-8)
        for actual, expected in ((row['alpha_before'],alpha), (row['alpha_after'],alpha+update),
                                 (row['exp_avg'],first),(row['exp_avg_sq'],second),
                                 (row['beta_before'],1/(1+math.exp(-alpha))),
                                 (row['beta_after'],1/(1+math.exp(-(alpha+update))))):
            error = abs(actual-expected)
            assert error <= 1e-11, 'Independent Adam recurrence disagreement'
            errors.append(error)
        # Use measured preceding alpha, rather than accumulating audit roundoff.
        alpha = row['alpha_after']
        delta = row['alpha_after']-row['alpha_before']
        if abs(g)>1e-7 and abs(delta)>1e-12 and g*delta>0:
            mismatches.append(t)
    return dict(max_adam_error=max(errors), resolved_against_current_gradient=len(mismatches),
                mismatch_steps=mismatches)


def audit_trajectory(folder, seed, arm, payload, probes, steps):
    trajectory = json.loads((folder/f'trajectory_{seed}_{arm}.json').read_text())
    assert len(trajectory['steps']) == steps
    assert [p['step'] for p in trajectory['snapshots']] == list(probes)
    initial = torch.load(study.ROOT/trajectory['snapshots'][0]['checkpoint'],weights_only=True)
    assert initial['step'] == 0 and not initial['optimizer']['state']
    checks = check_steps(trajectory['steps'],float(initial['model']['write_logit']))
    records = []
    for record in trajectory['snapshots']:
        cp = study.ROOT/record['checkpoint']
        assert study.digest(cp) == record['sha256']
        saved = torch.load(cp, weights_only=True)
        assert saved['step'] == record['step'] and len(saved['cells']) == 8
        assert {(c['load'],c['objective']) for c in saved['cells']} == {
            (m,o) for m in (2,8,16,32) for o in ('assigned','all_queries')}
        for cell in saved['cells']:
            checked = gradient_audit.audit_checkpoint(cp, payload['pools'][f"train_{cell['load']}"],
                                                     cell['objective'], cell)
            records.append(dict(step=record['step'],load=cell['load'],objective=cell['objective'],**checked))
        assert study.summarize(saved['cells']) == {k:record[k] for k in ('objectives','comparison')}
        beta = float(saved['model']['write_logit'].sigmoid())
        assert record['beta'] == beta
        if record['step']:
            # Adam assigns parameter IDs in model registration order; alpha is first.
            model = study.pressure.model_source.LoadReader('delta')
            assert next(iter(dict(model.named_parameters()))) == 'write_logit'
            state = saved['optimizer']['state'][saved['optimizer']['param_groups'][0]['params'][0]]
            row = trajectory['steps'][record['step']-1]
            assert float(saved['model']['write_logit']) == row['alpha_after']
            assert float(state['exp_avg']) == row['exp_avg']
            assert float(state['exp_avg_sq']) == row['exp_avg_sq']
            assert all(int(s['step']) == record['step'] for s in saved['optimizer']['state'].values())
    return dict(seed=seed,arm=arm,**checks,cells=len(records),
                predictions=sum(c['predictions'] for c in records),
                max_logit_error=max(c['max_logit_error'] for c in records),
                max_gradient_error=max(c['max_parameter_gradient_error'] for c in records),
                max_fd_error=max(c['max_fd_error'] for c in records),checks=records)


def main():
    torch.set_num_threads(1)
    folder = study.OUT
    report = json.loads((folder/'report.json').read_text())
    lock = json.loads((folder/'lock.json').read_text())
    assert report['status'] == 'complete' and report['lock_sha256'] == study.digest(folder/'lock.json')
    study.check_files(lock['files'])
    payload = json.loads((study.parent.OUT/'development_data.json').read_text())
    original = json.loads((study.parent.OUT/'report.json').read_text())
    assert [(r['seed'],r['arm']) for r in report['runs']] == [(r['seed'],r['arm']) for r in original['runs']]
    assert study.digest(folder/'final_scores.pt') == report['final_scores_sha256']
    scores = torch.load(folder/'final_scores.pt',weights_only=True)
    checks, final_errors = [], []
    for item, historical in zip(report['runs'], original['runs']):
        seed,arm = item['seed'],item['arm']
        checked = audit_trajectory(folder,seed,arm,payload,lock['probes'],3000)
        checks.append(checked)
        new = torch.load(study.ROOT/item['checkpoint'],weights_only=True)
        old = torch.load(study.ROOT/historical['checkpoint'],weights_only=True)
        assert study.equal(new,old), 'Audit endpoint mismatch'
        for field in ('initial_sha256','history_stream_sha256','potential_query_draws_sha256',
                      'actual_queries_labels_sha256','exposure','curve','metrics','beta'):
            assert study.equal(item[field],historical[field])
        for pool,rows in payload['pools'].items():
            for all_queries in (False,True):
                name = ('all_' if all_queries else '')+pool
                key = f'development_{seed}_{arm}_{name}'
                final_errors.append(study.parent.audit_cell(study.ROOT/item['checkpoint'],rows,
                                                            all_queries,item['metrics'][name],scores[key]))
        print(json.dumps(dict(seed=seed,arm=arm,cells_audited=checked['cells'])),flush=True)
    assert len(scores) == 64 and len(final_errors) == 64
    assert sum(c['cells'] for c in checks) == 224
    assert sum(c['predictions'] for c in checks) == 1333248
    study.check_files(lock['files'])
    result = dict(passed=True,report_sha256=study.digest(folder/'report.json'),runs=checks,
                  steps_checked=12000,cells=224,predictions=1333248,final_cells=64,
                  max_final_logit_error=max(final_errors),protected_files=len(lock['files']))
    study.write(folder/'audit.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='runs'}),flush=True)


if __name__ == '__main__':
    main()
