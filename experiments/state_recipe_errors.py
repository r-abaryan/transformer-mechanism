"""Checkpoint-only paired state error diagnostic, training inputs only."""
import csv
import json
from pathlib import Path

import torch

from state_tracking import ROOT, model_new, forward, tensors, save_json
from audit_state_tracking import independent, oracle
from state_sensitivity import variants, summarize, scalar_summary
from position_distractors import digest


def features(ids):
    ops = ids[1:-1]
    reset = max((i for i, op in enumerate(ops) if op >= 2), default=-1)
    suffix = ops[reset+1:]
    active = [i for i in range(reset+1, len(ops)) if ops[i] != 0]
    return dict(suffix_flips=suffix.count(1), last_set=reset,
                last_active_distance=len(ops)-1-active[-1] if active else len(ops))


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    target = ROOT/'results/state_recipe_errors'
    assert not target.exists()
    protocol = ROOT/'docs/protocols/state-recipe-errors.md'
    previous = ROOT/'results/state_recipe_development'
    report = json.loads((previous/'report.json').read_text())
    pools = json.loads((previous/'training_data.json').read_text())
    runs = [r for r in report['runs'] if r['recipe'] in ('constant_001', 'cosine_001')]
    assert len(runs) == 4 and all(k.startswith('train/') for k in pools)
    paths = [protocol, Path(__file__), previous/'report.json', previous/'training_data.json']
    paths += [ROOT/p for p in report['source_dependencies']]
    paths += [ROOT/r[k] for r in runs for k in ('checkpoint', 'scores')]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    assert hashes['results/state_recipe_development/training_data.json'] == report['training_snapshot_sha256']
    for r in runs:
        assert digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
        assert digest(ROOT/r['scores']) == r['scores_sha256']
    prepared = {}
    wrong_labels = 0
    for key, rows in pools.items():
        changed = []; metadata = []
        for i, (ids, y) in enumerate(rows):
            assert oracle(ids) == y
            for v in variants(ids):
                changed.append([v['ids'], v['label']])
                metadata.append(dict(original_index=i, name=v['name'], relevant=v['relevant']))
                if v['name'] != 'two_suffix':
                    wrong_labels += (1-y) != v['label']
        prepared[key] = (changed, metadata)
    assert wrong_labels > 0
    scores = {}; metrics = []; strata = []; paired = []
    max_logit = max_scalar = max_archive = 0.; evaluations = 0
    for r in runs:
        m = model_new().eval()
        m.load_state_dict(torch.load(ROOT/r['checkpoint'], weights_only=True)['model'])
        archive = torch.load(ROOT/r['scores'], weights_only=True)
        run_scores = {}; scores[f"{r['seed']}/{r['recipe']}"] = run_scores
        for key, rows in pools.items():
            x, y = tensors(rows); nx, ny = tensors(prepared[key][0])
            def compute(inputs, fn):
                with torch.no_grad():
                    return torch.cat([fn(m, b)[:, -1] for b in inputs.split(128)])
            a, ra = compute(x, forward), compute(x, independent)
            b, rb = compute(nx, forward), compute(nx, independent)
            error = max(float((a-ra).abs().max()), float((b-rb).abs().max()))
            ae = float((a-archive[key]).abs().max())
            assert max(error, ae) <= 1e-10
            assert torch.equal(a.argmax(-1), ra.argmax(-1)) and torch.equal(b.argmax(-1), rb.argmax(-1))
            max_logit = max(max_logit, error); max_archive = max(max_archive, ae)
            evaluations += len(a)+len(b)
            run_scores[key] = dict(original=a, intervened=b)
            metadata = prepared[key][1]
            for name in ('initial', 'first', 'middle', 'last', 'two_suffix'):
                for relevant in (False, True):
                    ix = [i for i, v in enumerate(metadata) if v['name'] == name and v['relevant'] == relevant]
                    if not ix: continue
                    oi = [metadata[i]['original_index'] for i in ix]
                    val = summarize(a[oi], b[ix], y[oi], ny[ix])
                    scalar = scalar_summary(ra[oi], rb[ix], y[oi], ny[ix])
                    se = max(abs(val[k]-scalar[k]) for k in val)
                    assert se <= 1e-10; max_scalar = max(max_scalar, se)
                    metrics.append(dict(seed=r['seed'], recipe=r['recipe'], pool=key,
                                        intervention=name, label_changes=relevant, **val))
            fs = [features(ids) for ids, _ in rows]
            for feature in fs[0]:
                for value in sorted({f[feature] for f in fs}):
                    ix = [i for i, f in enumerate(fs) if f[feature] == value]
                    correct = sum(int(a[i].argmax()) == int(y[i]) for i in ix)
                    assert correct == sum(int(ra[i].argmax()) == int(y[i]) for i in ix)
                    strata.append(dict(seed=r['seed'], recipe=r['recipe'], pool=key,
                                       feature=feature, value=value, count=len(ix), errors=len(ix)-correct,
                                       accuracy=correct/len(ix)))
        print(f"Checked {r['seed']} {r['recipe']}", flush=True)
    for seed in (2903, 2909):
        for key, rows in pools.items():
            y = tensors(rows)[1]
            a = scores[f'{seed}/constant_001'][key]['original'].argmax(-1) == y
            b = scores[f'{seed}/cosine_001'][key]['original'].argmax(-1) == y
            paired.append(dict(seed=seed, pool=key, count=len(y), constant_errors=int((~a).sum()),
                               scheduled_errors=int((~b).sum()), fixed=int((~a & b).sum()),
                               regressed=int((a & ~b).sum()), shared_errors=int((~a & ~b).sum())))
    assert all(digest(ROOT/p) == h for p, h in hashes.items())
    result = dict(scope='saved development checkpoints; training-only descriptive diagnostic',
                  preserved_sha256=hashes, paired=paired, strata=strata, interventions=metrics,
                  intervention_metadata={k:v[1] for k,v in prepared.items()},
                  verification=dict(passed=True, evaluations=evaluations, argmax_changes=0,
                                    max_logit_error=max_logit, max_archive_error=max_archive,
                                    max_scalar_error=max_scalar, wrong_label_control_errors=wrong_labels))
    target.mkdir()
    torch.save(scores, target/'scores.pt')
    result['scores_sha256'] = digest(target/'scores.pt')
    save_json(target/'report.json', result)
    for name, rows in [('paired', paired), ('strata', strata), ('interventions', metrics)]:
        with (target/f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
            writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result['verification']), flush=True)


if __name__ == '__main__':
    main()
