"""Audit saved evidence and reevaluate final checkpoints without training."""
import hashlib
import json
from pathlib import Path
import statistics

import torch
from compare_depth import StackedDecoder
from train_associative_recall import tensors, evaluate

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    report = json.loads((ROOT/'results/depth_confirmation.json').read_text())
    digest = lambda p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    assert digest(report['split_file']) == report['split_sha256']
    assert digest('docs/depth-confirmation-protocol.md') == report['protocol_sha256']
    for n,h in report['source_sha256'].items():
        assert digest('experiments/'+n) == h
    splits = json.loads((ROOT/report['split_file']).read_text())
    identities, mappings = {}, {}
    for name,rows in splits.items():
        identities[name], mappings[name] = set(), set()
        for x,y in rows:
            pairs = frozenset((x[i],x[i+1]) for i in range(0,len(x)-2,2))
            assert len(pairs) == (len(x)-2)//2
            assert sum(k==x[-1] and v==y for k,v in pairs) == 1
            identities[name].add((pairs,x[-1]))
            mappings[name].add(pairs)
        assert len(identities[name]) == len(rows)
    names = list(splits)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            assert not identities[a] & identities[b], (a,b)
    assert len(report['runs']) == 10
    assert len({(r['model'],r['seed']) for r in report['runs']}) == 10
    checks = []
    for r in report['runs']:
        assert digest(r['checkpoint']) == r['checkpoint_sha256']
        model = StackedDecoder(r['width'],r['hidden'],r['depth']).double()
        model.load_state_dict(torch.load(ROOT/r['checkpoint'],weights_only=True))
        assert sum(p.numel() for p in model.parameters()) == r['parameters']
        errors = {}
        for name in ('test','four_pairs','six_pairs'):
            actual = evaluate(model,tensors(splits[name]))
            errors[name] = max(abs(actual[k]-r['metrics'][name][k]) for k in actual)
            assert errors[name] < 1e-12
        checks.append(dict(model=r['model'],seed=r['seed'],max_metric_errors=errors))
    for seed in (137,163,191,223,257):
        assert len({r['minibatches_sha256'] for r in report['runs'] if r['seed']==seed}) == 1
    grouped = {(r['model'],r['seed']):r for r in report['runs']}
    gains = [grouped['deep',s]['metrics']['test']['accuracy']-grouped['wide',s]['metrics']['test']['accuracy']
             for s in (137,163,191,223,257)]
    assert statistics.mean(gains) == report['criteria']['mean_test_gain']
    assert statistics.stdev(gains) == report['criteria']['sd_test_gain']
    deep = [r for r in report['runs'] if r['model']=='deep']
    assert all(r['metrics'][n]['accuracy']>=.9 for r in deep for n in ('train','test')) == report['criteria']['deep_all_train_test_90']
    assert (statistics.mean(gains)>=.1) == report['criteria']['mean_test_gain_at_least_10_points']
    for n in ('four_pairs','six_pairs'):
        assert all(r['metrics'][n]['accuracy']>=.9 for r in deep) == report['criteria']['transfer_all_90'][n]
    audit = dict(report_sha256=digest('results/depth_confirmation.json'),
                 audit_source_sha256=digest('experiments/verify_depth_confirmation.py'),
                 canonical_splits_disjoint=True,labels_verified=True,
                 test_mapping_overlap_with_train=len(mappings['test']&mappings['train']),
                 test_distinct_mappings=len(mappings['test']),checkpoint_rechecks=checks)
    (ROOT/'results/depth_confirmation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
