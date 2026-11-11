"""Reload every matched checkpoint and reproduce all offset metrics."""
import json
import itertools
from pathlib import Path
import torch
from torch.nn import functional as F
from reserved_cycle import model_new
from activation_paths import forward
from position_distractors import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    path = ROOT/'results/position_training.json'
    r = json.loads(path.read_text())
    assert r['phase'] == 'complete' and len(r['runs']) == 10
    for name, sha in r['source_sha256'].items():
        assert digest(ROOT/'experiments'/name) == sha
    assert digest(ROOT/'docs/position-training-protocol.md') == r['protocol_sha256']
    sp = ROOT/'results/position_training_splits.json'
    old = ROOT/'results/reserved_cycle_splits.json'
    assert digest(sp) == r['split_sha256'] and digest(old) == r['excluded_split_sha256']
    splits = json.loads(sp.read_text())
    def key(row):
        return tuple(sorted((row[i], row[i+1]) for i in (0, 2, 4)))
    sets = {n: {key(x) for x, y in rows} for n, rows in splits.items()}
    assert {n: len(v) for n, v in sets.items()} == dict(train=1600, validation=400, test=400)
    assert len(set.union(*sets.values())) == 2400
    previous = {key(x) for rows in json.loads(old.read_text()).values() for x, y in rows}
    assert not set.union(*sets.values()) & previous
    for rows in splits.values():
        assert all(dict(key(x))[x[-1]] == y and len(set(x[1:6:2])) == 3 for x, y in rows)
        assert all(sum(y == v for x, y in rows) == len(rows)//6 for v in range(12, 18))
        assert all(sum(x[:6:2].index(x[-1]) == i for x, y in rows) == len(rows)//3 for i in range(3))
        for start in range(0, len(rows), 18):
            block = rows[start:start+18]
            m = key(block[0][0])
            expected = {(tuple(t for pair in order for t in pair)+(18, q), v)
                        for order in itertools.permutations(m) for q, v in m}
            assert {(tuple(x), y) for x, y in block} == expected
    pairs = {}
    for run in r['runs']:
        pairs.setdefault(run['seed'], {})[run['group']] = run
    for pair in pairs.values():
        assert set(pair) == {'fixed', 'varied'}
        assert all(pair['fixed'][k] == pair['varied'][k] for k in ('initial_sha256', 'minibatch_sha256'))
        assert pair['fixed']['offset_counts'] == [3000, 0, 0, 0, 0]
        assert sum(pair['varied']['offset_counts']) == 3000 and min(pair['varied']['offset_counts']) > 0
    x = torch.tensor([x for x, y in splits['test']])
    y = torch.tensor([y for x, y in splits['test']])
    error = 0.
    with torch.no_grad():
        for run in r['runs']:
            assert set(run['test']) == {'0', '1', '2', '3', '4', '6', '8', '12'}
            cp = ROOT/run['checkpoint']
            assert digest(cp) == run['checkpoint_sha256']
            model = model_new().eval()
            model.load_state_dict(torch.load(cp, weights_only=True))
            for shift, expected in run['test'].items():
                scores = torch.cat([forward(model, chunk, int(shift))[0] for chunk in x.split(256)])
                correct = scores.argmax(-1) == y
                observed = dict(accuracy=correct.double().mean().item(), loss=F.cross_entropy(scores, y).item(),
                                every_order_query_correct=correct.reshape(400, 18).all(1).double().mean().item())
                error = max(error, *(abs(observed[k]-expected[k]) for k in observed))
            print(run['seed'], run['group'], 'reloaded and verified', flush=True)
    assert error < 1e-10
    c = r['criteria']
    assert c['training_range_all_90'] == all(p['varied']['test'][str(s)]['accuracy'] >= .9 for p in pairs.values() for s in range(5))
    assert c['zero_preserved'] == all(p['fixed']['test']['0']['accuracy']-p['varied']['test']['0']['accuracy'] <= .01 for p in pairs.values())
    for s, result in c['outside'].items():
        gain = sum(p['varied']['test'][s]['accuracy']-p['fixed']['test'][s]['accuracy'] for p in pairs.values())/5
        assert abs(gain-result['mean_gain']) < 1e-12
        assert result['all_90'] == all(p['varied']['test'][s]['accuracy'] >= .9 for p in pairs.values())
        assert result['gain_5'] == (gain >= .05)
    assert c['outside_conjunction'] == all(v['all_90'] and v['gain_5'] for v in c['outside'].values())
    result = dict(passed=True, report_sha256=digest(path), audit_source_sha256=digest(Path(__file__)),
                  checkpoints=10, evaluated_conditions=80, maximum_metric_error=error,
                  mapping_isolation=True, balance=True, paired_initialization_and_batches=True, criteria_verified=True)
    out = ROOT/'results/position_training_audit.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
