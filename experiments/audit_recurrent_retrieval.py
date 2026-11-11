"""Independent gate/attention arithmetic and full-pool retrieval audit."""
import hashlib
import json

import torch
from torch.nn import functional as F
from audit_mqa_training import independent as attention, audit_inputs
from recurrent_retrieval import ROOT, SEEDS, ARMS, pair, datasets, evaluate, decision, write
from mqa_training import hash_state
from position_distractors import digest


def independent(model, ids, reset=False):
    if not hasattr(model, 'gru'):
        return attention(model, ids)
    g = model.gru
    x = model.embedding.weight[ids]
    h = x.new_zeros(len(ids), g.hidden_size)
    outputs = []
    for t in range(ids.shape[1]):
        if reset:
            h = torch.zeros_like(h)
        ir, iz, inc = (x[:, t]@g.weight_ih_l0.T+g.bias_ih_l0).chunk(3, -1)
        hr, hz, hn = (h@g.weight_hh_l0.T+g.bias_hh_l0).chunk(3, -1)
        r, z = (ir+hr).sigmoid(), (iz+hz).sigmoid()
        candidate = (inc+r*hn).tanh()
        h = z*h+(1-z)*candidate
        outputs.append(h@model.output.weight.T)
    return torch.stack(outputs, 1)


def baselines(data):
    result = {}
    for key, (x, y) in data.items():
        keys, values = x[:, :-2:2], x[:, 1:-2:2]
        match = keys == x[:, -1:]
        first = values.gather(1, match.long().argmax(1)[:, None]).squeeze(1)
        latest = values.gather(1, (keys.shape[1]-1-match.flip(1).long().argmax(1))[:, None]).squeeze(1)
        assert torch.equal(latest, y)
        repeated = match.sum(1) == 2
        result[key] = {}
        for kind, mask in [('repeated', repeated), ('unique', ~repeated)]:
            result[key][kind] = {name: (pred[mask] == y[mask]).double().mean().item() for name, pred in
                                [('constant12', torch.full_like(y, 12)), ('first_match', first), ('last_value', values[:, -1]), ('oracle', latest)]}
            result[key][kind]['uniform_expected'] = 1/6
    return result


def preflight(payload, data):
    checks = audit_inputs(payload)
    rows = []
    for seed in SEEDS:
        models = pair(seed)
        assert torch.equal(models['gru'].embedding.weight, models['transformer'].embedding.weight)
        for arm, model in models.items():
            x, y = data['train_4']
            x, y = x[:9], y[:9]
            native = model(x)
            other = independent(model, x)
            error = (native-other).abs().max().item()
            a = torch.autograd.grad(F.cross_entropy(native[:, -1], y), tuple(model.parameters()))
            b = torch.autograd.grad(F.cross_entropy(other[:, -1], y), tuple(model.parameters()))
            grad_error = max((u-v).abs().max().item() for u, v in zip(a, b))
            assert error < 1e-10 and grad_error < 1e-10
            with torch.no_grad():
                prefix_error = (model(x[:, :5])-native[:, :5]).abs().max().item()
                batch_error = (model(x[:1])-native[:1]).abs().max().item()
                assert max(prefix_error, batch_error) < 1e-10
                negative = (independent(model, x, reset=True)-native).abs().max().item() if arm == 'gru' else None
                if arm == 'gru':
                    assert negative > 1e-6
            rows.append(dict(seed=seed, arm=arm, logit_error=error, gradient_error=grad_error,
                             prefix_error=prefix_error, batch_error=batch_error, reset_error=negative))
    return dict(passed=True, inputs=checks, models=rows)


def metrics(scores, x, y):
    correct = scores.argmax(-1) == y
    losses = scores.logsumexp(-1)-scores.gather(1, y[:, None]).squeeze(1)
    n = (x.shape[1]-2)//2
    result = dict(accuracy=correct.double().mean().item(), loss=losses.mean().item(),
                  all_query_context_accuracy=correct.reshape(-1, n-1).all(1).double().mean().item())
    for kind, count in [('repeated', 2), ('unique', 1)]:
        mask = (x[:, :-2:2] == x[:, -1:]).sum(1) == count
        result[kind+'_accuracy'] = correct[mask].double().mean().item()
        result[kind+'_loss'] = losses[mask].mean().item()
    return result


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rp = ROOT/'results/recurrent_retrieval.json'
    report = json.loads(rp.read_text())
    assert report['phase'] == 'complete'
    for path, expected in report['provenance'].items():
        assert digest(ROOT/path) == expected, path
    payload, data = datasets()
    checks = audit_inputs(payload)
    assert baselines(data) == report['baselines']
    pretest = json.loads((ROOT/'results/recurrent_retrieval_pretest.json').read_text())
    assert pretest['phase'] == 'all_attempts_finished_before_test'
    assert {(r['seed'], r['arm']) for r in report['runs']} == {(s, a) for s in SEEDS for a in ARMS}
    cells = []
    for seed in SEEDS:
        paired = [r for r in report['runs'] if r['seed'] == seed]
        assert len({r['embedding_sha256'] for r in paired}) == 1
        assert len({r['stream_sha256'] for r in paired}) == 1
        generator = torch.Generator().manual_seed(seed+1000)
        stream = hashlib.sha256()
        for step in range(1, 6001):
            n = 3 if step % 2 else 4
            stream.update(json.dumps(torch.randint(len(data[f'train_{n}'][0]), (64,), generator=generator).tolist()).encode())
        for run in paired:
            model = pair(seed)[run['arm']]
            assert hash_state(model.state_dict()) == run['initial_sha256']
            assert sum(p.numel() for p in model.parameters()) == run['parameters']
            cp = ROOT/run['checkpoint']
            assert digest(cp) == run['checkpoint_sha256']
            earlier = next(r for r in pretest['runs'] if r['seed'] == seed and r['arm'] == run['arm'])
            assert earlier['checkpoint_sha256'] == run['checkpoint_sha256']
            assert not any(k.startswith('test_') for k in earlier['metrics'])
            if run['status'] != 'trained':
                continue
            assert stream.hexdigest() == run['stream_sha256']
            state = torch.load(cp, weights_only=True)
            assert state['step'] == 6000 and all(v['step'].item() == 6000 for v in state['optimizer']['state'].values())
            model.load_state_dict(state['model'])
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
            sp = ROOT/run['scores']
            assert digest(sp) == run['scores_sha256']
            saved = torch.load(sp, weights_only=True)
            for key, (x, y) in data.items():
                _, scores = evaluate(model, (x, y), independent)
                error = (scores-saved[key]).abs().max().item()
                changes = int((scores.argmax(-1) != saved[key].argmax(-1)).sum())
                actual = metrics(scores, x, y)
                metric_error = max(abs(actual[k]-run['metrics'][key][k]) for k in actual)
                assert error < 1e-10 and changes == 0 and metric_error < 1e-10
                if key in earlier['metrics']:
                    assert earlier['metrics'][key] == run['metrics'][key]
                cells.append(dict(seed=seed, arm=run['arm'], split=key, predictions=len(y), logit_error=error, metric_error=metric_error, argmax_changes=changes))
            print(json.dumps(dict(audited_seed=seed, arm=run['arm'])), flush=True)
    assert decision(report['runs']) == report['decision']
    result = dict(passed=True, report_sha256=digest(rp), input_checks=checks, cells=cells,
                  evaluations=len(cells), predictions=sum(c['predictions'] for c in cells),
                  max_logit_error=max(c['logit_error'] for c in cells), max_metric_error=max(c['metric_error'] for c in cells))
    write(ROOT/'results/recurrent_retrieval_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('cells', 'input_checks')}), flush=True)


if __name__ == '__main__':
    main()
