"""Independent wide-model equations, GPU precision and frozen fit gates."""
import copy
import hashlib
import json
import math

import torch
from torch.nn import functional as F
from composition_learning_adequacy import ROOT, SEEDS, ARMS, PREFIX, DATA, pair, decision, EXAMPLES
from recurrent_composition import expand
from audit_recurrent_composition import audit_inputs, independent_metrics, difference
from audit_recurrent_language import independent as recurrent_equations
from audit_kv_cache_equivalence import norm
from recurrent_retrieval import write
from position_distractors import digest
from mqa_training import hash_state
from wikitext2_training import configure


def independent(model, ids, broken=False):
    if hasattr(model, 'gru'):
        return recurrent_equations(model, ids, reset=broken)
    x = model.embedding.weight[ids]
    batch, length, width = x.shape
    head = model.blocks[0].k.out_features
    heads = width//head
    positions = torch.arange(length, dtype=x.dtype, device=x.device)+4
    frequency = torch.tensor([10000**(-2*i/head) for i in range(head//2)], dtype=x.dtype, device=x.device)
    angle = positions[:, None]*frequency
    c, s = angle.cos(), angle.sin()
    rotation = torch.stack((c, s, -s, c), -1).reshape(length, head//2, 2, 2)
    for block in model.blocks:
        z = norm(x, block.norm1)
        q = (z@block.q.weight.T+block.q.bias).reshape(batch, length, heads, head).transpose(1, 2)
        k = (z@block.k.weight.T+block.k.bias).reshape(batch, length, 1, head).transpose(1, 2)
        v = (z@block.v.weight.T+block.v.bias).reshape(batch, length, 1, head).transpose(1, 2)
        q, k = [torch.einsum('bhnpi,npij->bhnpj', a.reshape(batch, a.shape[1], length, head//2, 2), rotation).flatten(-2)
                for a in (q, k)]
        logit = torch.einsum('bhid,bhjd->bhij', q, k.expand(-1, heads, -1, -1))/math.sqrt(head)
        if not broken:
            allowed = torch.arange(length, device=x.device)[:, None] >= torch.arange(length, device=x.device)[None, :]
            logit = torch.where(allowed, logit, -torch.inf)
        exponential = (logit-logit.max(-1, keepdim=True).values).exp()
        a = exponential/exponential.sum(-1, keepdim=True)
        h = torch.einsum('bhij,bhjd->bhid', a, v.expand(-1, heads, -1, -1)).transpose(1, 2).reshape(batch, length, width)
        x = x+h@block.out.weight.T+block.out.bias
        z = norm(x, block.norm2)@block.up.weight.T+block.up.bias
        x = x+z.clamp_min(0)@block.down.weight.T+block.down.bias
    return norm(x, model.norm)@model.output.weight.T


def relative(a, b):
    return (a-b).norm().item()/max(b.norm().item(), 1e-12)


def preflight(data):
    records = []
    x, y = [t[:9] for t in data['train_6']]
    for seed in SEEDS:
        models = pair(seed)
        assert torch.equal(models['transformer'].embedding.weight, models['gru'].embedding.weight)
        for arm, base in models.items():
            cpu = copy.deepcopy(base).double()
            gpu = copy.deepcopy(base).cuda()
            native, reference = cpu(x), independent(cpu, x)
            ga = torch.autograd.grad(F.cross_entropy(native[:, -1], y), tuple(cpu.parameters()))
            gb = torch.autograd.grad(F.cross_entropy(reference[:, -1], y), tuple(cpu.parameters()))
            error = (native-reference).abs().max().item()
            ge = max((a-b).abs().max().item() for a, b in zip(ga, gb))
            assert error <= 1e-10 and ge <= 1e-10
            actual = gpu(x.cuda())
            F.cross_entropy(actual[:, -1], y.cuda()).backward()
            output = actual.detach().cpu().double()
            rg = torch.cat([v.flatten() for v in gb])
            g = torch.cat([p.grad.detach().cpu().double().flatten() for p in gpu.parameters()])
            absolute, rel = (output-reference).abs().max().item(), relative(output, reference)
            gabs, grel = (g-rg).abs().max().item(), relative(g, rg)
            assert absolute <= 1e-4 and rel <= 1e-5 and gabs <= 1e-5 and grel <= 1e-4
            with torch.no_grad():
                changed = x.clone()
                changed[:, 7:] = (changed[:, 7:]+1)%15
                prefix = (gpu(x[:, :7].cuda())-actual[:, :7]).abs().max().item()
                future = (gpu(changed.cuda())[:, :7]-actual[:, :7]).abs().max().item()
                batch = (gpu(x[:1].cuda())-actual[:1]).abs().max().item()
                repeat = (gpu(x.cuda())-actual).abs().max().item()
                negative = (independent(cpu, x, broken=True)-native).abs().max().item()
            assert max(prefix, future, batch, repeat) <= 1e-6 and negative > 1e-6
            records.append(dict(seed=seed, arm=arm, cpu_logit_error=error, cpu_gradient_error=ge,
                gpu_logit_error=absolute, gpu_logit_relative=rel, gpu_gradient_error=gabs, gpu_gradient_relative=grel,
                prefix_error=prefix, future_error=future, batch_error=batch, repeat_error=repeat,
                intentional_unmasked_or_reset_error=negative))
            del cpu, gpu, native, reference, actual, ga, gb, rg, g
    return dict(passed=True, models=records)


def independent_decision(runs):
    result = dict(one_step_fit={}, one_step_transfer={}, two_step_fit={}, two_step_transfer={})
    for arm in ARMS:
        rows = [r for r in runs if r['arm'] == arm]
        ready = len(rows) == 3 and all(r['status'] == 'trained' for r in rows)
        onefit = twofit = ready
        onetest = twotest = ready
        for row in rows:
            if row['status'] != 'trained':
                continue
            for pool, cells in row['metrics'].items():
                n = int(pool.rsplit('_', 1)[1])
                limit = .90 if n == 8 else .99
                if pool.startswith('test_'):
                    onetest = onetest and cells['one_step']['accuracy'] >= limit
                    twotest = twotest and all(cells[g]['accuracy'] >= limit for g in ('two_forward', 'two_reverse'))
                else:
                    onefit = onefit and cells['one_step']['accuracy'] >= .99
                    twofit = twofit and all(cells[g]['accuracy'] >= .99 for g in ('two_step', 'two_forward', 'two_reverse'))
        result['one_step_fit'][arm] = onefit
        result['one_step_transfer'][arm] = onefit and onetest
        result['two_step_fit'][arm] = twofit
        result['two_step_transfer'][arm] = twofit and twotest
    result['composition_comparison_eligible'] = all(result['one_step_fit'].values())
    return result


def main():
    configure()
    path = ROOT/f'results/{PREFIX}.json'
    r = json.loads(path.read_text())
    assert r['phase'] == 'complete'
    target = ROOT/f'results/{PREFIX}_audit.json'
    assert not target.exists(), 'Preserve completed audits.'
    for name, h in {**r['provenance'], **r['historical']}.items():
        assert digest(ROOT/name) == h, name
    assert digest(ROOT/r['profile']) == r['profile_sha256'] and digest(DATA) == r['input_sha256']
    pre = json.loads((ROOT/f'results/{PREFIX}_pretest.json').read_text())
    assert pre['phase'] == 'all_attempts_finished_before_test' and len(pre['runs']) == 6
    assert all(not any(k.startswith('test_') for k in row['metrics']) for row in pre['runs'])
    assert [row['checkpoint_sha256'] for row in pre['runs']] == [row['checkpoint_sha256'] for row in r['runs']]
    data = expand(json.loads(DATA.read_text()))
    inputs = audit_inputs(json.loads(DATA.read_text()), data)
    cells, checked, total_changes = [], 0, 0
    by_seed = {seed:[] for seed in SEEDS}
    independent_runs = []
    robust = True
    for row in r['runs']:
        if row['status'] != 'trained':
            independent_runs.append(row)
            continue
        cp, sp = ROOT/row['checkpoint'], ROOT/row['scores']
        assert digest(cp) == row['checkpoint_sha256'] and digest(sp) == row['scores_sha256']
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        scores = torch.load(sp, map_location='cpu', weights_only=True)
        model = pair(row['seed'])[row['arm']]
        assert hash_state(model.state_dict()) == row['initial_sha256']
        assert hash_state({'embedding':model.embedding.weight}) == row['embedding_sha256']
        model.load_state_dict(saved['model'])
        model.double()
        assert saved['step'] == row['step'] == r['updates_per_run']
        assert all(int(s['step']) == row['step'] for s in saved['optimizer']['state'].values())
        gen = torch.Generator().manual_seed(row['seed']+1000)
        stream = hashlib.sha256()
        groups = {n:[torch.where(data[f'train_{n}'][0][:, -1] == h)[0] for h in (13, 14)] for n in (4, 6)}
        for step in range(1, r['updates_per_run']+1):
            n = 4 if step % 2 else 6
            idx = torch.cat([g[torch.randint(len(g), (r['effective_batch']//2,), generator=gen)] for g in groups[n]])
            stream.update(json.dumps(idx.tolist()).encode())
        assert stream.hexdigest() == row['stream_sha256']
        assert r['effective_batch']*r['updates_per_run'] == EXAMPLES
        by_seed[row['seed']].append(row)
        other_row = {**row, 'metrics':{}}
        for key, (x, y) in data.items():
            with torch.no_grad():
                other = torch.cat([independent(model, chunk)[:, -1] for chunk in x.split(256)])
            retained = scores[key].double()
            max_logit = (retained-other).abs().max().item()
            loss_error = (F.cross_entropy(retained, y, reduction='none')-F.cross_entropy(other, y, reduction='none')).abs().max().item()
            assert max_logit <= 1e-3 and loss_error <= 1e-3
            stored_metric = independent_metrics(retained, x, y)
            ref_metric = independent_metrics(other, x, y)
            assert difference(stored_metric, row['metrics'][key]) < 1e-12
            other_row['metrics'][key] = ref_metric
            changed = retained.argmax(-1) != other.argmax(-1)
            group_bounds = {}
            for group in ('one_step', 'two_step', 'two_forward', 'two_reverse'):
                if group == 'one_step':
                    mask = x[:, -1] == 13
                elif group == 'two_step':
                    mask = x[:, -1] == 14
                else:
                    # Independent dictionary determines the intermediate node.
                    from audit_recurrent_composition import oracle
                    selected = []
                    for item in x.tolist():
                        source = item[:-3:2]
                        forward = source.index(item[-2]) < source.index(oracle(item, one_only=True))
                        selected.append(item[-1] == 14 and forward == (group == 'two_forward'))
                    mask = torch.tensor(selected)
                count = int(mask.sum())
                changes = int(changed[mask].sum())
                delta = changes/count
                a = stored_metric[group]['accuracy']
                n = int(key.rsplit('_', 1)[1])
                threshold = .90 if n == 8 else .99
                stable = (a-delta >= threshold) if a >= threshold else (a+delta < threshold)
                robust = robust and stable
                group_bounds[group] = dict(argmax_changes=changes, accuracy_error_bound=delta, threshold=threshold, stable=stable)
                assert abs(a-ref_metric[group]['accuracy']) <= delta+1e-12
            cells.append(dict(seed=row['seed'], arm=row['arm'], pool=key, predictions=len(y),
                max_logit_error=max_logit, max_loss_error=loss_error, argmax_changes=int(changed.sum()),
                groups=group_bounds, independent_metrics=ref_metric))
            checked += len(y)
            total_changes += int(changed.sum())
            print(json.dumps(dict(seed=row['seed'], arm=row['arm'], pool=key, max_logit_error=max_logit,
                                  argmax_changes=int(changed.sum()))), flush=True)
        independent_runs.append(other_row)
    for rows in by_seed.values():
        if len(rows) == 2:
            assert rows[0]['stream_sha256'] == rows[1]['stream_sha256']
            assert rows[0]['embedding_sha256'] == rows[1]['embedding_sha256']
    recomputed = independent_decision(r['runs'])
    alternate = independent_decision(independent_runs)
    assert recomputed == r['decision'] == decision(r['runs']) == alternate
    assert robust, 'Precision uncertainty crosses a frozen classification threshold.'
    # Deliberately inflate failed fit cells: the gate must change.
    injected = copy.deepcopy(r['runs'])
    for row in injected:
        row['status'] = 'trained'
        for key, metric in row['metrics'].items():
            if not key.startswith('test_'):
                metric['one_step']['accuracy'] = 1.
    wrong = independent_decision(injected)
    # If both fit already, inject zero instead.
    if wrong['one_step_fit'] == recomputed['one_step_fit']:
        for row in injected:
            for key, metric in row['metrics'].items():
                if not key.startswith('test_'):
                    metric['one_step']['accuracy'] = 0.
        wrong = independent_decision(injected)
    assert wrong['one_step_fit'] != recomputed['one_step_fit']
    for name, h in r['historical'].items():
        assert digest(ROOT/name) == h
    result = dict(passed=True, report_sha256=digest(path), input_checks=inputs,
                  pools_checked=len(cells), predictions_checked=checked, argmax_changes=total_changes,
                  max_logit_error=max(c['max_logit_error'] for c in cells),
                  max_loss_error=max(c['max_loss_error'] for c in cells),
                  decision=recomputed, decision_precision_bounds_pass=robust, intentional_gate_error_detected=True,
                  historical_artifacts_preserved=len(r['historical']), frozen_files_checked=len(r['provenance']), cells=cells)
    write(target, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cells','input_checks','decision')}), flush=True)


if __name__ == '__main__':
    main()
