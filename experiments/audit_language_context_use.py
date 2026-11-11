"""Independent CPU equations and reductions for context-use interventions."""
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from audit_kv_cache_equivalence import norm, rotation
from wikitext2_training import ROOT, model_new, load_stream
from language_context_use import OUT, SEEDS, CONDITIONS, digest, write


def verify_inputs(inputs, stream):
    count = len(range(0, len(stream) - 64, 64))
    assert len(inputs['targets']) == count
    for index in range(count):
        start = index * 64
        assert inputs['starts'][index] == start
        assert torch.equal(inputs['ids'][index], stream[start:start+64])
        assert inputs['targets'][index] == stream[start+64]
    for field, width in (('earlier_order', 32), ('recent_order', 31)):
        assert torch.equal(inputs[field].sort(1).values, torch.arange(width).expand(count, width))
    generators = [torch.Generator().manual_seed(s) for s in (202610041, 202610042)]
    for generator, (field, width) in zip(generators, (('earlier_order', 32), ('recent_order', 31))):
        for row in inputs[field]:
            assert torch.equal(row, torch.randperm(width, generator=generator))
    return count


def independent_input(inputs, condition, first, last):
    rows = []
    for i in range(first, min(last, len(inputs['targets']))):
        original = inputs['ids'][i].tolist()
        if condition == 'recent_only':
            row = original[32:]
        elif condition == 'earlier_shuffle':
            row = [original[j] for j in inputs['earlier_order'][i].tolist()] + original[32:]
        elif condition == 'recent_shuffle':
            row = original[:32] + [original[32+j] for j in inputs['recent_order'][i].tolist()] + original[63:]
        else:
            assert condition == 'original'
            row = original
        assert row[-1] == original[-1]
        rows.append(row)
    return torch.tensor(rows), (36 if condition == 'recent_only' else 4)


def independent_logits(model, ids, start_position):
    x = model.embedding.weight[ids]
    n = ids.shape[1]
    positions = torch.arange(start_position, start_position+n)
    for block in model.blocks:
        z = norm(x, block.norm1)
        m = block.attn
        q, k, v = [z @ getattr(m, t+'_weight').T + getattr(m, t+'_bias') for t in ('q', 'k', 'v')]
        q = q.reshape(len(ids), n, 2, 8).permute(0, 2, 1, 3)
        k, v = [t.reshape(len(ids), n, model.groups, 8).permute(0, 2, 1, 3).expand(len(ids), 2, n, 8)
                for t in (k, v)]
        q, k = rotation(q, positions), rotation(k, positions)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).permute(0, 2, 1, 3).reshape(len(ids), n, 16)
        x = x + y @ m.out_proj.weight.T + m.out_proj.bias
        z = norm(x, block.norm2) @ block.linear1.weight.T + block.linear1.bias
        x = x + z.clamp_min(0) @ block.linear2.weight.T + block.linear2.bias
    return norm(x[:, -1], model.final_norm) @ model.output.weight.T


def independent_metrics(cell, base):
    ys, ps, ls = [cell[k].tolist() for k in ('targets', 'predictions', 'losses')]
    bp, bl = [base[k].tolist() for k in ('predictions', 'losses')]
    result = {}
    predicates = dict(all=lambda y: True, ordinary=lambda y: y > 1, unknown=lambda y: y == 0, row_end=lambda y: y == 1)
    for name, predicate in predicates.items():
        indices = [i for i, y in enumerate(ys) if predicate(y)]
        n = len(indices)
        row = dict(count=n, ce=None, perplexity=None, accuracy=None)
        if n:
            ce = math.fsum(ls[i] for i in indices) / n
            delta = math.fsum(ls[i]-bl[i] for i in indices) / n
            row.update(ce=ce, perplexity=math.exp(ce), accuracy=sum(ps[i] == ys[i] for i in indices)/n,
                delta_ce=delta, perplexity_ratio=math.exp(delta),
                loss_improved=sum(ls[i] < bl[i] for i in indices), loss_worsened=sum(ls[i] > bl[i] for i in indices),
                loss_equal=sum(ls[i] == bl[i] for i in indices),
                rescues=sum(bp[i] != ys[i] and ps[i] == ys[i] for i in indices),
                harms=sum(bp[i] == ys[i] and ps[i] != ys[i] for i in indices))
        result[name] = row
    return result


def verify_metrics(stored, actual):
    assert stored.keys() == actual.keys()
    for group, row in actual.items():
        assert stored[group].keys() == row.keys()
        for key, value in row.items():
            before = stored[group][key]
            if value is None or isinstance(value, int):
                assert before == value, (group, key)
            else:
                assert abs(before-value) <= 1e-10 * max(1, abs(value)), (group, key)


@torch.inference_mode()
def audit_checkpoint(checkpoint, vocabulary, inputs, cells, records, batch=8):
    model = model_new('mqa', vocabulary).float().double().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
    errors, differences, fixtures = [], 0, []
    cpu_cells = {}
    for condition in CONDITIONS:
        cell = cells[condition]
        assert torch.equal(cell['targets'], inputs['targets'])
        losses, preds = [], []
        for first in range(0, len(inputs['targets']), batch):
            ids, position = independent_input(inputs, condition, first, first+batch)
            scores = independent_logits(model, ids, position)
            y = inputs['targets'][first:first+batch]
            # Explicit log-partition instead of producer cross_entropy.
            loss = scores.logsumexp(1) - scores[torch.arange(len(y)), y]
            losses.append(loss); preds.append(scores.argmax(1))
            if first == 0:
                error = float((scores[:2] - cell['fixtures'][0]['logits'].double()).abs().max())
                assert error <= 1e-3
                fixtures.append(error)
        loss, pred = torch.cat(losses), torch.cat(preds)
        error = float((loss-cell['losses'].double()).abs().max())
        assert error <= 1e-4
        errors.append(error)
        differences += int((pred != cell['predictions']).sum())
        cpu_cells[condition] = dict(losses=loss, predictions=pred, targets=cell['targets'])
    for condition in CONDITIONS:
        actual = independent_metrics(cells[condition], cells['original'])
        verify_metrics(records[condition], actual)
        reference = independent_metrics(cpu_cells[condition], cpu_cells['original'])
        for group in actual:
            if actual[group]['count']:
                for key in ('ce', 'delta_ce'):
                    assert abs(actual[group][key]-reference[group][key]) <= 1e-5
    return dict(max_token_ce_error=max(errors), max_fixture_logit_error=max(fixtures), argmax_differences=differences)


def main():
    torch.set_num_threads(1)
    report = json.loads((OUT/'report.json').read_text())
    lock = json.loads((OUT/'lock.json').read_text())
    assert report['phase'] == 'complete' and report['lock_sha256'] == digest(OUT/'lock.json')
    for path, sha in lock['files'].items():
        assert digest(ROOT/path) == sha, path
    assert digest(OUT/'inputs.pt') == lock['inputs_sha256']
    assert digest(OUT/'scores.pt') == report['scores_sha256']
    inputs = torch.load(OUT/'inputs.pt', weights_only=True)
    count = verify_inputs(inputs, load_stream('validation'))
    saved = torch.load(OUT/'scores.pt', weights_only=True)
    assert len(saved) == len(report['cells']) == 12
    assert {(c['seed'],c['condition']) for c in saved} == {(s,c) for s in SEEDS for c in CONDITIONS}
    parent = json.loads((ROOT/'results/recurrent_language.json').read_text())
    checks = []
    for seed in SEEDS:
        cells = {c['condition']:c for c in saved if c['seed'] == seed}
        records = {c['condition']:c['metrics'] for c in report['cells'] if c['seed'] == seed}
        run = next(r for r in parent['runs'] if r['seed'] == seed and r['arm'] == 'transformer')
        checked = audit_checkpoint(ROOT/run['checkpoint'], report['vocabulary'], inputs, cells, records)
        checks.append(dict(seed=seed, **checked))
        print(json.dumps(checks[-1]), flush=True)
    result = dict(passed=True, report_sha256=digest(OUT/'report.json'), cells=12, predictions=count*12,
                  protected_files=len(lock['files']), checks=checks)
    write(OUT/'audit.json', result)


if __name__ == '__main__':
    main()
