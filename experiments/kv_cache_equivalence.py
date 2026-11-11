"""Frozen bounded inference check; cached execution versus full RoPE forward."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from positional_comparison import forward as full_forward, rotate
from reserved_cycle import model_new
from position_distractors import digest

ROOT = Path(__file__).resolve().parents[1]
ROWS = (0, 1, 36, 37)
OFFSETS = (4, 17)
SCHEDULES = ('single', 'prefill')


def sizes(length, schedule):
    first = 1 if schedule == 'single' else length // 2
    return [first] + [1] * (length - first)


def cached_forward(model, ids, offset, schedule, mode='correct'):
    """Append each layer's K/V; only fresh queries are computed per chunk."""
    cache = [None] * len(model.blocks)
    outputs, trace = [], []
    cursor = 0
    for count in sizes(ids.shape[1], schedule):
        x = model.embedding(ids[:, cursor:cursor + count])
        positions = torch.arange(count) + offset + (0 if mode == 'reset' else cursor)
        visible = torch.arange(cursor + count)[None, :] <= torch.arange(cursor, cursor + count)[:, None]
        for layer, block in enumerate(model.blocks):
            z = block.norm1(x)
            q, k, v = F.linear(z, block.self_attn.in_proj_weight,
                               block.self_attn.in_proj_bias).chunk(3, -1)
            q, k, v = [t.reshape(len(ids), count, 2, 8).transpose(1, 2) for t in (q, k, v)]
            q, k = rotate(q, positions), rotate(k, positions)
            if cache[layer] is not None:
                old_k, old_v = cache[layer]
                k, v = torch.cat((old_k, k), -2), torch.cat((old_v, v), -2)
            cache[layer] = (k, v)
            allowed = visible.clone()
            if mode == 'omit' and cursor:
                allowed[:, 0] = False
            a = (q @ k.transpose(-1, -2) / (8 ** .5)).masked_fill(~allowed, float('-inf')).softmax(-1)
            h = (a @ v).transpose(1, 2).reshape(len(ids), count, 16)
            x = x + block.self_attn.out_proj(h)
            x = x + block.linear2(block.activation(block.linear1(block.norm2(x))))
        cursor += count
        shapes = [list(k.shape) for k, v in cache]
        byte_count = sum(t.numel() * t.element_size() for pair in cache for t in pair)
        assert all(shape == [len(ids), 2, cursor, 8] for shape in shapes)
        assert byte_count == len(ids) * len(model.blocks) * 2 * cursor * (8 + 8) * 8
        trace.append(dict(tokens=cursor,shapes=shapes,payload_bytes=byte_count))
        outputs.append(model.output(model.final_norm(x)))
    return torch.cat(outputs, 1), cache, trace


def discrepancy(reference, candidate):
    gaps = reference.topk(2, -1).values.diff(dim=-1).abs()
    return dict(max_logit_error=(reference-candidate).abs().max().item(),
                argmax_differences=int((reference.argmax(-1)!=candidate.argmax(-1)).sum()),
                min_reference_top2_gap=gaps.min().item(),
                reference_near_ties=int((gaps <= 1e-10).sum()))


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    target = ROOT/'results/kv_cache_equivalence.json'
    tensors = ROOT/'results/kv_cache_equivalence_tensors.pt'
    assert not target.exists() and not tensors.exists(), 'Preserve opened experiment artifacts.'
    prior_path = ROOT/'results/query_exposure.json'
    prior = json.loads(prior_path.read_text())
    assert prior['phase'] == 'complete' and len(prior['runs']) == 10
    for name, expected in prior['provenance'].items():
        path = ROOT/('docs/query-exposure-protocol.md' if name == 'protocol' else 'experiments/'+name)
        assert digest(path) == expected
    data_path = ROOT/'results/query_exposure_data.json'
    assert digest(data_path) == prior['input_sha256']
    data = json.loads(data_path.read_text())['datasets']
    paths = ('experiments/kv_cache_equivalence.py','experiments/audit_kv_cache_equivalence.py',
             'docs/kv-cache-protocol.md','docs/decisions/0002-kv-cache-equivalence.md')
    report = dict(phase='running',torch=torch.__version__,device='cpu',dtype='float64',threads=1,
                  prior_sha256=digest(prior_path),input_sha256=digest(data_path),
                  provenance={p:digest(ROOT/p) for p in paths},rows=list(ROWS),cases=[])
    saved = []
    with torch.no_grad():
        for run in prior['runs']:
            path = ROOT/run['checkpoint']
            assert digest(path) == run['checkpoint_sha256'] and run['status']=='trained'
            model = model_new().eval()
            model.load_state_dict(torch.load(path,weights_only=True)['model'])
            assert all(p.dtype == torch.float64 for p in model.parameters())
            assert all(b.dropout.p==b.dropout1.p==b.dropout2.p==0 for b in model.blocks)
            for pairs in (3,4,6):
                rows = [data[f'test_{pairs}'][i] for i in ROWS]
                ids = torch.tensor([x for x,y in rows]); labels = torch.tensor([y for x,y in rows])
                for offset in OFFSETS:
                    reference = full_forward(model, ids, 'rope', offset)
                    for schedule in SCHEDULES:
                        key = f"{run['seed']}_{run['group']}_{pairs}_{offset}_{schedule}"
                        scores, cache, trace = cached_forward(model,ids,offset,schedule)
                        reset, _, _ = cached_forward(model,ids,offset,schedule,'reset')
                        omitted, _, _ = cached_forward(model,ids,offset,schedule,'omit')
                        assert all(torch.isfinite(t).all() for t in (reference,scores,reset,omitted))
                        case = dict(key=key,seed=run['seed'],group=run['group'],pairs=pairs,offset=offset,
                                    schedule=schedule,checkpoint=run['checkpoint'],checkpoint_sha256=digest(path),
                                    correct=discrepancy(reference,scores),reset=discrepancy(reference,reset),
                                    omit=discrepancy(reference,omitted),trace=trace,
                                    reference_task_correct=int((reference[:,-1].argmax(-1)==labels).sum()),
                                    cached_task_correct=int((scores[:,-1].argmax(-1)==labels).sum()))
                        report['cases'].append(case)
                        saved.append(dict(key=key,full=reference,cached=scores,reset=reset,omit=omitted,cache=cache))
            print(f"Checked seed {run['seed']} / {run['group']}",flush=True)
    assert len(report['cases']) == 120
    checks = dict(equivalence=all(c['correct']['max_logit_error'] <= 1e-10 for c in report['cases']),
                  reset_detected=any(c['reset']['max_logit_error'] > 1e-6 for c in report['cases']),
                  omission_detected=any(c['omit']['max_logit_error'] > 1e-6 for c in report['cases']),
                  cache_lengths_and_payload=True)
    torch.save(saved,tensors)
    report.update(phase='complete',checks=checks,passed=all(checks.values()),tensors_sha256=digest(tensors))
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],checks=checks,
                         max_logit_error=max(c['correct']['max_logit_error'] for c in report['cases']))))
    if not report['passed']:
        raise SystemExit('Fixed cache contract failed; results retained.')


if __name__ == '__main__':
    main()
