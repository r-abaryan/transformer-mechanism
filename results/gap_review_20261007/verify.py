"""Read-only evidence review and first-query implementation check; no training."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
import torch
from torch.nn import functional as F
import copy_rope_model as rope
from copy_marker_model import make_model
from encoder_decoder_copy import inputs, logits


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = Path(__file__).with_name('verification.json')
    assert not destination.exists(), 'Preserve earlier review evidence.'
    torch.set_num_threads(1)
    reports, bindings, protected = {}, {}, {}
    for name in ('copy_marker_training', 'copy_marker_alignment',
                 'fitting_trajectory', 'map_exposure', 'shared_readout_recovery'):
        folder = ROOT / 'results' / name
        report = json.loads((folder / 'report.json').read_text())
        audit = json.loads((folder / 'audit.json').read_text())
        assert report['status'] == 'complete' and audit['passed']
        assert audit['report_sha256'] == digest(folder / 'report.json')
        reports[name] = report
        bindings[name] = {'report_sha256': digest(folder / 'report.json'),
                          'audit_sha256': digest(folder / 'audit.json')}
        for file in ('report.json', 'audit.json'):
            protected[str((folder / file).relative_to(ROOT))] = digest(folder / file)
    alignment = reports['copy_marker_alignment']
    rows = []
    for n in (6, 8, 10):
        cells = [c for c in alignment['cells'] if c['pool'] == f'test_{n}']
        native = [c for c in cells if c['mode'] == 'native']
        aligned = [c for c in cells if c['mode'] == 'aligned']
        assert len(native) == len(aligned) == 3
        rows.append({'length': n,
                     'native_full_mean': sum(c['metrics']['all']['exact_accuracy'] for c in native)/3,
                     'aligned_full_min': min(c['metrics'][g]['exact_accuracy'] for c in aligned for g in c['metrics']),
                     'first_token_errors': sum(c['diagnostics']['all']['first_error_counts'][0] for c in native),
                     'sequences': sum(c['metrics']['all']['sequences'] for c in native)})
    training = reports['copy_marker_training']
    query_checks = []
    original = rope.attention
    for run in training['runs']:
        if run['arm'] != 'rope':
            continue
        path = ROOT / run['checkpoint']
        assert digest(path) == run['checkpoint_sha256']
        protected[run['checkpoint']] = digest(path)
        model = make_model('rope').eval()
        model.load_state_dict(torch.load(path, weights_only=True)['model'])
        captured = []

        def observe(module, query, key, value, padding, qp, kp, *args, **kwargs):
            if module is model.base.decoder.multihead_attn:
                width = module.embed_dim // module.num_heads
                q = F.linear(query, module.in_proj_weight.chunk(3)[0], module.in_proj_bias.chunk(3)[0])
                q = q.reshape(len(query), query.shape[1], module.num_heads, width).transpose(1, 2)
                captured.append(rope.rotate(q, qp)[:, :, 0].detach().clone())
            return original(module, query, key, value, padding, qp, kp, *args, **kwargs)

        rope.attention = observe
        try:
            with torch.no_grad():
                for n in (6, 8, 10):
                    source = torch.tensor(training['evaluation_pools'][f'test_{n}'][:2])
                    for prefix in (torch.ones(2, 1, dtype=torch.long), inputs(source)[0]):
                        logits(model, source, prefix)
        finally:
            rope.attention = original
        assert len(captured) == 6
        discrepancy = max(float((q - captured[0][:1]).abs().max()) for q in captured)
        assert discrepancy < 1e-12
        query_checks.append({'seed': run['seed'], 'forward_calls': 6, 'max_first_query_difference': discrepancy})
    assert len(query_checks) == 3
    for name in ('copy_marker_model.py', 'copy_rope_model.py', 'encoder_decoder_copy.py', 'encoder_decoder_contract.py'):
        p = ROOT / 'experiments' / name
        protected[str(p.relative_to(ROOT))] = digest(p)
    assert all(digest(ROOT / p) == h for p, h in protected.items())
    output = {'scope': 'Review bindings and implementation constraint, not a rerun of research audits',
              'bindings': bindings, 'copy_summary': rows, 'first_query_checks': query_checks,
              'training_updates': 0, 'forward_calls': 18, 'protected_hashes': protected,
              'script_sha256': digest(Path(__file__)), 'passed': True}
    destination.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in output.items() if k not in ('protected_hashes', 'bindings')}, indent=2))


if __name__ == '__main__':
    main()
