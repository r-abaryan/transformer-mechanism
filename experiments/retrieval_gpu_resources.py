"""Separate resource diagnostic for frozen compact readers; no quality training."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import gc
import itertools
import json
import platform
import statistics
import subprocess
import sys
import time
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F
import delta_retrieval as parent

ROOT = parent.ROOT
OUT = ROOT / 'results/retrieval_gpu_resources'
PROTOCOL = ROOT / 'docs/protocols/retrieval-gpu-resources.md'
SHAPES = tuple(itertools.product((1, 32), (8, 32, 128)))
TRAIN_LENGTHS = (8, 32, 128)
WORKERS = (0, 1, 2)
WARMUPS, SAMPLES = 3, 10
ATOL, RTOL = 1e-4, 1e-4


def configure():
    assert torch.cuda.is_available(), 'A native CUDA GPU is required.'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def clean():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def dependencies(include_protocol=True):
    paths = ['experiments/retrieval_gpu_resources.py', 'experiments/audit_retrieval_gpu_resources.py',
             'tests/test_retrieval_gpu_resources.py', 'experiments/delta_retrieval.py']
    if include_protocol:
        paths.append(str(PROTOCOL.relative_to(ROOT)).replace('\\', '/'))
    return {p: parent.digest(ROOT / p) for p in paths}


def runs():
    report = json.loads((parent.OUT / 'report.json').read_text())
    assert report['source_dependencies'] == parent.dependencies()
    assert report['conditional_tradeoff_passes'] is False
    assert report['benchmark'] == dict(passed=False, reason='Confirmation quality gate fails; no benchmark executed')
    audit = json.loads((parent.OUT / 'independent_audit.json').read_text())
    assert audit['passed'] and audit['report_sha256'] == parent.digest(parent.OUT / 'report.json')
    chosen = [r for r in report['runs'] if r['phase'] == 'confirmation']
    assert {(r['seed'], r['arm']) for r in chosen} == set(itertools.product(parent.CONFIRM_SEEDS, parent.ARMS))
    assert len(chosen) == 6
    return sorted(chosen, key=lambda r: (r['seed'], r['arm']))


def load(run, device='cpu', rounded=False):
    path = ROOT / run['checkpoint']
    assert parent.digest(path) == run['checkpoint_sha256']
    model = parent.Reader(run['arm'])
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'])
    if rounded or device == 'cuda':
        model.float()
    return model.to(device).eval()


def fixture(batch, length):
    gen = torch.Generator().manual_seed(2026100421 + batch * 1000 + length)
    e = torch.stack((torch.randint(8, (batch, length), generator=gen),
                     torch.randint(6, (batch, length), generator=gen)), -1)
    q = e[:, 0, 0].clone()
    # The query is present; labels independently follow the last assignment.
    y = torch.tensor([dict(events)[int(query)] for events, query in zip(e.tolist(), q)])
    return e, q, y


def check_scores(actual, reference, require_argmax=True):
    actual = actual.detach().cpu().double()
    reference = reference.detach().cpu().double()
    diff = (actual - reference).abs()
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all(), 'Nonfinite logits'
    assert torch.all(diff <= ATOL + RTOL * reference.abs()), 'Numerical tolerance failed'
    changes = int((actual.argmax(-1) != reference.argmax(-1)).sum())
    if require_argmax:
        assert changes == 0, 'Argmax disagreement'
    return dict(max_absolute_error=float(diff.max()), argmax_changes=changes, predictions=len(actual))


def payload_bytes(state):
    assert len({t.untyped_storage().data_ptr() for t in state}) == len(state), 'Aliased state'
    return sum(t.untyped_storage().nbytes() for t in state)


def expected_payload(arm, batch, length):
    return 4 * batch * (length * 24 if arm == 'softmax' else 128)


def measured_time(operation):
    for _ in range(WARMUPS):
        result = operation()
        del result
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Initialize event handles outside the timed region.
    begin.record(); end.record(); torch.cuda.synchronize()
    wall, events = [], []
    for _ in range(SAMPLES):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        begin.record()
        result = operation()
        end.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - tick) * 1000)
        events.append(begin.elapsed_time(end))
        del result
    return dict(wall_ms=wall, event_ms=events, wall_median_ms=statistics.median(wall),
                event_median_ms=statistics.median(events))


def inference_case(run, batch, length, timed=True):
    clean()
    model = load(run, 'cuda')
    cpu = load(run, rounded=True).double()
    e, q, _ = fixture(batch, length)
    with torch.inference_mode():
        ref = parent.reference(cpu, e, q)
        eg, qg = e.cuda(), q.cuda()
        # Warm allocation paths before establishing the memory baseline.
        warm = model(eg, qg)
        del warm
        clean()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        state = model.prefill(eg)
        torch.cuda.synchronize()
        retained = torch.cuda.memory_allocated() - baseline
        prefill_peak = torch.cuda.max_memory_allocated()
        payload = payload_bytes(state)
        assert payload == expected_payload(run['arm'], batch, length), 'State payload mismatch'
        assert retained >= payload, 'State allocation below payload'
        torch.cuda.reset_peak_memory_stats()
        out = model.read(state, qg)
        torch.cuda.synchronize()
        query_peak = torch.cuda.max_memory_allocated()
        scores = out.cpu()
        numerical = check_scores(scores, ref)
        full = model(eg, qg)
        cache_check = check_scores(full, scores)
        assert torch.equal(full, out), 'Full and retained-state read disagree'
        del full, out
        cell = dict(seed=run['seed'], arm=run['arm'], batch=batch, length=length,
                    checkpoint_sha256=run['checkpoint_sha256'], numerical=numerical,
                    cache_check=cache_check, state_shapes=[list(t.shape) for t in state],
                    payload_bytes=payload, retained_allocated_bytes=retained,
                    baseline_allocated_bytes=baseline, prefill_peak_allocated_bytes=prefill_peak,
                    query_peak_allocated_bytes=query_peak)
        if timed:
            cell['prefill'] = measured_time(lambda: model.prefill(eg))
            cell['query'] = measured_time(lambda: model.read(state, qg))
        evidence = dict(seed=run['seed'], arm=run['arm'], batch=batch, length=length,
                        events=e, query=q, reference=ref, gpu_scores=scores)
    del model, cpu, state, eg, qg
    clean()
    return cell, evidence


def training_profile(run, length):
    clean()
    model = load(run, 'cuda')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, foreach=False)
    e, q, y = fixture(64, length)
    e, q, y = e.cuda(), q.cuda(), y.cuda()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    logits = model(e, q)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    torch.cuda.synchronize()
    cell = dict(seed=run['seed'], arm=run['arm'], batch=64, length=length,
                checkpoint_sha256=run['checkpoint_sha256'], updates=1, targets=64, events=64 * length,
                baseline_allocated_bytes=baseline, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
                optimizer_state_bytes=sum(v.numel() * v.element_size() for state in optimizer.state.values()
                                          for v in state.values() if isinstance(v, torch.Tensor)),
                optimizer_parameter_states=len(optimizer.state), loss=float(loss.detach().cpu()))
    assert len(optimizer.state) == len(list(model.parameters()))
    assert all(float(state['step']) == 1 for state in optimizer.state.values())
    del logits, loss, optimizer, model, e, q, y
    clean()
    return cell


def preserved_manifest():
    prior = json.loads((parent.OUT / 'prior_hashes.json').read_text())
    prior.update(parent.dependencies())
    prior['experiments/audit_delta_retrieval.py'] = parent.digest(ROOT / 'experiments/audit_delta_retrieval.py')
    recency = json.loads((ROOT / 'results/recency_ablation/report.json').read_text())
    prior.update(recency['source_dependencies'])
    for folder in (parent.OUT, ROOT / 'results/recency_ablation'):
        prior.update({str(p.relative_to(ROOT)).replace('\\', '/'): parent.digest(p)
                      for p in folder.iterdir() if p.is_file()})
    for p, h in prior.items():
        assert parent.digest(ROOT / p) == h, f'Prior evidence changed: {p}'
    return prior


def preflight():
    configure()
    OUT.mkdir(exist_ok=False)
    parent.write(OUT / 'preserved.json', preserved_manifest())
    # Import and execute the actual GPU helpers before freezing resource sizes.
    sys.path.insert(0, str(ROOT / 'tests'))
    import test_retrieval_gpu_resources as cases
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(cases))
    assert result.wasSuccessful(), 'Preflight tests failed'
    profiles, cells = [], []
    for run in runs():
        if run['seed'] == parent.CONFIRM_SEEDS[0]:
            profiles.append(training_profile(run, max(TRAIN_LENGTHS)))
            cell, _ = inference_case(run, 32, 128, timed=False)
            cells.append(cell)
    saved, pool_checks = [], []
    data = json.loads((parent.OUT / 'confirmation_data.json').read_text())
    with torch.inference_mode():
        for run in runs():
            original = load(run)
            rounded = load(run, rounded=True).double()
            model = load(run, 'cuda')
            for pool, rows in data['pools'].items():
                e, q, _ = parent.tensors(rows)
                gpu = torch.cat([model(a.cuda(), b.cuda()).cpu() for a, b in zip(e.split(128), q.split(128))])
                ref = torch.cat([parent.reference(rounded, a, b) for a, b in zip(e.split(128), q.split(128))])
                old = torch.cat([parent.reference(original, a, b) for a, b in zip(e.split(128), q.split(128))])
                checked = check_scores(gpu, ref)
                old_check = check_scores(gpu, old)
                pool_checks.append(dict(seed=run['seed'], arm=run['arm'], pool=pool,
                                        rounded_reference=checked, original_reference=old_check))
                saved.append(dict(seed=run['seed'], arm=run['arm'], pool=pool, gpu_scores=gpu))
            del model, rounded, original
            clean()
    torch.save(saved, OUT / 'preflight_scores.pt')
    parent.write(OUT / 'preflight.json', dict(passed=True, methods=result.testsRun,
                 source_dependencies=dependencies(False), profiles=profiles, cells=cells, pool_checks=pool_checks,
                 predictions=sum(c['rounded_reference']['predictions'] for c in pool_checks),
                 scores_sha256=parent.digest(OUT / 'preflight_scores.pt')))
    print('GPU preflight complete, all original pools checked.', flush=True)


def hardware():
    smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,temperature.gpu,power.draw',
                          '--format=csv'], capture_output=True, text=True, check=True).stdout.strip()
    return dict(torch=torch.__version__, cuda=torch.version.cuda, python=sys.version,
                platform=platform.platform(), device=torch.cuda.get_device_name(),
                total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                allocator=torch.cuda.get_allocator_backend(), tf32=False, dtype='float32',
                deterministic=True, cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'], smi=smi)


def worker(index):
    configure()
    lock = json.loads((OUT / 'lock.json').read_text())
    assert lock['source_dependencies'] == dependencies()
    path = OUT / f'worker_{index}.json'
    assert not path.exists()
    report = dict(status='running', worker=index, process_id=os.getpid(), hardware=hardware(),
                  source_dependencies=dependencies(), lock_sha256=parent.digest(OUT / 'lock.json'),
                  cells=[], profiles=[])
    chosen, shapes = runs(), list(SHAPES)
    if index % 2:
        chosen.reverse(); shapes.reverse()
    tensors = []
    for batch, length in shapes:
        for run in chosen:
            cell, evidence = inference_case(run, batch, length)
            report['cells'].append(cell); tensors.append(evidence)
        parent.write(path, report)
        print(f'Worker {index}: batch {batch}, events {length}, six readers measured.', flush=True)
    for run in chosen:
        for length in TRAIN_LENGTHS:
            report['profiles'].append(training_profile(run, length))
    torch.save(tensors, OUT / f'worker_{index}_scores.pt')
    report.update(status='complete', scores_sha256=parent.digest(OUT / f'worker_{index}_scores.pt'),
                  hardware_after=hardware())
    parent.write(path, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('preflight', 'measure'))
    parser.add_argument('--worker', type=int, choices=WORKERS)
    args = parser.parse_args()
    if args.phase == 'preflight':
        preflight(); return
    if args.worker is not None:
        worker(args.worker); return
    assert not (OUT / 'lock.json').exists(), 'Formal measurement already started'
    checked = json.loads((OUT / 'preflight.json').read_text())
    assert checked['passed'] and checked['source_dependencies'] == dependencies(False)
    assert PROTOCOL.exists()
    parent.write(OUT / 'lock.json', dict(source_dependencies=dependencies(), shapes=SHAPES,
                 workers=WORKERS, train_lengths=TRAIN_LENGTHS, warmups=WARMUPS, samples=SAMPLES,
                 atol=ATOL, rtol=RTOL, preflight_sha256=parent.digest(OUT / 'preflight.json'),
                 parent_report_sha256=parent.digest(parent.OUT / 'report.json')))
    for index in WORKERS:
        subprocess.run([sys.executable, __file__, 'measure', '--worker', str(index)], check=True)
    print('All three fresh worker processes complete. Run independent audit next.', flush=True)


if __name__ == '__main__':
    main()
