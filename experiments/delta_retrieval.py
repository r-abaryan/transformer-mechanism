"""Bounded structured retrieval: softmax cache versus learned delta memory."""
import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/delta_retrieval'
ARMS = ('softmax', 'delta')
DEV_SEEDS = (3271, 3277)
CONFIRM_SEEDS = (3299, 3301, 3307)
RECIPES = ('constant', 'cosine')
STEPS = 3000
BATCH = 64
GROUPS = ('unique', 'repeat')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


class Reader(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.key = nn.Embedding(8, 16)
        self.value = nn.Embedding(6, 8)
        self.output = nn.Linear(8, 6)
        if arm == 'delta':
            self.write_logit = nn.Parameter(torch.tensor(2.))
        else:
            self.log_scale = nn.Parameter(torch.tensor(math.log(16.)))
            self.log_recency = nn.Parameter(torch.tensor(math.log(.5)))
        self.double()

    def keys(self, ids):
        return F.normalize(self.key(ids), dim=-1, eps=1e-12)

    def prefill(self, events):
        k = self.keys(events[..., 0])
        v = self.value(events[..., 1])
        if self.arm == 'softmax':
            return (k, v)
        s = v.new_zeros(len(events), 16, 8)
        beta = self.write_logit.sigmoid()
        for t in range(events.shape[1]):
            old = (k[:, t, :, None] * s).sum(1)
            s = s + beta * k[:, t, :, None] * (v[:, t] - old)[:, None, :]
        return (s,)

    def read(self, state, query):
        q = self.keys(query)
        if self.arm == 'softmax':
            k, v = state
            positions = torch.arange(k.shape[1], device=k.device, dtype=k.dtype)
            scores = self.log_scale.exp() * (k * q[:, None]).sum(-1)
            scores = scores + self.log_recency.exp() * (positions - (k.shape[1] - 1))
            read = (scores.softmax(-1)[..., None] * v).sum(1)
        else:
            read = (q[:, :, None] * state[0]).sum(1)
        return self.output(read)

    def forward(self, events, query):
        return self.read(self.prefill(events), query)


def pair(seed):
    torch.manual_seed(seed)
    first = Reader('softmax')
    second = Reader('delta')
    second.load_state_dict(first.state_dict(), strict=False)
    return dict(softmax=first, delta=second)


def reference(model, events, query, wrong=False):
    """Independent matrix transitions; softmax uses explicit stable exponentials."""
    key = model.key.weight[events[..., 0]]
    key = key / key.square().sum(-1, keepdim=True).clamp_min(1e-24).sqrt()
    value = model.value.weight[events[..., 1]]
    q = model.key.weight[query]
    q = q / q.square().sum(-1, keepdim=True).clamp_min(1e-24).sqrt()
    if model.arm == 'delta':
        state = value.new_zeros(len(events), 16, 8)
        eye = torch.eye(16, dtype=value.dtype, device=value.device)
        beta = torch.sigmoid(model.write_logit)
        for t in range(events.shape[1]):
            kt = key[:, t]
            transition = eye if wrong else eye - beta * kt[:, :, None] * kt[:, None, :]
            state = transition @ state + beta * kt[:, :, None] * value[:, t, None, :]
        read = (q[:, None, :] @ state).squeeze(1)
    else:
        logits = (key @ q[:, :, None]).squeeze(-1) * torch.exp(model.log_scale)
        bias = torch.tensor(list(range(events.shape[1])), dtype=value.dtype, device=value.device)
        if not wrong:
            logits = logits + torch.exp(model.log_recency) * bias
        weights = torch.exp(logits - logits.max(-1, keepdim=True).values)
        weights = weights / weights.sum(-1, keepdim=True)
        read = (weights[:, None, :] @ value).squeeze(1)
    return read @ model.output.weight.T + model.output.bias


def label(events, query):
    return next(value for key, value in reversed(events) if key == query)


def generate(seed, confirmation=False, exclude=()):
    rng = random.Random(seed)
    seen = set(exclude)
    specs = [('train', 4, 64), ('train', 8, 64), ('validation', 4, 16), ('validation', 8, 16)]
    if confirmation:
        specs += [('test', n, 32) for n in (4, 8, 16, 32)]
    pools = {}
    for split, n, count in specs:
        rows = []
        for group in GROUPS:
            accepted = 0
            for _ in range(100000):
                if accepted == count:
                    break
                q = rng.randrange(8)
                others = [k for k in range(8) if k != q]
                ev = [[rng.choice(others), rng.randrange(6)] for _ in range(n)]
                occurrences = 1 if group == 'unique' else rng.randint(2, min(6, n))
                places = sorted(rng.sample(range(n), occurrences))
                for p in places:
                    ev[p][0] = q
                if group == 'repeat':
                    ev[places[-1]][1] = (ev[places[0]][1] + rng.randint(1, 5)) % 6
                # All cyclic value renamings of a context stay in one pool,
                # regardless of query. Symbols/templates remain shared.
                family = min(tuple((k, (v + s) % 6) for k, v in ev) for s in range(6))
                if family in seen:
                    continue
                seen.add(family)
                accepted += 1
                for shift in range(6):
                    changed = [[k, (v + shift) % 6] for k, v in ev]
                    rows.append(dict(events=changed, query=q, label=label(changed, q), group=group))
            if accepted != count:
                raise RuntimeError('Data proposal limit reached.')
        pools[f'{split}_{n}'] = rows
    return dict(seed=seed, pools=pools)


def tensors(rows):
    return (torch.tensor([r['events'] for r in rows]),
            torch.tensor([r['query'] for r in rows]), torch.tensor([r['label'] for r in rows]))


def metrics(scores, rows):
    y = torch.tensor([r['label'] for r in rows], device=scores.device)
    correct = scores.argmax(-1) == y
    loss = F.cross_entropy(scores, y, reduction='none')
    result = {}
    for group in ('all', *GROUPS):
        mask = torch.tensor([group == 'all' or r['group'] == group for r in rows], device=scores.device)
        if not mask.any():
            continue
        result[group] = dict(count=int(mask.sum()), correct=int(correct[mask].sum()),
                             accuracy=correct[mask].double().mean().item(), loss=loss[mask].mean().item())
    return result


def evaluate(model, rows, independent=False):
    events, query, _ = tensors(rows)
    fn = reference if independent else lambda m, e, q: m(e, q)
    with torch.no_grad():
        scores = torch.cat([fn(model, e, q) for e, q in zip(events.split(128), query.split(128))])
    return metrics(scores, rows), scores


def audit_data(payload):
    ownership = {}
    baselines = {}
    for name, rows in payload['pools'].items():
        counts = {g: dict(n=0, first=0, last=0, constant=0, oracle=0) for g in GROUPS}
        for row in rows:
            ev, q = row['events'], row['query']
            # Labels are checked by a forward dictionary, not backward lookup.
            memory = {}
            for key, value in ev:
                memory[key] = value
            if memory[q] != row['label']:
                raise AssertionError('Wrong label')
            hits = [v for k, v in ev if k == q]
            assert (len(hits) == 1) == (row['group'] == 'unique')
            if row['group'] == 'repeat':
                assert hits[0] != hits[-1]
            family = min(tuple((k, (v+s) % 6) for k, v in ev) for s in range(6))
            assert ownership.setdefault(family, name) == name
            c = counts[row['group']]
            for key, value in dict(n=1, first=hits[0] == memory[q], last=ev[-1][1] == memory[q],
                                   constant=memory[q] == 0, oracle=True).items():
                c[key] += int(value)
        for c in counts.values():
            assert c['oracle'] == c['n'] and c['constant'] * 6 == c['n']
        baselines[name] = counts
    return baselines


def selection(runs):
    selected = {}
    for arm in ARMS:
        candidates = []
        for recipe in RECIPES:
            group = [r for r in runs if r['arm'] == arm and r['recipe'] == recipe]
            cells = [r['metrics'][p][g] for r in group if r['status']=='complete' for p in ('train_4', 'train_8') for g in GROUPS]
            fit = len(group) == len(DEV_SEEDS) and all(r['status'] == 'complete' for r in group) and all(c['accuracy'] >= .99 for c in cells)
            candidates.append(dict(recipe=recipe, fit=fit, worst_accuracy=min((c['accuracy'] for c in cells),default=None),
                                   worst_loss=max((c['loss'] for c in cells),default=None)))
        eligible = [c for c in candidates if c['fit']]
        chosen = min(eligible, key=lambda c: (c['worst_loss'], RECIPES.index(c['recipe'])))['recipe'] if eligible else None
        selected[arm] = dict(selected_recipe=chosen, candidates=candidates)
    return selected


def dependencies():
    paths = ['experiments/delta_retrieval.py', 'tests/test_delta_retrieval.py',
             'docs/protocols/delta-retrieval.md']
    return {p: digest(ROOT / p) for p in paths}


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in state.items():
        h.update(name.encode())
        h.update(bytes(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist()))
    return h.hexdigest()


def preflight():
    import importlib.util
    import unittest
    OUT.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location('test_delta_retrieval', ROOT / 'tests/test_delta_retrieval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if not result.wasSuccessful():
        raise RuntimeError('Preflight failed. Preserve failure; do not train.')
    profiles = []
    for arm, model in pair(3271).items():
        rows = generate(2026100407)['pools']['train_8'][:BATCH]
        e, q, y = tensors(rows)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        loss = F.cross_entropy(model(e, q), y)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        profiles.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),
                             optimizer_tensors=sum(torch.is_tensor(v) for s in optimizer.state.values() for v in s.values()),
                             initial_step_loss=loss.item(),device='cpu',dtype='float64'))
    write(OUT / 'preflight.json', dict(passed=True,methods=result.testsRun,profiles=profiles,
                                     source_dependencies=dependencies(),profile_optimizer_steps=2))


def train_run(phase, seed, arm, recipe, payload, report):
    model = pair(seed)[arm]
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    data = {name: tensors(rows) for name, rows in payload['pools'].items() if name.startswith('train_')}
    generator = torch.Generator().manual_seed(seed + 10000)
    stream = hashlib.sha256()
    run = dict(phase=phase,seed=seed,arm=arm,recipe=recipe,parameters=sum(p.numel() for p in model.parameters()),
               initial_sha256=state_hash(model.state_dict()),shared_sha256=state_hash({k:v for k,v in model.state_dict().items() if k.startswith(('key.', 'value.', 'output.'))}),
               status='running',curve=[],metrics={})
    report['runs'].append(run)
    write(OUT / 'report.json', report)
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        run['last_started_step'] = step
        if recipe == 'cosine':
            optimizer.param_groups[0]['lr'] = .003 * (.1 + .9 * .5 * (1 + math.cos(math.pi * (step-1)/(STEPS-1))))
        e, q, y = data[f'train_{4 if step % 2 else 8}']
        idx = torch.randint(len(e), (BATCH,), generator=generator)
        stream.update(bytes(idx.contiguous().view(torch.uint8).flatten().tolist()))
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(e[idx], q[idx]), y[idx])
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss; no restart allowed.')
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        if step % 1000 == 0:
            run['curve'].append(dict(step=step,loss=loss.item(),lr=optimizer.param_groups[0]['lr']))
            print(dict(phase=phase,seed=seed,arm=arm,recipe=recipe,step=step,loss=loss.item()),flush=True)
            write(OUT / 'report.json', report)
    cp = OUT / f'{phase}_{seed}_{arm}_{recipe}.pt'
    assert not cp.exists()
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=STEPS), cp)
    for name, rows in payload['pools'].items():
        if not name.startswith('test_'):
            run['metrics'][name], _ = evaluate(model, rows)
    run.update(status='complete',updates=STEPS,examples=STEPS*BATCH,input_events=STEPS*BATCH*6,
               elapsed_seconds=time.perf_counter()-started,stream_sha256=stream.hexdigest(),
               checkpoint=str(cp.relative_to(ROOT)).replace('\\', '/'),checkpoint_sha256=digest(cp))
    write(OUT / 'report.json', report)


def audit(report, datasets, expected_steps=STEPS):
    cells = []
    for run in report['runs']:
        if run['status'] != 'complete':
            continue
        cp = ROOT / run['checkpoint']
        assert digest(cp) == run['checkpoint_sha256']
        saved = torch.load(cp, weights_only=True)
        assert saved['step'] == expected_steps
        model = pair(run['seed'])[run['arm']]
        model.load_state_dict(saved['model'])
        for name, rows in datasets[run['phase']]['pools'].items():
            if run['phase'] == 'development' and name.startswith('test_'):
                raise AssertionError('Development test leakage')
            actual, scores = evaluate(model, rows)
            independent, other = evaluate(model, rows, independent=True)
            error = (scores-other).abs().max().item()
            changes = int((scores.argmax(-1) != other.argmax(-1)).sum())
            assert error < 1e-9 and changes == 0
            assert actual == run['metrics'][name]
            for g in actual:
                assert actual[g]['correct'] == independent[g]['correct']
                assert abs(actual[g]['loss']-independent[g]['loss']) < 1e-10
            cells.append(dict(phase=run['phase'],seed=run['seed'],arm=run['arm'],recipe=run['recipe'],
                              pool=name,predictions=len(rows),max_logit_error=error,argmax_changes=changes))
    # Independently reconstruct every minibatch stream and common initialization.
    for run in report['runs']:
        if run['status'] != 'complete':
            continue
        payload = datasets[run['phase']]
        rng = torch.Generator().manual_seed(run['seed']+10000)
        stream = hashlib.sha256()
        for step in range(1, expected_steps+1):
            count = len(payload['pools'][f'train_{4 if step%2 else 8}'])
            idx = torch.randint(count, (BATCH,), generator=rng)
            stream.update(bytes(idx.view(torch.uint8).tolist()))
        assert stream.hexdigest() == run['stream_sha256']
        assert state_hash(pair(run['seed'])[run['arm']].state_dict()) == run['initial_sha256']
    return dict(passed=True,cells=cells,predictions=sum(c['predictions'] for c in cells),
                max_logit_error=max(c['max_logit_error'] for c in cells),argmax_changes=0,
                streams_replayed=sum(r['status']=='complete' for r in report['runs']),source_dependencies=dependencies())


def confirm_decision(runs):
    result = {}
    for arm in ARMS:
        selected = [r for r in runs if r['phase']=='confirmation' and r['arm']==arm]
        ready=len(selected)==len(CONFIRM_SEEDS) and all(r['status']=='complete' for r in selected)
        result[arm] = dict(training_fit=ready and all(r['metrics'][p][g]['accuracy'] >= .99 for r in selected for p in ('train_4','train_8') for g in GROUPS),
                           reliability=ready and all(r['metrics'][p][g]['accuracy'] >= .95 for r in selected for p in ('validation_4','validation_8','test_4','test_8','test_16','test_32') for g in GROUPS))
    pairs = [{r['arm']:r for r in runs if r['phase']=='confirmation' and r['seed']==seed} for seed in CONFIRM_SEEDS]
    result['quality_preservation'] = all(result[a]['training_fit'] for a in ARMS) and all(p['delta']['metrics'][pool][g]['accuracy'] >= p['softmax']['metrics'][pool][g]['accuracy']-.02-1e-12
                                       for p in pairs for pool in ('test_4','test_8','test_16','test_32') for g in GROUPS)
    return result


def benchmark(report, payload):
    """Explicit cache/state payload and eager GPU costs, no optimized-kernel claim."""
    assert torch.cuda.is_available()
    cells = []
    training = []
    for run in [r for r in report['runs'] if r['phase']=='confirmation']:
        saved = torch.load(ROOT / run['checkpoint'], weights_only=True)
        cpu = Reader(run['arm']);cpu.load_state_dict(saved['model'])
        model = copy.deepcopy(cpu).float().cuda()
        # Check GPU FP32 predictions on every confirmation pool before timing.
        precision = []
        for name, rows in payload['pools'].items():
            e,q,_ = tensors(rows)
            with torch.no_grad():
                expected = cpu(e,q)
                actual = torch.cat([model(a.cuda(),b.cuda()).cpu().double() for a,b in zip(e.split(128),q.split(128))])
            changes = int((expected.argmax(-1)!=actual.argmax(-1)).sum())
            precision.append(dict(pool=name,argmax_changes=changes,max_logit_error=(expected-actual).abs().max().item()))
        if any(p['argmax_changes'] for p in precision):
            return dict(passed=False,reason='GPU precision changes',precision=precision)
        run['gpu_precision'] = precision
        for batch in (1,32):
            for n in (8,32,128):
                rng=torch.Generator().manual_seed(7000+n+batch)
                e=torch.stack((torch.randint(8,(batch,n),generator=rng),torch.randint(6,(batch,n),generator=rng)),dim=-1).cuda()
                q=torch.arange(batch,device='cuda')%8
                with torch.no_grad():
                    torch.cuda.synchronize()
                    baseline=torch.cuda.memory_allocated()
                    state=model.prefill(e)
                    torch.cuda.synchronize()
                    state_allocation=torch.cuda.memory_allocated()-baseline
                    payload_bytes=sum(t.numel()*t.element_size() for t in state)
                    expected=batch*(24*n if model.arm=='softmax' else 128)*4
                    assert payload_bytes==expected
                    full=model(e,q);incremental=model.read(state,q)
                    assert torch.allclose(full,incremental,atol=1e-5,rtol=1e-5)
                    timings={}
                    for mode in ('prefill','query'):
                        fn=(lambda:model.prefill(e)) if mode=='prefill' else (lambda:model.read(state,q))
                        sessions=[]
                        for session in range(2):
                            for _ in range(3):fn()
                            torch.cuda.synchronize();values=[]
                            for _ in range(5):
                                torch.cuda.synchronize();start=time.perf_counter();fn();torch.cuda.synchronize()
                                values.append((time.perf_counter()-start)*1000)
                            sessions.append(dict(samples_ms=values,median_ms=statistics.median(values)))
                        timings[mode]=sessions
                    cells.append(dict(seed=run['seed'],arm=run['arm'],batch=batch,events=n,
                                      payload_bytes=payload_bytes,gpu_state_allocation_bytes=state_allocation,timings=timings))
                    del state,full,incremental
                del e,q
        # Complete training-step allocator profile, including first Adam state creation.
        rows=payload['pools']['train_8'][:BATCH];e,q,y=[t.cuda() for t in tensors(rows)]
        optimizer=torch.optim.Adam(model.parameters(),lr=.003)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        base=torch.cuda.memory_allocated()
        loss=F.cross_entropy(model(e,q),y);loss.backward();optimizer.step();torch.cuda.synchronize()
        training.append(dict(seed=run['seed'],arm=run['arm'],batch=BATCH,events=8,
                             baseline_allocated_bytes=base,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                             peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        del optimizer,loss,model,e,q,y
        torch.cuda.empty_cache()
    reductions=[]
    for seed in CONFIRM_SEEDS:
        for batch in (1,32):
            pair={c['arm']:c for c in cells if c['seed']==seed and c['batch']==batch and c['events']==32}
            reductions.append(1-pair['delta']['payload_bytes']/pair['softmax']['payload_bytes'])
    return dict(passed=True,device=torch.cuda.get_device_name(0),dtype='float32',cells=cells,
                training_profiles=training,payload_reduction_at_32=reductions,
                memory_gate=all(r>=.5 for r in reductions),precision_prediction_changes=0,
                scope='Compact structured readers, eager PyTorch, wall clock with CUDA synchronization; two repeated timing sessions per case, not independent training runs.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('preflight','train'))
    args=parser.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if args.command=='preflight':
        preflight();return
    assert not (OUT/'report.json').exists()
    checked=json.loads((OUT/'preflight.json').read_text());assert checked['passed'] and checked['source_dependencies']==dependencies()
    dev=generate(2026100407);audit_data(dev);write(OUT/'development_data.json',dev)
    report=dict(status='development',source_dependencies=dependencies(),preflight_sha256=digest(OUT/'preflight.json'),
                data_sha256={'development':digest(OUT/'development_data.json')},runs=[],
                training_device='cpu',dtype='float64',python=sys.version,torch=torch.__version__,
                development_selection_uses_training_only=True)
    def attempt(phase,seed,arm,recipe,payload):
        try:
            train_run(phase,seed,arm,recipe,payload,report)
        except Exception as error:
            report['runs'][-1].update(status='failed',error=repr(error))
            write(OUT/'report.json',report)
            print(dict(phase=phase,seed=seed,arm=arm,recipe=recipe,error=repr(error)),flush=True)
    for recipe in RECIPES:
        for seed in DEV_SEEDS:
            for arm in ARMS if seed==DEV_SEEDS[0] else ARMS[::-1]:
                attempt('development',seed,arm,recipe,dev)
    report['selection']=selection(report['runs']);write(OUT/'development_locked.json',report)
    datasets={'development':dev}
    if all(v['selected_recipe'] is not None for v in report['selection'].values()):
        blocked={min(tuple((k,(v+s)%6) for k,v in r['events']) for s in range(6)) for rows in dev['pools'].values() for r in rows}
        fresh=generate(2026100413,confirmation=True,exclude=blocked);audit_data(fresh)
        # Reject family overlap between development and fresh confirmation.
        audit_data(dict(pools={**{'dev_'+k:v for k,v in dev['pools'].items()},**{'confirm_'+k:v for k,v in fresh['pools'].items()}}))
        write(OUT/'confirmation_data.json',fresh);datasets['confirmation']=fresh
        report['data_sha256']['confirmation']=digest(OUT/'confirmation_data.json')
        for seed in CONFIRM_SEEDS:
            for arm in ARMS if seed!=CONFIRM_SEEDS[1] else ARMS[::-1]:
                attempt('confirmation',seed,arm,report['selection'][arm]['selected_recipe'],fresh)
        write(OUT/'pretest.json',report)
        for run in [r for r in report['runs'] if r['phase']=='confirmation' and r['status']=='complete']:
            model=Reader(run['arm']);model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
            for name,rows in fresh['pools'].items():
                if name.startswith('test_'):run['metrics'][name],_=evaluate(model,rows)
        report['confirmation_decision']=confirm_decision(report['runs'])
    report['audit']=audit(report,datasets)
    if 'confirmation_decision' in report:
        decision=report['confirmation_decision']
        quality=all(decision[a]['training_fit'] and decision[a]['reliability'] for a in ARMS) and decision['quality_preservation']
        report['benchmark']=benchmark(report,datasets['confirmation']) if quality else dict(passed=False,reason='Confirmation quality gate fails; no benchmark executed')
        report['conditional_tradeoff_passes']=quality and report['benchmark']['passed'] and report['benchmark']['memory_gate']
    else:
        report['confirmation_skipped']='At least one arm failed development training fit.'
        report['conditional_tradeoff_passes']=False
    assert dependencies()==report['source_dependencies']
    report['status']='complete';write(OUT/'report.json',report)
    print(json.dumps(dict(status=report['status'],selection=report['selection'],confirmation=report.get('confirmation_decision'),
                          tradeoff=report['conditional_tradeoff_passes'],audit_predictions=report['audit']['predictions'])),flush=True)


if __name__=='__main__':main()
