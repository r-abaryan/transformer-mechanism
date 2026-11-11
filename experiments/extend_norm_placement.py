"""Fixed continuation with exact replay and resumable Adam state."""
import hashlib
import itertools
import json
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from norm_placement import ROOT, model_for
from reserved_cycle import evaluate
from position_training import shifted, metrics, weight_hash
from position_distractors import digest, logits


def identity(x):
    return tuple(sorted(zip(x[:-2:2], x[1:-2:2])))


def excluded_maps():
    excluded = set()
    for name in ('reserved_cycle_splits.json', 'position_training_splits.json',
                 'joint_routing_confirmation_inputs.json', 'canonical_positions_inputs.json',
                 'final_length_inputs.json', 'norm_placement_splits.json'):
        data = json.loads((ROOT/'results'/name).read_text())
        for rows in (data.values() if isinstance(data, dict) else [data]):
            excluded.update(identity(x) for x, y in rows if len(x) == 8)
    assert len(excluded) == 8420
    return excluded


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    old_path = ROOT/'results/norm_placement.json'
    old = json.loads(old_path.read_text())
    for name, sha in old['source_sha256'].items():
        assert digest(ROOT/'experiments'/name) == sha
    sp = ROOT/'results/norm_placement_splits.json'
    assert digest(sp) == old['split_sha256']
    splits = json.loads(sp.read_text())
    data = {n: (torch.tensor([x for x,y in rows]), torch.tensor([y for x,y in rows])) for n,rows in splits.items()}
    out = ROOT/'results/norm_extension.json'
    inp = ROOT/'results/norm_extension_inputs.json'
    directory = ROOT/'results/norm_extension_checkpoints'
    directory.mkdir(exist_ok=True)
    provenance = dict(original_sha256=digest(old_path), source_sha256=digest(Path(__file__)),
                      protocol_sha256=digest(ROOT/'docs/norm-extension-protocol.md'))
    if out.exists():
        report = json.loads(out.read_text())
        assert report['provenance'] == provenance
        assert digest(inp) == report['input_sha256']
        if report['phase'] == 'complete':
            print('Already complete; no rerun.')
            return
    else:
        assert not inp.exists()
        excluded = excluded_maps()
        rng = random.Random(20261230)
        maps = []
        for values in itertools.combinations(range(12,18),3):
            candidates = [tuple(zip(ks,vs)) for ks in itertools.combinations(range(12),3)
                          for vs in itertools.permutations(values) if tuple(zip(ks,vs)) not in excluded]
            rng.shuffle(candidates)
            maps.extend(candidates[:20])
        assert len(set(maps)) == 400 and not set(maps) & excluded
        rows = [([t for pair in order for t in pair]+[18,q],v)
                for m in maps for order in itertools.permutations(m) for q,v in m]
        inp.write_text(json.dumps(rows,separators=(',',':'))+'\n')
        report = dict(phase='training',provenance=provenance,input_sha256=digest(inp),runs=[])
    def save():
        tmp = out.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2)+'\n')
        tmp.replace(out)
    save()
    x,y = data['train']
    for original in old['runs']:
        seed,p,w = (original[n] for n in ('seed','placement','schedule'))
        path = directory/f'{p}_{w}_{seed}.pt'
        run = next((r for r in report['runs'] if (r['seed'],r['placement'],r['schedule'])==(seed,p,w)),None)
        if run and run['status'] in ('trained','failed'):
            assert digest(path) == run['checkpoint_sha256']
            continue
        torch.manual_seed(seed)
        model = model_for(p)
        assert weight_hash(model) == original['initial_sha256']
        opt = torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
        batches = torch.Generator().manual_seed(seed+1000)
        offsets = random.Random(seed+2000)
        start = 0
        if path.exists():
            state = torch.load(path,weights_only=True)
            assert state['provenance'] == provenance
            model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
            batches.set_state(state['batches']); offsets.setstate(state['offsets'])
            torch.set_rng_state(state['torch_rng'])
            start = state['step']; run = state['run']
            report['runs'] = [r for r in report['runs'] if (r['seed'],r['placement'],r['schedule'])!=(seed,p,w)]
            report['runs'].append(run)
        if run is None:
            run = dict(seed=seed,placement=p,schedule=w,status='running',curve=[])
            report['runs'].append(run)
        # Reconstruct hash accumulators without changing restored training RNGs.
        hb,ho = hashlib.sha256(),hashlib.sha256()
        rb = torch.Generator().manual_seed(seed+1000); ro=random.Random(seed+2000)
        for _ in range(start):
            hb.update(json.dumps(torch.randint(len(x),(64,),generator=rb).tolist()).encode())
            ho.update(bytes([ro.randrange(5)]))
        for step in range(start+1,6001):
            idx=torch.randint(len(x),(64,),generator=batches); shift=offsets.randrange(5)
            hb.update(json.dumps(idx.tolist()).encode()); ho.update(bytes([shift]))
            opt.param_groups[0]['lr']=.001*min(step/300,1) if w=='warmup' else .001
            opt.zero_grad(); loss=F.cross_entropy(shifted(model,x[idx],shift),y[idx])
            finite=bool(torch.isfinite(loss))
            if finite:
                loss.backward()
                finite=all(g.grad is not None and bool(torch.isfinite(g.grad).all()) for g in model.parameters())
            if finite: opt.step()
            else: run.update(status='failed',failure_step=step)
            if step == 3000:
                old_cp=ROOT/original['checkpoint']; assert digest(old_cp)==original['checkpoint_sha256']
                ref=torch.load(old_cp,weights_only=True)
                assert all(torch.equal(v,ref[n]) for n,v in model.state_dict().items())
                assert hb.hexdigest()==original['minibatch_sha256'] and ho.hexdigest()==original['offset_sha256']
                run['replay_exact']=True
            if step%500==0 or not finite:
                if step>=3000:
                    run['curve'].append(dict(step=step,**evaluate(model,data['validation'])))
                if step==6000 and finite: run['status']='trained'
                run.update(step=step,minibatch_sha256=hb.hexdigest(),offset_sha256=ho.hexdigest())
                state=dict(model=model.state_dict(),optimizer=opt.state_dict(),batches=batches.get_state(),
                           offsets=offsets.getstate(),torch_rng=torch.get_rng_state(),step=step,run=dict(run),provenance=provenance)
                tmp=path.with_suffix('.tmp'); torch.save(state,tmp); tmp.replace(path)
                run.update(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=digest(path)); save()
                print(json.dumps(dict(seed=seed,placement=p,schedule=w,step=step,status=run['status'],validation=run['curve'][-1:])),flush=True)
            if not finite: break
    assert len(report['runs'])==20
    report['phase']='all_training_attempts_finished_before_test'; save()
    rows=json.loads(inp.read_text()); tx=torch.tensor([x for x,y in rows]); ty=torch.tensor([y for x,y in rows])
    scores=[]
    for run in report['runs']:
        original=next(r for r in old['runs'] if all(r[n]==run[n] for n in ('seed','placement','schedule')))
        for step in (3000,6000):
            if step==6000 and run['status']!='trained': continue
            cp=ROOT/(original['checkpoint'] if step==3000 else run['checkpoint'])
            assert digest(cp)==(original['checkpoint_sha256'] if step==3000 else run['checkpoint_sha256'])
            state=torch.load(cp,weights_only=True)
            model=model_for(run['placement']).eval(); model.load_state_dict(state if step==3000 else state['model'])
            s=logits(model,tx); run[f'test_{step}']=metrics(s,ty)
            scores.append(dict(seed=run['seed'],placement=run['placement'],schedule=run['schedule'],step=step,scores=s.tolist()))
    dp=ROOT/'results/norm_extension_scores.json'; dp.write_text(json.dumps(scores,separators=(',',':'))+'\n')
    report.update(phase='complete',scores_sha256=digest(dp))
    save()


if __name__=='__main__': main()
