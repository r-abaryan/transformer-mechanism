"""Matched final-only versus auxiliary operation-state supervision."""
import hashlib
import json
import time

import torch
from torch.nn import functional as F
from state_tracking import ROOT, STRATA, model_new, forward, tensors, metrics, save_json, provenance as prior_provenance
from mqa_training import hash_state
from position_distractors import digest

SEEDS = (2039,2053,2063)
ARMS = ('final','intermediate')
STEPS = 3000


def labels(ids):
    state = ids[:,0]-4
    result = []
    for pos in range(1,ids.shape[1]-1):
        op = ids[:,pos]
        state = torch.where(op==1,1-state,torch.where(op>=2,op-2,state))
        result.append(state)
    return torch.stack(result,1)


def objective(logits, targets, arm):
    final = F.cross_entropy(logits[:,-1],targets[:,-1])
    if arm=='intermediate':
        return final+F.cross_entropy(logits[:,1:-1].reshape(-1,2),targets.reshape(-1))
    assert arm=='final'
    return final


def evaluate(model, rows):
    x,y = tensors(rows)
    with torch.no_grad():
        logits = torch.cat([forward(model,b) for b in x.split(128)])
    targets = labels(x)
    assert torch.equal(targets[:,-1],y)
    m = metrics(logits[:,-1],y)
    m['operation_accuracy'] = (logits[:,1:-1].argmax(-1)==targets).double().mean().item()
    return m,logits


def provenance():
    paths = list(prior_provenance())+['experiments/state_supervision.py',
            'experiments/audit_state_supervision.py','docs/protocols/state-supervision.md']
    return {p:digest(ROOT/p) for p in paths}


def decision(runs):
    complete = len(runs)==6 and all(r['status']=='trained' and r['step']==STEPS for r in runs)
    checks = {'completed':complete}
    for seed in SEEDS:
        a,b = [next(r for r in runs if r['seed']==seed and r['arm']==arm) for arm in ARMS]
        def mean(run,split):
            return sum(run.get('evaluation',{}).get(f'{split}/12/{k}',{}).get('accuracy',0) for k in ('reset_free','long_tail'))/2
        checks[f'{seed}/validation_gain'] = complete and mean(b,'validation')-mean(a,'validation')>=.10-1e-12
        checks[f'{seed}/training_fit'] = complete and mean(b,'train')>=.90
        checks[f'{seed}/mixed_retention'] = complete and all(b['evaluation'][f'validation/{n}/mixed']['accuracy']>=a['evaluation'][f'validation/{n}/mixed']['accuracy']-.05-1e-12 for n in (8,12))
    return dict(passed=all(checks.values()),checks=checks)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rp = ROOT/'results/state_supervision.json'
    assert not rp.exists()
    ip = ROOT/'results/state_tracking_data.json'
    cells = json.loads(ip.read_text())
    prior = json.loads((ROOT/'results/state_tracking.json').read_text())
    assert digest(ip)==prior['input_sha256']
    from audit_state_supervision import preflight
    verification = preflight(cells)
    prov = provenance()
    report = dict(phase='training',provenance=prov,input_sha256=digest(ip),preflight=verification,
                  torch=torch.__version__,device='cpu',dtype='float64',runs=[])
    save_json(rp,report)
    train = {n:tensors([r for k in STRATA for r in cells[f'train/{n}/{k}']])[0] for n in (8,12)}
    target = {n:labels(x) for n,x in train.items()}
    directory = ROOT/'results/state_supervision_checkpoints'
    directory.mkdir(exist_ok=True)
    for pair_index,seed in enumerate(SEEDS):
        for arm in (ARMS if pair_index%2==0 else ARMS[::-1]):
            torch.manual_seed(seed)
            model = model_new()
            run = dict(seed=seed,arm=arm,initial_sha256=hash_state(model.state_dict()),status='running',step=0,
                       parameters=sum(p.numel() for p in model.parameters()),curve=[])
            report['runs'].append(run)
            opt = torch.optim.Adam(model.parameters(),lr=.001)
            gen = torch.Generator().manual_seed(seed+1000)
            stream = hashlib.sha256()
            tick = time.perf_counter()
            for step in range(1,STEPS+1):
                n = 8 if step%2 else 12
                ix = torch.randint(len(train[n]),(64,),generator=gen)
                stream.update(json.dumps(ix.tolist()).encode())
                opt.zero_grad()
                loss = objective(forward(model,train[n][ix]),target[n][ix],arm)
                finite = bool(torch.isfinite(loss))
                if finite:
                    loss.backward()
                    finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                if finite:
                    opt.step()
                    finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
                run['step'] = step
                if not finite:
                    run.update(status='failed',failure_step=step)
                    break
                if step%500==0:
                    scores = {k:evaluate(model,r)[0] for k,r in cells.items() if k.startswith(('train/','validation/'))}
                    run['curve'].append(dict(step=step,metrics=scores))
                    save_json(rp,report)
                    print(json.dumps(dict(seed=seed,arm=arm,step=step,train12={k:scores[f'train/12/{k}']['accuracy'] for k in STRATA})),flush=True)
            if run['status']!='failed':
                run['status']='trained'
            run.update(elapsed_seconds=time.perf_counter()-tick,stream_sha256=stream.hexdigest())
            cp = directory/f'{seed}_{arm}.pt'
            assert not cp.exists()
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),generator=gen.get_state(),provenance=prov),cp)
            run.update(checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=digest(cp))
            save_json(rp,report)
    scores = {}
    # Freeze validation-only decision before any test scoring.
    for run in report['runs']:
        if run['status']!='trained':
            continue
        model = model_new().eval()
        model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        run['evaluation'] = {}
        name = f"{run['seed']}/{run['arm']}"
        scores[name] = {}
        for key,rows in cells.items():
            if not key.startswith('test/'):
                run['evaluation'][key],scores[name][key] = evaluate(model,rows)
    report.update(phase='validation_decision_frozen',decision=decision(report['runs']))
    dp = ROOT/'results/state_supervision_decision.json'
    assert not dp.exists()
    save_json(dp,dict(decision=report['decision'],provenance=prov,checkpoint_hashes=[r['checkpoint_sha256'] for r in report['runs']]))
    report['decision_sha256'] = digest(dp)
    save_json(rp,report)
    for run in report['runs']:
        if run['status']!='trained':
            continue
        model = model_new().eval()
        model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        name = f"{run['seed']}/{run['arm']}"
        for key,rows in cells.items():
            if key.startswith('test/'):
                run['evaluation'][key],scores[name][key] = evaluate(model,rows)
    sp = ROOT/'results/state_supervision_scores.pt'
    assert not sp.exists()
    torch.save(scores,sp)
    report.update(phase='complete',scores_sha256=digest(sp))
    save_json(rp,report)
    print(json.dumps(report['decision']),flush=True)


if __name__=='__main__':
    main()
