"""One parameter-comparable GRU/Transformer state-tracking comparison."""
import hashlib
import json
import time

import torch
from torch import nn
from state_tracking import ROOT, STRATA, model_new as transformer_new, tensors, metrics, save_json
from state_supervision import labels, objective, provenance as prior_provenance
from mqa_training import hash_state
from position_distractors import digest

SEEDS = (2081,2099,2111)
ARMS = ('transformer','gru')
STEPS = 3000


class RecurrentState(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(7,16)
        self.rnn = nn.GRU(16,28,batch_first=True)
        self.output = nn.Linear(28,2,bias=False)
        self.double()

    def forward(self, ids):
        hidden,_ = self.rnn(self.embedding(ids))
        return self.output(hidden)


def make_pair(seed):
    torch.manual_seed(seed)
    transformer = transformer_new()
    torch.manual_seed(seed)
    gru = RecurrentState()
    with torch.no_grad():
        gru.embedding.weight.copy_(transformer.embedding.weight)
    return dict(transformer=transformer,gru=gru)


def evaluate(model, rows):
    x,y = tensors(rows)
    with torch.no_grad():
        logits = torch.cat([model(b) for b in x.split(128)])
    targets = labels(x)
    assert torch.equal(targets[:,-1],y)
    m = metrics(logits[:,-1],y)
    m['operation_accuracy'] = (logits[:,1:-1].argmax(-1)==targets).double().mean().item()
    return m,logits


def provenance():
    old = json.loads((ROOT/'results/state_supervision.json').read_text())
    assert old['provenance']==prior_provenance()
    paths = list(old['provenance'])+['experiments/recurrent_state.py',
            'experiments/audit_recurrent_state.py','docs/protocols/recurrent-state.md']
    return {p:digest(ROOT/p) for p in paths}


def decision(runs):
    complete = len(runs)==6 and all(r['status']=='trained' and r['step']==STEPS for r in runs)
    def acc(run,key):
        return run.get('evaluation',{}).get(key,{}).get('accuracy',0)
    records = {}
    for seed in SEEDS:
        t,g = [next(r for r in runs if r['seed']==seed and r['arm']==arm) for arm in ARMS]
        short = [f'{split}/{n}/{k}' for split in ('train','validation') for n in (8,12) for k in STRATA]
        keys = [f'test/{n}/{k}' for n in (16,24) for k in ('reset_free','long_tail')]
        gap = sum(acc(g,k)-acc(t,k) for k in keys)/4
        records[str(seed)] = dict(
            gru_short_fit=complete and all(acc(g,k)>=.99 for k in short),
            both_short_fit=complete and all(acc(r,k)>=.99 for r in (t,g) for k in short),
            gru_test_reliable=complete and all(acc(g,f'test/{n}/{k}')>=(.99 if n<=12 else .90) for n in (8,12,16,24) for k in STRATA),
            long_gap_pp=100*gap,long_advantage=complete and gap>=.10-1e-12)
    reliable = complete and all(v['gru_short_fit'] and v['gru_test_reliable'] for v in records.values())
    advantage = complete and all(v['long_advantage'] for v in records.values())
    return dict(passed=reliable and advantage,completed=complete,gru_reliability=reliable,
                relative_long_advantage=advantage,fit_comparable=complete and all(v['both_short_fit'] for v in records.values()),seeds=records)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rp = ROOT/'results/recurrent_state.json'
    assert not rp.exists()
    ip = ROOT/'results/state_tracking_data.json'
    oldp = ROOT/'results/state_supervision.json'
    auditp = ROOT/'results/state_supervision_audit.json'
    old = json.loads(oldp.read_text()); oldaudit = json.loads(auditp.read_text())
    assert oldaudit['passed'] and oldaudit['report_sha256']==digest(oldp) and digest(ip)==old['input_sha256']
    cells = json.loads(ip.read_text())
    from audit_recurrent_state import preflight
    prov = provenance()
    verification = preflight(cells)
    report = dict(phase='training',provenance=prov,input_sha256=digest(ip),
                  prior_report_sha256=digest(oldp),prior_audit_sha256=digest(auditp),
                  preflight=verification,torch=torch.__version__,device='cpu',dtype='float64',runs=[])
    save_json(rp,report)
    train = {n:tensors([r for k in STRATA for r in cells[f'train/{n}/{k}']])[0] for n in (8,12)}
    target = {n:labels(x) for n,x in train.items()}
    directory = ROOT/'results/recurrent_state_checkpoints'
    directory.mkdir(exist_ok=True)
    for pair_index,seed in enumerate(SEEDS):
        for arm in (ARMS if pair_index%2==0 else ARMS[::-1]):
            model = make_pair(seed)[arm]
            run = dict(seed=seed,arm=arm,initial_sha256=hash_state(model.state_dict()),
                       embedding_sha256=hash_state({'embedding':model.embedding.weight}),
                       parameters=sum(p.numel() for p in model.parameters()),status='running',step=0,curve=[])
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
                loss = objective(model(train[n][ix]),target[n][ix],'intermediate')
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
                    measured = {k:evaluate(model,r)[0] for k,r in cells.items() if k.startswith(('train/','validation/'))}
                    run['curve'].append(dict(step=step,metrics=measured))
                    save_json(rp,report)
                    print(json.dumps(dict(seed=seed,arm=arm,step=step,train12={k:measured[f'train/12/{k}']['accuracy'] for k in STRATA})),flush=True)
            if run['status']!='failed':
                run['status']='trained'
            run.update(elapsed_seconds=time.perf_counter()-tick,stream_sha256=stream.hexdigest())
            cp = directory/f'{seed}_{arm}.pt'
            assert not cp.exists()
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),generator=gen.get_state(),
                            rng=torch.get_rng_state(),provenance=prov),cp)
            run.update(checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=digest(cp))
            save_json(rp,report)
    saved = {}
    for run in report['runs']:
        if run['status']!='trained': continue
        model = make_pair(run['seed'])[run['arm']].eval()
        model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        run['evaluation'] = {}
        name = f"{run['seed']}/{run['arm']}"
        saved[name] = {}
        for key,rows in cells.items():
            if not key.startswith('test/'):
                run['evaluation'][key],saved[name][key] = evaluate(model,rows)
    dp = ROOT/'results/recurrent_state_pretest.json'
    assert not dp.exists()
    save_json(dp,dict(provenance=prov,runs=[{k:r[k] for k in ('seed','arm','status','step','checkpoint_sha256','evaluation') if k in r} for r in report['runs']]))
    report.update(phase='pretest_frozen',pretest_sha256=digest(dp))
    save_json(rp,report)
    for run in report['runs']:
        if run['status']!='trained': continue
        model = make_pair(run['seed'])[run['arm']].eval()
        model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        name = f"{run['seed']}/{run['arm']}"
        for key,rows in cells.items():
            if key.startswith('test/'):
                run['evaluation'][key],saved[name][key] = evaluate(model,rows)
    sp = ROOT/'results/recurrent_state_scores.pt'
    assert not sp.exists()
    torch.save(saved,sp)
    report.update(phase='complete',scores_sha256=digest(sp),decision=decision(report['runs']))
    save_json(rp,report)
    print(json.dumps(report['decision']),flush=True)


if __name__=='__main__':
    main()
