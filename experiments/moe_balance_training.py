"""Paired auxiliary-loss training study with fixed batch-dependent capacity."""
import copy
import hashlib
import json
import math
import platform
import random
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

from moe_routing import ROOT, digest, dispatch, error
from moe_balancing import balance

SEEDS = (2819, 2833, 2837)
ARMS = {'off': 0., 'on': .01}
STEPS = 3000
BATCH = 128
PREFIX = 'moe_balance_training'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def state_hash(state):
    return hashlib.sha256(json.dumps({k: v.tolist() for k, v in state.items()}, sort_keys=True).encode()).hexdigest()


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(6, 16)
        self.router = nn.Parameter(torch.randn(16, 4) / 4)
        self.first = nn.Parameter(torch.randn(4, 16, 32) / 4)
        self.second = nn.Parameter(torch.randn(4, 32, 16) / math.sqrt(32))
        self.output = nn.Linear(16, 2)
        self.double()

    def forward(self, x):
        h = self.input(x).tanh()
        valid = torch.ones(len(x), dtype=torch.bool)
        branch, choices, accepted = dispatch([h, self.router, self.first, self.second], valid, 1.)
        aux, fractions, probabilities = balance(h @ self.router, valid)
        return self.output(h + branch), aux, choices, accepted.any(-1), fractions, probabilities


def reference(model, x, omit_gate=False):
    h = (x @ model.input.weight.T + model.input.bias).tanh()
    z = h @ model.router
    weights = (z-z.max(-1, keepdim=True).values).exp()
    p = weights / weights.sum(-1, keepdim=True)
    counts = [0]*4
    indices, keep = [], []
    cap = math.ceil(len(x)/4)
    for row in z.detach().tolist():
        e = max(range(4), key=lambda j: row[j])
        indices.append(e); counts[e] += 1; keep.append(counts[e] <= cap)
    chosen = torch.tensor(indices)
    retained = torch.tensor(keep)
    hidden = torch.einsum('td,edh->teh', h, model.first).tanh()
    values = torch.einsum('teh,ehd->ted', hidden, model.second)
    gates = F.one_hot(chosen, 4).to(x.dtype)*retained[:, None]
    branch = (values * (gates if omit_gate else gates*p)[:, :, None]).sum(1)
    scores = (h+branch) @ model.output.weight.T + model.output.bias
    fractions = torch.tensor(counts, dtype=x.dtype)/len(x)
    mean = p.mean(0)
    aux = sum(4*fractions[i]*mean[i] for i in range(4))
    return scores, aux, chosen, retained, fractions, mean


def encode(rows):
    return (torch.tensor([r['x'] for r in rows], dtype=torch.float64),
            torch.tensor([r['y'] for r in rows]), torch.tensor([r['group'] for r in rows]))


def make_data():
    rng = random.Random(2026100307)
    pools = {}
    seen = set()
    for split, per_cell in [('train', 256), ('validation', 128), ('test', 128)]:
        rows = []
        for group in range(2):
            for label in range(2):
                for j in range(per_cell):
                    sign_u = 1 if j % 2 else -1
                    equal_sign = bool(label) if group == 0 else not bool(label)
                    sign_v = sign_u if equal_sign else -sign_u
                    u, v = sign_u*rng.uniform(.2, 1), sign_v*rng.uniform(.2, 1)
                    x = [float(group == 0), float(group == 1), u, v, rng.uniform(-1, 1), rng.uniform(-1, 1)]
                    assert tuple(x) not in seen
                    seen.add(tuple(x))
                    rows.append(dict(x=x, y=label, group=group))
        rng.shuffle(rows)
        pools[split] = rows
    path = ROOT/f'results/{PREFIX}_data.json'
    assert not path.exists()
    write(path, pools)
    return pools


def audit_data(pools):
    all_inputs = []
    baselines = {}
    for split, rows in pools.items():
        values = {}
        for group in range(2):
            rs = [r for r in rows if r['group'] == group]
            correct = []
            for r in rs:
                x = r['x']
                # Boolean equality supplies a second label equation, without products.
                expected = int(((x[2]>0) == (x[3]>0)) != bool(group))
                assert expected == r['y'] and (expected ^ 1) != r['y']
                assert x[:2] == [float(group==0), float(group==1)]
                correct.append(expected)
                all_inputs.append(tuple(x))
            assert sum(correct) == len(rs)//2
            values[str(group)] = dict(oracle=1., constant_zero=sum(r['y']==0 for r in rs)/len(rs),
                                     sign_u=sum(int(r['x'][2]>0)==r['y'] for r in rs)/len(rs), examples=len(rs))
        baselines[split] = values
    assert len(set(all_inputs)) == len(all_inputs)
    return dict(disjoint_examples=len(all_inputs), shifted_label_control=True, baselines=baselines)


def evaluate(model, rows, fn=None):
    x, y, groups = encode(rows)
    records = []
    with torch.no_grad():
        for start in range(0, len(x), BATCH):
            records.append((fn or (lambda m, v: m(v)))(model, x[start:start+BATCH]))
    scores = torch.cat([r[0] for r in records])
    chosen = torch.cat([r[2] for r in records])
    accepted = torch.cat([r[3] for r in records])
    ce = scores.logsumexp(-1) - scores.gather(1, y[:, None]).squeeze(1)
    result = dict(ce=float(ce.mean()), accuracy=float((scores.argmax(-1)==y).double().mean()),
                  mean_batch_tv=sum(float((r[4]-.25).abs().sum()/2) for r in records)/len(records),
                  overflow=float((~accepted).double().mean()),
                  assignment_fractions=torch.bincount(chosen, minlength=4).double().div(len(y)).tolist(),
                  mean_aux=sum(float(r[1]) for r in records)/len(records), groups={})
    for group in range(2):
        ids = groups == group
        result['groups'][str(group)] = dict(accuracy=float((scores[ids].argmax(-1)==y[ids]).double().mean()),
                                           ce=float(ce[ids].mean()), overflow=float((~accepted[ids]).double().mean()),
                                           assignments=torch.bincount(chosen[ids], minlength=4).double().div(int(ids.sum())).tolist())
    return result, dict(scores=scores, chosen=chosen, accepted=accepted)


def preflight(pools):
    rows = []
    x,y,_ = encode(pools['train'][:16])
    for seed in SEEDS:
        torch.manual_seed(seed);m=Classifier()
        a,b=m(x),reference(m,x)
        ga=torch.autograd.grad(F.cross_entropy(a[0],y)+.01*a[1],tuple(m.parameters()))
        gb=torch.autograd.grad((b[0].logsumexp(-1)-b[0].gather(1,y[:,None]).squeeze()).mean()+.01*b[1],tuple(m.parameters()))
        oe=error(a[0],b[0]);ge=max(error(v,w) for v,w in zip(ga,gb));ae=error(a[1],b[1])
        bad=error(a[0],reference(m,x,True)[0])
        assert max(oe,ge,ae)<1e-10 and bad>1e-6
        assert torch.equal(a[2],b[2]) and torch.equal(a[3],b[3])
        rows.append(dict(seed=seed,output_error=oe,gradient_error=ge,aux_error=ae,omit_gate_effect=bad))
    return dict(labels=audit_data(pools),models=rows)


def profile():
    torch.set_num_threads(1);torch.manual_seed(2803)
    model=Classifier();x=torch.randn(BATCH,6,dtype=torch.float64);y=torch.arange(BATCH)%2
    opt=torch.optim.Adam(model.parameters(),lr=.003)
    start=time.perf_counter();scores,aux,*_=model(x)
    loss=F.cross_entropy(scores,y)+.01*aux;loss.backward();opt.step()
    assert len(opt.state)==len(list(model.parameters()))
    report=dict(device='cpu',dtype='float64',batch=BATCH,parameters=sum(p.numel() for p in model.parameters()),
                optimizer_states=len(opt.state),optimizer_state_bytes=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v)),
                seconds=time.perf_counter()-start,gpu_peak_allocated_bytes=0,gpu_peak_reserved_bytes=0,cpu_peak_memory_measured=False,
                source_sha256=digest(ROOT/'experiments/moe_balance_training.py'),python=platform.python_version(),torch=torch.__version__)
    path=ROOT/f'results/{PREFIX}_profile.json';assert not path.exists();write(path,report);print(report)


def decisions(runs):
    fit={arm:all(r['status']=='trained' and all(r['metrics'][s]['groups'][str(g)]['accuracy']>=threshold
             for s,threshold in [('train',.99),('validation',.95)] for g in range(2)) for r in runs if r['arm']==arm) for arm in ARMS}
    comparisons=[]
    for seed in SEEDS:
        pair={r['arm']:r for r in runs if r['seed']==seed}
        if any(r['status']!='trained' for r in pair.values()):continue
        for split in ['validation','test']:
            a,b=[pair[arm]['metrics'][split] for arm in ARMS]
            tv=a['mean_batch_tv']-b['mean_batch_tv'];drop=a['overflow']-b['overflow']
            quality=all(b['groups'][str(g)]['accuracy']>=a['groups'][str(g)]['accuracy']-.01 and
                        b['groups'][str(g)]['ce']<=a['groups'][str(g)]['ce']*1.05 for g in range(2))
            comparisons.append(dict(seed=seed,split=split,tv_reduction=tv,overflow_reduction=drop,
                                    quality_preserved=quality,passed=tv>=.05 and drop>=.02 and quality))
    return dict(fit=fit,paired=comparisons,adopt=all(fit.values()) and len(comparisons)==6 and all(c['passed'] for c in comparisons))


def train():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    path=ROOT/f'results/{PREFIX}.json';assert not path.exists()
    old={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'results').glob('*.json')}
    pools=make_data();x,y,g=encode(pools['train'])
    cells=[((g==group)&(y==label)).nonzero().flatten() for group in range(2) for label in range(2)]
    sources=['experiments/moe_balance_training.py','experiments/moe_routing.py','experiments/moe_balancing.py','docs/protocols/moe-balance-training.md',f'results/{PREFIX}_data.json',f'results/{PREFIX}_profile.json']
    report=dict(phase='preflight',provenance={p:digest(ROOT/p) for p in sources},historical=old,runs=[])
    write(path,report);report['preflight']=preflight(pools);write(path,report)
    directory=ROOT/f'results/{PREFIX}_checkpoints';directory.mkdir(exist_ok=False)
    for index,seed in enumerate(SEEDS):
        torch.manual_seed(seed);base=Classifier()
        for arm in list(ARMS)[::1 if index%2==0 else -1]:
            m=copy.deepcopy(base);opt=torch.optim.Adam(m.parameters(),lr=.003)
            gen=torch.Generator().manual_seed(seed+10000);stream=hashlib.sha256();tick=time.perf_counter()
            run=dict(seed=seed,arm=arm,alpha=ARMS[arm],status='running',initial_sha256=state_hash(m.state_dict()),curve=[],metrics={})
            report['runs'].append(run);write(path,report);completed=0
            try:
                for step in range(1,STEPS+1):
                    ids=torch.cat([cell[torch.randint(len(cell),(32,),generator=gen)] for cell in cells])
                    ids=ids[torch.randperm(BATCH,generator=gen)];stream.update(json.dumps(ids.tolist()).encode())
                    opt.zero_grad();scores,aux,_,accepted,f,_=m(x[ids]);ce=F.cross_entropy(scores,y[ids]);loss=ce+ARMS[arm]*aux
                    assert bool(torch.isfinite(loss));loss.backward()
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters())
                    opt.step();completed=step
                    if step%500==0:
                        row=dict(step=step,ce=float(ce.detach()),aux=float(aux.detach()),tv=float((f-.25).abs().sum()/2),overflow=float((~accepted).double().mean()))
                        run['curve'].append(row);write(path,report);print(dict(seed=seed,arm=arm,**row),flush=True)
                run['status']='trained'
            except Exception as exc:run.update(status='failed',failure=repr(exc))
            cp=directory/f'{seed}_{arm}.pt';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),steps=completed),cp)
            run.update(updates=completed,examples=completed*BATCH,supervised_targets=completed*BATCH,seconds=time.perf_counter()-tick,
                       stream_sha256=stream.hexdigest(),checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=digest(cp))
            if run['status']=='trained':
                for split in ['train','validation']:run['metrics'][split],_=evaluate(m,pools[split])
            write(path,report)
    report['phase']='all_attempts_finished_before_test';write(ROOT/f'results/{PREFIX}_pretest.json',report)
    for run in report['runs']:
        if run['status']!='trained':continue
        m=Classifier();m.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model']);saved={}
        for split in pools:
            metric,saved[split]=evaluate(m,pools[split])
            if split in run['metrics']:assert metric==run['metrics'][split]
            run['metrics'][split]=metric
        sp=directory/f"{run['seed']}_{run['arm']}_scores.pt";torch.save(saved,sp);run.update(scores=str(sp.relative_to(ROOT)),scores_sha256=digest(sp))
    for seed in SEEDS:
        pair=[r for r in report['runs'] if r['seed']==seed]
        assert pair[0]['initial_sha256']==pair[1]['initial_sha256']
        if all(r['status']=='trained' for r in pair):assert pair[0]['stream_sha256']==pair[1]['stream_sha256']
    report['decision']=decisions(report['runs']);report['phase']='closed_training'
    assert all(digest(ROOT/p)==h for p,h in old.items());write(path,report)
    print(json.dumps(report['decision'],indent=2),flush=True)


def audit():
    torch.set_num_threads(1)
    path=ROOT/f'results/{PREFIX}.json';r=json.loads(path.read_text());pools=json.loads((ROOT/f'results/{PREFIX}_data.json').read_text())
    audit_data(pools)
    assert all(digest(ROOT/p)==h for p,h in r['provenance'].items())
    assert all(digest(ROOT/p)==h for p,h in r['historical'].items())
    cells=[]
    for run in r['runs']:
        assert run['status']=='trained'
        assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256'] and digest(ROOT/run['scores'])==run['scores_sha256']
        cp=torch.load(ROOT/run['checkpoint'],weights_only=True)
        assert cp['steps']==STEPS and all(int(s['step'])==STEPS for s in cp['optimizer']['state'].values())
        m=Classifier();m.load_state_dict(cp['model']);saved=torch.load(ROOT/run['scores'],weights_only=True)
        for split,rows in pools.items():
            metrics,pred=evaluate(m,rows,reference)
            oe=error(pred['scores'],saved[split]['scores']);changes=int((pred['scores'].argmax(-1)!=saved[split]['scores'].argmax(-1)).sum())
            def numbers(v):
                if isinstance(v,dict):return sum([numbers(v[k]) for k in sorted(v)],[])
                if isinstance(v,list):return v
                return [v]
            me=max(abs(a-b) for a,b in zip(numbers(metrics),numbers(run['metrics'][split])))
            assert oe<1e-10 and me<1e-10 and changes==0
            assert torch.equal(pred['chosen'],saved[split]['chosen']) and torch.equal(pred['accepted'],saved[split]['accepted'])
            cells.append(dict(seed=run['seed'],arm=run['arm'],split=split,examples=len(rows),logit_error=oe,metric_error=me,argmax_changes=changes))
    out=dict(passed=True,cells=cells,evaluations=len(cells),predictions=sum(c['examples'] for c in cells),max_logit_error=max(c['logit_error'] for c in cells),max_metric_error=max(c['metric_error'] for c in cells),report_sha256=digest(path))
    target=ROOT/f'results/{PREFIX}_audit.json';assert not target.exists();write(target,out);print({k:v for k,v in out.items() if k!='cells'})


if __name__=='__main__':
    {'profile':profile,'train':train,'audit':audit}[sys.argv[1]]()
