"""Fixed matched-cost FFN comparison; earlier executed sources remain immutable."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from wikitext2_training import ROOT, configure, load_stream, angles, tensor_hash
from verify_wikitext2_baseline import model_new as template_model
from mqa_decoding import rotate_fp
from position_distractors import digest

SEEDS = (1783, 1801, 1823)
ARMS = ('relu', 'swiglu')
PROTOCOL = 'docs/decisions/0004-swiglu-matched-ffn.md'


def model_new(arm, vocabulary, seed):
    assert arm in ARMS
    torch.manual_seed(seed)
    model = template_model('mqa', vocabulary).float()
    generator = torch.Generator().manual_seed(seed + 3000)
    hidden = 33 if arm == 'relu' else 22
    def linear(n_in, n_out):
        # Constructors use an isolated RNG too; only explicit uniform draws define weights.
        with torch.random.fork_rng(devices=[]):
            layer = nn.Linear(n_in, n_out, bias=False)
        with torch.no_grad():
            layer.weight.uniform_(-1/math.sqrt(n_in), 1/math.sqrt(n_in), generator=generator)
        return layer
    for block in model.blocks:
        block.linear1 = linear(16, hidden)
        if arm == 'swiglu':
            block.content = linear(16, hidden)
        block.linear2 = linear(hidden, 16)
    return model


def common_hash(model):
    state = {k:v for k,v in model.state_dict().items()
             if not any(part in k.split('.') for part in ('linear1', 'linear2', 'content'))}
    h = hashlib.sha256()
    for name, t in sorted(state.items()):
        h.update(json.dumps([name, list(t.shape), t.tolist()]).encode())
    return h.hexdigest()


def forward_device(model, ids):
    # The frozen earlier forward hardcodes ReLU. This counterpart changes only the FFN.
    n = ids.shape[1]
    x = model.embedding(ids)
    c, s = angles(n, ids.device, x.dtype)
    mask = torch.ones(n, n, device=ids.device, dtype=torch.bool).triu(1)
    for block in model.blocks:
        z = block.norm1(x)
        m = block.attn
        q, k, v = [F.linear(z, getattr(m, t+'_weight'), getattr(m, t+'_bias')) for t in ('q','k','v')]
        q = q.reshape(len(ids), n, 2, 8).transpose(1,2)
        k, v = [t.reshape(len(ids), n, model.groups, 8).transpose(1,2) for t in (k,v)]
        q, k = rotate_fp(q,c,s), rotate_fp(k,c,s)
        k, v = [t.repeat_interleave(2//model.groups,dim=1) for t in (k,v)]
        a = (q@k.transpose(-1,-2)/math.sqrt(8)).masked_fill(mask,-torch.inf).softmax(-1)
        y = (a@v).transpose(1,2).reshape(len(ids),n,16)
        x = x+m.out_proj(y)
        z = block.norm2(x)
        hidden = F.silu(block.linear1(z))*block.content(z) if hasattr(block,'content') else F.relu(block.linear1(z))
        x = x+block.linear2(hidden)
    return model.output(model.final_norm(x))


def evaluate(model, stream, retain=False):
    # Same audited chunk contract as Decision0003, with the new FFN forward.
    complete, tail = divmod(len(stream)-1,64)
    losses, predictions, fixtures = [], [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for first in range(0,complete,8):
            positions = torch.arange(first,min(first+8,complete))[:,None]*64+torch.arange(64)[None,:]
            x, y = stream[positions].to(device), stream[positions+1].to(device)
            logits = forward_device(model,x)
            losses.append(F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').cpu())
            predictions.append(logits.argmax(-1).flatten().cpu())
            if retain and first==0:
                fixtures.append(dict(start=0,logits=logits[:2].cpu()))
        if tail:
            start = complete*64
            logits = forward_device(model,stream[start:start+tail][None,:].to(device))
            y = stream[start+1:start+tail+1].to(device)
            losses.append(F.cross_entropy(logits[0],y,reduction='none').cpu())
            predictions.append(logits[0].argmax(-1).cpu())
            if retain:
                fixtures.append(dict(start=start,logits=logits.cpu()))
    losses, predictions = torch.cat(losses), torch.cat(predictions)
    assert len(losses)==len(predictions)==len(stream)-1 and bool(losses.isfinite().all())
    mean = losses.double().mean().item()
    metrics = dict(targets=len(losses),loss=mean,perplexity=math.exp(mean),accuracy=(predictions==stream[1:]).double().mean().item())
    return metrics, dict(losses=losses,predictions=predictions,fixtures=fixtures) if retain else None


def provenance():
    earlier = json.loads((ROOT/'results/wikitext2_training.json').read_text())
    sources = dict(earlier['provenance'])
    assert all(digest(ROOT/name)==h for name,h in sources.items())
    for name in (PROTOCOL,'experiments/swiglu_math.py','experiments/swiglu_training.py',
                 'experiments/verify_swiglu_training.py','experiments/audit_swiglu_training.py'):
        sources[name] = digest(ROOT/name)
    return sources


def decision(runs, unigram_loss):
    checks = dict(completed=len(runs)==6 and all(r['status']=='trained' and r['step']==6000 for r in runs))
    checks['learning_sanity'] = len(runs)==6 and all('final_validation' in r and unigram_loss-r['final_validation']['loss']>=.05 for r in runs)
    differences = []
    for seed in SEEDS:
        pair = [next((r for r in runs if (r['seed'],r['arm'])==(seed,arm)),{}) for arm in ARMS]
        if all('final_validation' in r for r in pair):
            differences.append(pair[1]['final_validation']['loss']-pair[0]['final_validation']['loss'])
    ratios = [math.exp(d) for d in differences]
    mean = math.exp(sum(differences)/3) if len(differences)==3 else None
    checks['mean_improvement2'] = mean is not None and mean<=.98
    checks['no_worse_seed'] = len(ratios)==3 and max(ratios)<=1.
    return dict(adopt_swiglu=all(checks.values()),checks=checks,geometric_mean_ratio=mean,paired_ratios=ratios,metric='final_validation')


def main():
    configure()
    assert torch.cuda.is_available()
    rp = ROOT/'results/swiglu_training.json'
    dp = ROOT/'results/swiglu_training_scores.pt'
    directory = ROOT/'results/swiglu_training_checkpoints'
    directory.mkdir(exist_ok=True)
    prov = provenance()
    preflight = ROOT/'results/swiglu_training_verification.json'
    verified = json.loads(preflight.read_text())
    assert verified['passed'] and verified['provenance']==prov
    encoding = ROOT/'results/wikitext2_encoding.json'
    vocabulary = json.loads(encoding.read_text())['vocabulary']
    assert json.loads((ROOT/'results/wikitext2_data_audit.json').read_text())['passed']
    train, validation = load_stream('train'), load_stream('validation')
    if rp.exists():
        report = json.loads(rp.read_text())
        assert report['provenance']==prov and report['encoding_sha256']==digest(encoding)
        if report['phase']=='complete':
            print('Already complete.');return
    else:
        assert not dp.exists()
        report = dict(phase='training',provenance=prov,encoding_sha256=digest(encoding),
                      verification_sha256=digest(preflight),torch=torch.__version__,cuda=torch.version.cuda,
                      device=torch.cuda.get_device_name(0),platform=platform.platform(),runs=[])
    def save():
        tmp = rp.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2)+'\n')
        tmp.replace(rp)
    save()
    train_gpu = train.cuda()
    offsets = torch.arange(64,device='cuda')[None,:]
    for index, seed in enumerate(SEEDS):
        order = ARMS if index%2==0 else ARMS[::-1]
        for arm in order:
            cp = directory/f'{arm}_{seed}.pt'
            run = next((r for r in report['runs'] if (r['seed'],r['arm'])==(seed,arm)),None)
            if run and run['status'] in ('trained','failed'):
                assert digest(cp)==run['checkpoint_sha256'];continue
            model = model_new(arm,vocabulary,seed)
            initial, common = tensor_hash(model),common_hash(model)
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(),lr=.001)
            generator = torch.Generator().manual_seed(seed+1000)
            start = 0
            if cp.exists():
                state = torch.load(cp,map_location='cpu',weights_only=True)
                assert state['provenance']==prov
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
                generator.set_state(state['generator'])
                run = state['run'];start=run['step']
                report['runs'] = [r for r in report['runs'] if (r['seed'],r['arm'])!=(seed,arm)]+[run]
            else:
                run = dict(seed=seed,arm=arm,parameters=sum(p.numel() for p in model.parameters()),
                           initial_sha256=initial,common_initial_sha256=common,status='running',step=0,
                           targets=0,curve=[dict(step=0,validation=evaluate(model,validation)[0])],elapsed_seconds=0.)
                report['runs'].append(run)
            stream_hash = hashlib.sha256()
            replay = torch.Generator().manual_seed(seed+1000)
            for _ in range(start):
                stream_hash.update(json.dumps(torch.randint(len(train)-64,(8,),generator=replay).tolist()).encode())
            tick = time.perf_counter();window=[]
            for step in range(start+1,6001):
                starts = torch.randint(len(train)-64,(8,),generator=generator)
                stream_hash.update(json.dumps(starts.tolist()).encode())
                positions = starts.cuda()[:,None]+offsets
                optimizer.zero_grad()
                logits = forward_device(model,train_gpu[positions])
                loss = F.cross_entropy(logits.flatten(0,1),train_gpu[positions+1].flatten())
                finite = bool(loss.isfinite())
                if finite:
                    loss.backward()
                    finite = all(p.grad is not None for p in model.parameters()) and bool(torch.stack([p.grad.isfinite().all() for p in model.parameters()]).all())
                if finite:
                    optimizer.step()
                    finite = bool(torch.stack([p.isfinite().all() for p in model.parameters()]).all())
                    window.append(loss.detach())
                if not finite:
                    run.update(status='failed',failure_step=step)
                run['targets']+=512
                if step%1000==0 or not finite:
                    point = dict(step=step,finite_training=finite,
                                 training_window_mean_loss=torch.stack(window).mean().item() if window else None)
                    # Do not score a nonfinite model. Retain its failure checkpoint.
                    if finite:
                        point['validation'] = evaluate(model,validation)[0]
                    run['curve'].append(point);window=[]
                    run.update(step=step,stream_sha256=stream_hash.hexdigest())
                    if step==6000 and finite:
                        run['status']='trained'
                    torch.cuda.synchronize()
                    run['elapsed_seconds']+=time.perf_counter()-tick
                    state = dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                                 optimizer=optimizer.state_dict(),generator=generator.get_state(),run=dict(run),provenance=prov)
                    tmp = cp.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(cp)
                    run.update(checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=digest(cp));save();tick=time.perf_counter()
                    print(json.dumps(dict(seed=seed,arm=arm,step=step,finite=finite,
                                          validation_ppl=point.get('validation',{}).get('perplexity'))),flush=True)
                if not finite:break
            del model,optimizer
    assert len(report['runs'])==6
    report['phase']='all_attempts_finished_before_final_scoring';save()
    counts = torch.bincount(train,minlength=vocabulary).double()
    logp = -((counts+1)/(len(train)+vocabulary)).log()
    nll = logp[validation[1:]]
    report['unigram_validation'] = dict(loss=nll.mean().item(),perplexity=math.exp(nll.mean().item()))
    details=[]
    for run in report['runs']:
        if run['status']!='trained':continue
        model = model_new(run['arm'],vocabulary,run['seed']).cuda().eval()
        assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256']
        model.load_state_dict(torch.load(ROOT/run['checkpoint'],map_location='cpu',weights_only=True)['model'])
        run['final_validation'], detail = evaluate(model,validation,True)
        details.append(dict(seed=run['seed'],arm=run['arm'],validation=detail))
        del model
    report['decision'] = decision(report['runs'],report['unigram_validation']['loss'])
    report['phase']='validation_decision_frozen_before_test';save()
    frozen_decision_sha = hashlib.sha256(json.dumps(report['decision'],sort_keys=True).encode()).hexdigest()
    # Reused test is descriptive and cannot alter the validation decision.
    test = load_stream('test')
    nll = logp[test[1:]]
    report['unigram_test'] = dict(loss=nll.mean().item(),perplexity=math.exp(nll.mean().item()))
    for run in report['runs']:
        if run['status']!='trained':continue
        model = model_new(run['arm'],vocabulary,run['seed']).cuda().eval()
        model.load_state_dict(torch.load(ROOT/run['checkpoint'],map_location='cpu',weights_only=True)['model'])
        run['test'], detail = evaluate(model,test,True)
        next(d for d in details if (d['seed'],d['arm'])==(run['seed'],run['arm']))['test']=detail
        del model
    assert not dp.exists();torch.save(details,dp)
    assert hashlib.sha256(json.dumps(report['decision'],sort_keys=True).encode()).hexdigest()==frozen_decision_sha
    report.update(phase='complete',scores_sha256=digest(dp),validation_decision_sha256=frozen_decision_sha)
    save();print(json.dumps(report['decision']),flush=True)


if __name__=='__main__':main()
