"""Bounded gain-only normalization transfer; reuse frozen ReLU arithmetic."""
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
from wikitext2_training import ROOT, configure, load_stream, tensor_hash, forward_device, evaluate
from verify_wikitext2_baseline import model_new as template_model
from position_distractors import digest

SEEDS = (1861, 1877, 1901)
ARMS = ('ln', 'rms')
PROTOCOL = 'docs/decisions/0005-wikitext2-normalization.md'


def model_new(arm, vocabulary, seed):
    assert arm in ARMS
    torch.manual_seed(seed)
    model = template_model('mqa', vocabulary).float()
    def layer():
        return nn.LayerNorm(16, eps=1e-5, bias=False) if arm == 'ln' else nn.RMSNorm(16, eps=1e-5)
    for block in model.blocks:
        block.norm1, block.norm2 = layer(), layer()
    model.final_norm = layer()
    return model


def common_hash(model):
    return tensor_hash(model)


def provenance():
    earlier = json.loads((ROOT/'results/wikitext2_training.json').read_text())
    sources = dict(earlier['provenance'])
    assert all(digest(ROOT/name)==h for name,h in sources.items())
    for name in (PROTOCOL,'experiments/rmsnorm_math.py','experiments/audit_rmsnorm_math.py','experiments/wikitext2_norm.py',
                 'experiments/verify_wikitext2_norm.py','experiments/audit_wikitext2_norm.py'):
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
    return dict(adopt_rmsnorm=all(checks.values()),checks=checks,geometric_mean_ratio=mean,paired_ratios=ratios,metric='final_validation')


def main():
    configure()
    assert torch.cuda.is_available()
    rp = ROOT/'results/wikitext2_norm.json'
    dp = ROOT/'results/wikitext2_norm_scores.pt'
    directory = ROOT/'results/wikitext2_norm_checkpoints'
    directory.mkdir(exist_ok=True)
    prov = provenance()
    preflight = ROOT/'results/wikitext2_norm_verification.json'
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
