"""Frozen paired external-corpus training, preserving earlier experiment sources."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import array
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from verify_wikitext2_baseline import model_new
from mqa_decoding import rotate_fp
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(1723,1741,1753)


def configure():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def tensor_hash(model,kind='all'):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        kv=name.endswith(('.k_weight','.k_bias','.v_weight','.v_bias'))
        if kind=='common' and kv:continue
        if kind=='kv' and not kv:continue
        if kind=='kv':t=t[:8]
        h.update(json.dumps([name,list(t.shape),'float32']).encode())
        a=array.array('f',t.detach().cpu().float().flatten().tolist())
        assert a.itemsize==4
        import sys
        if sys.byteorder!='little':a.byteswap()
        h.update(a.tobytes())
    return h.hexdigest()


def load_stream(split):
    r=json.loads((ROOT/'results/wikitext2_encoding.json').read_text());s=r['splits'][split]
    path=ROOT/s['path'];assert digest(path)==s['sha256']
    return torch.frombuffer(bytearray(path.read_bytes()),dtype=torch.int32).long()


def angles(n,device,dtype):
    a=(torch.arange(n,device=device,dtype=torch.float64)+4)[:,None]*torch.tensor([1.,.1,.01,.001],device=device,dtype=torch.float64)
    return a.cos().to(dtype),a.sin().to(dtype)


def forward_device(model,ids):
    n=ids.shape[1];x=model.embedding(ids);c,s=angles(n,ids.device,x.dtype)
    mask=torch.ones(n,n,device=ids.device,dtype=torch.bool).triu(1)
    for b in model.blocks:
        z=b.norm1(x);m=b.attn
        q,k,v=[F.linear(z,getattr(m,t+'_weight'),getattr(m,t+'_bias')) for t in ('q','k','v')]
        q=q.reshape(len(ids),n,2,8).transpose(1,2)
        k,v=[t.reshape(len(ids),n,model.groups,8).transpose(1,2) for t in (k,v)]
        q,k=rotate_fp(q,c,s),rotate_fp(k,c,s)
        k,v=[t.repeat_interleave(2//model.groups,dim=1) for t in (k,v)]
        a=(q@k.transpose(-1,-2)/(8**.5)).masked_fill(mask,-torch.inf).softmax(-1)
        y=(a@v).transpose(1,2).reshape(len(ids),n,16)
        x=x+m.out_proj(y);x=x+b.linear2(F.relu(b.linear1(b.norm2(x))))
    return model.output(model.final_norm(x))


def evaluate(model,stream,retain=False):
    n=len(stream)-1;complete,tail=divmod(n,64);losses=[];predictions=[];fixtures=[]
    device=next(model.parameters()).device
    with torch.no_grad():
        for first in range(0,complete,8):
            starts=torch.arange(first,min(first+8,complete))*64
            positions=starts[:,None]+torch.arange(64)[None,:]
            x,y=stream[positions].to(device),stream[positions+1].to(device)
            logits=forward_device(model,x)
            loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').cpu();pred=logits.argmax(-1).flatten().cpu()
            losses.append(loss);predictions.append(pred)
            if retain and first==0:fixtures.append(dict(start=0,logits=logits[:2].cpu()))
        if tail:
            start=complete*64;x=stream[start:start+tail][None,:].to(device);y=stream[start+1:start+tail+1].to(device)
            logits=forward_device(model,x);losses.append(F.cross_entropy(logits[0],y,reduction='none').cpu())
            predictions.append(logits[0].argmax(-1).cpu())
            if retain:fixtures.append(dict(start=start,logits=logits.cpu()))
    losses=torch.cat(losses);predictions=torch.cat(predictions)
    assert len(losses)==len(predictions)==n and bool(torch.isfinite(losses).all())
    mean=losses.double().mean().item()
    metrics=dict(targets=n,loss=mean,perplexity=math.exp(mean),accuracy=(predictions==stream[1:]).double().mean().item())
    return metrics,dict(losses=losses,predictions=predictions,fixtures=fixtures) if retain else None


def provenance():
    old=json.loads((ROOT/'results/wikitext2_baseline_preflight.json').read_text())
    sources={f'experiments/{n}':digest(ROOT/'experiments'/n) for n in old['sources']}
    assert all(sources['experiments/'+n]==h for n,h in old['sources'].items())
    for p in ('experiments/wikitext2_training.py','experiments/verify_wikitext2_gpu.py','experiments/audit_wikitext2_training.py',
              'experiments/prepare_wikitext2.py','experiments/audit_wikitext2_data.py','experiments/mqa_decoding.py',
              'docs/wikitext2-execution-protocol.md','docs/decisions/0003-wikitext2-matched-baseline.md'):
        sources[p]=digest(ROOT/p)
    return sources


def decision(runs,unigram):
    checks=dict(completed=len(runs)==6 and all(r['status']=='trained' and r['step']==6000 for r in runs))
    checks['learned_beyond_unigram']=len(runs)==6 and all('test' in r and unigram-r['test']['loss']>=.05 for r in runs)
    gains=[]
    for seed in SEEDS:
        p=[next((r for r in runs if r['seed']==seed and r['group']==g),{}) for g in ('mha','mqa')]
        if all('test' in r for r in p):gains.append(p[1]['test']['loss']-p[0]['test']['loss'])
    geometric=math.exp(sum(gains)/3) if len(gains)==3 else None
    ratios=[math.exp(v) for v in gains]
    checks['mean_ratio5']=geometric is not None and geometric<=1.05
    checks['seed_ratio10']=len(ratios)==3 and max(ratios)<=1.10
    return dict(accept_mqa=all(checks.values()),checks=checks,geometric_mean_ratio=geometric,paired_ratios=ratios)


def main():
    configure();assert torch.cuda.is_available()
    rp=ROOT/'results/wikitext2_training.json';dp=ROOT/'results/wikitext2_training_scores.pt'
    directory=ROOT/'results/wikitext2_training_checkpoints';directory.mkdir(exist_ok=True)
    data_report=json.loads((ROOT/'results/wikitext2_encoding.json').read_text());vocabulary=data_report['vocabulary'];prov=provenance()
    verification=json.loads((ROOT/'results/wikitext2_gpu_verification.json').read_text());assert verification['passed'] and verification['provenance']==prov
    data_audit=json.loads((ROOT/'results/wikitext2_data_audit.json').read_text());assert data_audit['passed'] and data_audit['encoding_sha256']==digest(ROOT/'results/wikitext2_encoding.json')
    train,validation=load_stream('train'),load_stream('validation')
    if rp.exists():
        r=json.loads(rp.read_text());assert r['provenance']==prov and r['encoding_sha256']==digest(ROOT/'results/wikitext2_encoding.json')
        if r['phase']=='complete':print('Already complete.');return
    else:
        assert not dp.exists();r=dict(phase='training',provenance=prov,encoding_sha256=digest(ROOT/'results/wikitext2_encoding.json'),
            data_audit_sha256=digest(ROOT/'results/wikitext2_data_audit.json'),gpu_verification_sha256=digest(ROOT/'results/wikitext2_gpu_verification.json'),
            torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),platform=platform.platform(),runs=[])
    def save():
        tmp=rp.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(rp)
    save()
    train_gpu=train.cuda();offsets=torch.arange(64,device='cuda')[None,:]
    for seed in SEEDS:
        for group in ('mha','mqa'):
            run=next((v for v in r['runs'] if (v['seed'],v['group'])==(seed,group)),None);cp=directory/f'{group}_{seed}.pt'
            if run and run['status'] in ('trained','failed'):assert digest(cp)==run['checkpoint_sha256'];continue
            torch.manual_seed(seed);model=model_new(group,vocabulary).float()
            initial=tensor_hash(model);common=tensor_hash(model,'common');kv=tensor_hash(model,'kv')
            model.cuda();opt=torch.optim.Adam(model.parameters(),lr=.001)
            generator=torch.Generator().manual_seed(seed+1000);start=0
            if cp.exists():
                state=torch.load(cp,map_location='cpu',weights_only=True);assert state['provenance']==prov
                model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);generator.set_state(state['generator'])
                run=state['run'];start=run['step'];r['runs']=[v for v in r['runs'] if (v['seed'],v['group'])!=(seed,group)];r['runs'].append(run)
            else:
                run=dict(seed=seed,group=group,parameters=sum(p.numel() for p in model.parameters()),initial_sha256=initial,
                         common_initial_sha256=common,retained_kv_initial_sha256=kv,status='running',step=0,tokens=0,targets=0,curve=[],elapsed_seconds=0.)
                run['curve'].append(dict(step=0,validation=evaluate(model,validation)[0]));r['runs'].append(run)
            h=hashlib.sha256();replay=torch.Generator().manual_seed(seed+1000)
            for _ in range(start):h.update(json.dumps(torch.randint(len(train)-64,(8,),generator=replay).tolist()).encode())
            tick=time.perf_counter();window=[]
            for step in range(start+1,6001):
                starts=torch.randint(len(train)-64,(8,),generator=generator);h.update(json.dumps(starts.tolist()).encode())
                positions=starts.cuda()[:,None]+offsets;x,y=train_gpu[positions],train_gpu[positions+1]
                opt.zero_grad();logits=forward_device(model,x);loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
                finite=bool(torch.isfinite(loss))
                if finite:
                    loss.backward();finite=all(p.grad is not None for p in model.parameters()) and bool(torch.stack([p.grad.isfinite().all() for p in model.parameters()]).all())
                if finite:
                    opt.step();finite=bool(torch.stack([p.isfinite().all() for p in model.parameters()]).all());window.append(loss.detach())
                if not finite:run.update(status='failed',failure_step=step)
                run['tokens']+=512;run['targets']+=512
                if step%1000==0 or not finite:
                    point=dict(step=step,finite_training=finite,training_window_mean_loss=torch.stack(window).mean().item() if window else None,
                               validation=evaluate(model,validation)[0]);window=[];run['curve'].append(point)
                    run.update(step=step,stream_sha256=h.hexdigest())
                    if step==6000 and finite:run['status']='trained'
                    torch.cuda.synchronize();run['elapsed_seconds']+=time.perf_counter()-tick
                    state=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),generator=generator.get_state(),run=dict(run),provenance=prov)
                    tmp=cp.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(cp)
                    run.update(checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=digest(cp));save();tick=time.perf_counter()
                    print(json.dumps(dict(seed=seed,group=group,step=step,validation_ppl=point['validation']['perplexity'])),flush=True)
                if not finite:break
            del model,opt
    assert len(r['runs'])==6;r['phase']='all_attempts_finished_before_test';save()
    test=load_stream('test');counts=torch.bincount(train,minlength=vocabulary).double()
    baseline=-((counts+1)/(len(train)+vocabulary)).log()[test[1:]]
    r['unigram_test']=dict(targets=len(test)-1,loss=baseline.mean().item(),perplexity=math.exp(baseline.mean().item()))
    details=[]
    for run in r['runs']:
        if run['status']!='trained':continue
        cp=ROOT/run['checkpoint'];assert digest(cp)==run['checkpoint_sha256']
        model=model_new(run['group'],vocabulary).float().cuda().eval();model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model'])
        run['test'],detail=evaluate(model,test,True);details.append(dict(seed=run['seed'],group=run['group'],**detail));del model;save()
    assert not dp.exists();torch.save(details,dp)
    r.update(phase='complete',scores_sha256=digest(dp),decision=decision(r['runs'],r['unigram_test']['loss']));save();print(json.dumps(r['decision']))


if __name__=='__main__':main()
