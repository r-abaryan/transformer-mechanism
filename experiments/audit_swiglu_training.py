"""Independent SDPA/manual FFN reference and all-token validation/test audit."""
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from audit_kv_cache_equivalence import norm, rotation
from swiglu_training import ROOT,SEEDS,ARMS,model_new,common_hash,load_stream,tensor_hash,provenance
from position_distractors import digest


def independent(model,ids,wrong_product=False):
    x=model.embedding.weight[ids]
    n=ids.shape[1];positions=torch.arange(n)+4
    for block in model.blocks:
        z=norm(x,block.norm1);m=block.attn
        q,k,v=[z@getattr(m,t+'_weight').T+getattr(m,t+'_bias') for t in ('q','k','v')]
        q=q.reshape(len(ids),n,2,8).permute(0,2,1,3)
        k,v=[t.reshape(len(ids),n,model.groups,8).permute(0,2,1,3).expand(len(ids),2,n,8) for t in (k,v)]
        q,k=rotation(q,positions),rotation(k,positions)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True).permute(0,2,1,3).reshape(len(ids),n,16)
        x=x+y@m.out_proj.weight.T+m.out_proj.bias
        z=norm(x,block.norm2)
        a=z@block.linear1.weight.T
        if block.linear1.bias is not None:a=a+block.linear1.bias
        if hasattr(block,'content'):
            u=z@block.content.weight.T
            hidden=a/(1+(-a).exp())
            if not wrong_product:hidden=hidden*u
        else:
            hidden=a.clamp_min(0)
        y=hidden@block.linear2.weight.T
        if block.linear2.bias is not None:y=y+block.linear2.bias
        x=x+y
    return norm(x,model.final_norm)@model.output.weight.T


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rp=ROOT/'results/swiglu_training.json';r=json.loads(rp.read_text())
    assert r['phase']=='complete' and r['provenance']==provenance()
    assert digest(ROOT/'results/wikitext2_encoding.json')==r['encoding_sha256']
    assert digest(ROOT/'results/swiglu_training_verification.json')==r['verification_sha256']
    assert {(run['seed'],run['arm']) for run in r['runs']}=={(s,g) for s in SEEDS for g in ARMS}
    assert hashlib.sha256(json.dumps(r['decision'],sort_keys=True).encode()).hexdigest()==r['validation_decision_sha256']
    vocabulary=json.loads((ROOT/'results/wikitext2_encoding.json').read_text())['vocabulary']
    train=load_stream('train');streams={split:load_stream(split) for split in ('validation','test')}
    counts=[0]*vocabulary
    for token in train.tolist():counts[token]+=1
    logs=[-math.log((n+1)/(len(train)+vocabulary)) for n in counts]
    baselines={split:sum(logs[t] for t in stream[1:].tolist())/(len(stream)-1) for split,stream in streams.items()}
    for split,loss in baselines.items():
        assert abs(loss-r['unigram_'+split]['loss'])<=1e-10
    score_path=ROOT/'results/swiglu_training_scores.pt'
    assert digest(score_path)==r['scores_sha256']
    saved=torch.load(score_path,map_location='cpu',weights_only=True)
    for seed in SEEDS:
        pair=[run for run in r['runs'] if run['seed']==seed]
        for key in ('common_initial_sha256','stream_sha256'):
            assert pair[0][key]==pair[1][key]
    out=ROOT/'results/swiglu_training_audit.json';assert not out.exists()
    partial=ROOT/'results/swiglu_training_audit_progress.json'
    records=[]
    if partial.exists():
        progress=json.loads(partial.read_text())
        assert progress['report_sha256']==digest(rp) and progress['source_sha256']==digest(Path(__file__))
        records=progress['records']
    for run in r['runs']:
        base=model_new(run['arm'],vocabulary,run['seed'])
        assert tensor_hash(base)==run['initial_sha256'] and common_hash(base)==run['common_initial_sha256']
        assert sum(p.numel() for p in base.parameters())==run['parameters']==32*vocabulary+3904
        assert all(sum(p.numel() for m in (b.linear1,b.linear2) for p in m.parameters())+(sum(p.numel() for p in b.content.parameters()) if hasattr(b,'content') else 0)==1056 for b in base.blocks)
        cp=ROOT/run['checkpoint'];assert digest(cp)==run['checkpoint_sha256']
        state=torch.load(cp,map_location='cpu',weights_only=True);assert state['provenance']==r['provenance']
        generator=torch.Generator().manual_seed(run['seed']+1000);h=hashlib.sha256()
        for _ in range(run['step']):
            h.update(json.dumps(torch.randint(len(train)-64,(8,),generator=generator).tolist()).encode())
        assert h.hexdigest()==run['stream_sha256'] and torch.equal(generator.get_state(),state['generator'])
        if run['status']!='trained':continue
        assert run['step']==6000 and run['targets']==3072000
        assert all(p['step'].item()==6000 for p in state['optimizer']['state'].values())
        assert len(state['optimizer']['state'])==len(list(base.parameters()))
        base.load_state_dict(state['model']);model=base.double().eval()
        detail=next(d for d in saved if (d['seed'],d['arm'])==(run['seed'],run['arm']))
        for split,stream in streams.items():
            if any((d['seed'],d['arm'],d['split'])==(run['seed'],run['arm'],split) for d in records):continue
            stored=detail[split]
            summary=run['final_validation' if split=='validation' else 'test']
            n=len(stream)-1
            losses,predictions=[],[];fixture_error=0.;covered=0
            # Independently enumerate consecutive chunks; do not call producer evaluation.
            with torch.no_grad():
                for start in range(0,n,8*64):
                    counts_in_batch=[min(64,n-s) for s in range(start,min(start+8*64,n),64)]
                    full=sum(c==64 for c in counts_in_batch)
                    if full:
                        offsets=torch.arange(full)[:,None]*64+start+torch.arange(64)[None,:]
                        x,y=stream[offsets],stream[offsets+1]
                        scores=independent(model,x)
                        losses.append((scores.logsumexp(-1)-scores.gather(-1,y[...,None]).squeeze(-1)).flatten())
                        predictions.append(scores.argmax(-1).flatten());covered+=full*64
                        if start==0:
                            fixture=next(f for f in stored['fixtures'] if f['start']==0)
                            fixture_error=(scores[:2]-fixture['logits'].double()).abs().max().item()
                    if counts_in_batch[-1]!=64:
                        tail=counts_in_batch[-1];tailstart=start+full*64
                        x=stream[tailstart:tailstart+tail][None,:];y=stream[tailstart+1:tailstart+tail+1]
                        scores=independent(model,x)
                        losses.append(scores[0].logsumexp(-1)-scores[0,torch.arange(tail),y])
                        predictions.append(scores[0].argmax(-1));covered+=tail
                        fixture=next(f for f in stored['fixtures'] if f['start']==tailstart)
                        fixture_error=max(fixture_error,(scores-fixture['logits'].double()).abs().max().item())
            losses,predictions=torch.cat(losses),torch.cat(predictions)
            assert covered==n==len(losses)==len(stored['losses'])==summary['targets']
            maximum=(losses-stored['losses'].double()).abs().max().item()
            mean=losses.mean().item()
            assert maximum<=1e-4 and abs(mean-summary['loss'])<=1e-5 and fixture_error<=1e-4
            assert abs(math.exp(mean)/summary['perplexity']-1)<=2e-5
            assert abs((stored['predictions']==stream[1:]).double().mean().item()-summary['accuracy'])<=1e-10
            record=dict(seed=run['seed'],arm=run['arm'],split=split,targets=n,loss=mean,
                        max_token_loss_error=maximum,mean_loss_error=abs(mean-summary['loss']),fixture_logit_error=fixture_error,
                        argmax_differences=int((predictions!=stored['predictions']).sum()))
            records.append(record)
            partial.write_text(json.dumps(dict(report_sha256=digest(rp),source_sha256=digest(Path(__file__)),records=records),indent=2)+'\n')
            print(json.dumps(record),flush=True)
    assert len(records)==2*sum(run['status']=='trained' for run in r['runs'])
    differences=[]
    for seed in SEEDS:
        pair=[next((d for d in records if (d['seed'],d['arm'],d['split'])==(seed,arm,'validation')),None) for arm in ARMS]
        if all(p is not None for p in pair):differences.append(pair[1]['loss']-pair[0]['loss'])
    mean=math.exp(sum(differences)/3) if len(differences)==3 else None
    checks=dict(completed=len(r['runs'])==6 and all(run['status']=='trained' and run['step']==6000 for run in r['runs']))
    checks['learning_sanity']=sum(d['split']=='validation' for d in records)==6 and all(baselines['validation']-d['loss']>=.05 for d in records if d['split']=='validation')
    checks['mean_improvement2']=mean is not None and mean<=.98
    checks['no_worse_seed']=len(differences)==3 and max(differences)<=0.
    assert checks==r['decision']['checks'] and all(checks.values())==r['decision']['adopt_swiglu']
    if mean is not None:assert abs(mean-r['decision']['geometric_mean_ratio'])<=2e-5
    result=dict(passed=True,records=records,targets_checked=sum(d['targets'] for d in records),
                max_token_loss_error=max((d['max_token_loss_error'] for d in records),default=0.),
                independent_validation_checks=checks,report_sha256=digest(rp),source_sha256=digest(Path(__file__)))
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cells=len(records),targets_checked=result['targets_checked'])),flush=True)


if __name__=='__main__':main()
