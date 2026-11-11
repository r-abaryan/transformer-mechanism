"""Matched activation substitutions in a frozen MQA reader's last block."""
import csv
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from state_tracking import ROOT, model_new, forward, tensors, save_json, answer
from audit_state_tracking import oracle
from audit_kv_cache_equivalence import norm, rotation
from positional_comparison import rotate
from position_distractors import digest

DIRECTORY=ROOT/'results/state_initial_paths'
MODES=('sham','residual','attention','both','Q_only','KV_initial','KV_other','QKV_all')


def trace(model,ids,reference=False):
    # Older MHA patch helpers omit this model's grouped K/V and RoPE convention.
    x=model.embedding.weight[ids] if reference else model.embedding(ids)
    n=ids.shape[1];positions=torch.arange(n)+4;mask=torch.ones(n,n,dtype=torch.bool).triu(1)
    cache=[]
    for block in model.blocks:
        incoming=x;z=norm(x,block.norm1) if reference else block.norm1(x);m=block.attn
        if reference:parts=[z@getattr(m,t+'_weight').T+getattr(m,t+'_bias') for t in ('q','k','v')]
        else:parts=[F.linear(z,getattr(m,t+'_weight'),getattr(m,t+'_bias')) for t in ('q','k','v')]
        q=parts[0].reshape(len(ids),n,2,8).transpose(1,2)
        k,v=[t.reshape(len(ids),n,model.groups,8).transpose(1,2) for t in parts[1:]]
        q,k=(rotation(q,positions),rotation(k.expand(-1,2,-1,-1),positions)) if reference else (rotate(q,positions),rotate(k,positions))
        if reference:
            kk=k;vv=v.expand(-1,2,-1,-1)
            h=F.scaled_dot_product_attention(q,kk,vv,is_causal=True).transpose(1,2).reshape(len(ids),n,16)
            attention=h@m.out_proj.weight.T+m.out_proj.bias
            weights=(q@kk.transpose(-1,-2)/(8**.5)).masked_fill(mask,-torch.inf).softmax(-1)
            k=k[:,:model.groups]
        else:
            kk,vv=[t.repeat_interleave(2//model.groups,dim=1) for t in (k,v)]
            weights=(q@kk.transpose(-1,-2)/(8**.5)).masked_fill(mask,-torch.inf).softmax(-1)
            h=(weights@vv).transpose(1,2).reshape(len(ids),n,16);attention=m.out_proj(h)
        x=x+attention
        if reference:
            z=norm(x,block.norm2)@block.linear1.weight.T+block.linear1.bias
            x=x+z.clamp_min(0)@block.linear2.weight.T+block.linear2.bias
        else:x=x+block.linear2(F.relu(block.linear1(block.norm2(x))))
        cache.append(dict(incoming=incoming,attention=attention,q=q,k=k,v=v,weights=weights,outgoing=x))
    scores=(norm(x,model.final_norm)@model.output.weight.T if reference else model.output(model.final_norm(x)))[:,-1]
    return scores,cache


def patched(model,cache,donor,mode,reference=False):
    b=model.blocks[-1];m=b.attn;c=cache[-1]
    residual=c['incoming'][:,-1]
    if mode in ('residual','both'):residual=residual[donor]
    attention=c['attention'][:,-1]
    if mode in ('attention','both'):attention=attention[donor]
    if mode in ('Q_only','KV_initial','KV_other','QKV_all'):
        q=c['q'][:,:,-1:].clone();k=c['k'].clone();v=c['v'].clone()
        if mode in ('Q_only','QKV_all'):q=q[donor]
        if mode=='KV_initial':k[:,:,0]=k[donor,:,0];v[:,:,0]=v[donor,:,0]
        if mode=='KV_other':k[:,:,1:]=k[donor,:,1:];v[:,:,1:]=v[donor,:,1:]
        if mode=='QKV_all':k=k[donor];v=v[donor]
        if reference:
            k,v=[t.expand(-1,2,-1,-1) for t in (k,v)]
            # Last query has the entire prefix available; is_causal=True here is wrong.
            h=F.scaled_dot_product_attention(q,k,v,is_causal=False).transpose(1,2).reshape(len(q),16)
            attention=h@m.out_proj.weight.T+m.out_proj.bias
        else:
            k,v=[t.repeat_interleave(2//model.groups,dim=1) for t in (k,v)]
            weights=(q@k.transpose(-1,-2)/(8**.5)).softmax(-1)
            h=(weights@v).transpose(1,2).reshape(len(q),16);attention=m.out_proj(h)
    x=residual+attention
    if reference:
        z=norm(x,b.norm2)@b.linear1.weight.T+b.linear1.bias
        x=x+z.clamp_min(0)@b.linear2.weight.T+b.linear2.bias
        return norm(x,model.final_norm)@model.output.weight.T
    x=x+b.linear2(F.relu(b.linear1(b.norm2(x))))
    return model.output(model.final_norm(x))


def metrics(original,donor,patched_scores,y):
    op,dp,pp=[v.argmax(-1) for v in (original,donor,patched_scores)]
    changed=op!=dp;before=op==y;after=pp==y
    gaps=[z[:,1]-z[:,0] for z in (original,donor,patched_scores)]
    denominator=float((gaps[0]-gaps[1]).abs().sum());error=float((gaps[2]-gaps[1]).abs().sum())
    return dict(count=len(y),accuracy=float(after.double().mean()),rescued=int((~before&after).sum()),harmed=int((before&~after).sum()),
                class_changing_inputs=int(changed.sum()),donor_agreement_changed=float((pp[changed]==dp[changed]).double().mean()) if bool(changed.any()) else None,
                mean_abs_donor_gap_error=error/len(y),original_abs_gap_sum=denominator,recovery=1-error/denominator if denominator else None)


def scalar_metrics(original,donor,scores,labels):
    correct=rescued=harmed=changes=agree=0;denominator=error=0.
    for a,b,c,y in zip(original.tolist(),donor.tolist(),scores.tolist(),labels.tolist()):
        op,dp,pp=[int(v[1]>v[0]) for v in (a,b,c)]
        before=op==y;after=pp==y;correct+=after;rescued+=not before and after;harmed+=before and not after
        changes+=op!=dp;agree+=op!=dp and pp==dp
        denominator+=abs((a[1]-a[0])-(b[1]-b[0]));error+=abs((c[1]-c[0])-(b[1]-b[0]))
    return dict(count=len(labels),accuracy=correct/len(labels),rescued=rescued,harmed=harmed,class_changing_inputs=changes,
                donor_agreement_changed=agree/changes if changes else None,mean_abs_donor_gap_error=error/len(labels),
                original_abs_gap_sum=denominator,recovery=1-error/denominator if denominator else None)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert not DIRECTORY.exists()
    protocol=ROOT/'docs/protocols/state-initial-paths.md';parent=ROOT/'results/state_reset_exposure'
    old=json.loads((parent/'report.json').read_text(encoding='utf-8'));pools=json.loads((parent/'data.json').read_text(encoding='utf-8'))['evaluation']
    runs=[v for v in old['runs'] if v['arm']=='independent'];assert {v['seed'] for v in runs}=={3011,3019,3023}
    paths=[protocol,Path(__file__),parent/'report.json',parent/'data.json',parent/'audit.json']+[ROOT/p for p in old['source_dependencies']]
    paths+=[ROOT/r[k] for r in runs for k in ('checkpoint','scores')]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    assert digest(parent/'data.json')==old['data_sha256'] and all(digest(ROOT/p)==h for p,h in old['source_dependencies'].items())
    records=[];attention_rows=[];scores={};errors=[];evaluations=0;max_scalar=0.
    for run in runs:
        assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256'] and digest(ROOT/run['scores'])==run['scores_sha256']
        model=model_new().eval();model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        archived=torch.load(ROOT/run['scores'],weights_only=True);scores[run['seed']]={}
        for prefix in ('train/12','test/12'):
            rows=pools[prefix+'/long_tail_agree']+pools[prefix+'/long_tail_disagree'];x,y=tensors(rows)
            half=len(rows)//2;donor=(torch.arange(len(rows))+half)%len(rows)
            assert torch.equal(x[:,1:],x[donor,1:]) and torch.equal(y,y[donor]) and torch.equal(x[:,0]+x[donor,0],torch.full_like(x[:,0],9))
            assert all(oracle(ids)==yy==answer(ids[0]-4,ids[1:-1]) for ids,yy in rows)
            with torch.no_grad():
                original,c=trace(model,x);reference,rc=trace(model,x,True);native=forward(model,x)[:,-1]
                cache_error=max(float((a[k]-b[k]).abs().max()) for a,b in zip(c,rc) for k in ('incoming','attention','q','k','v','weights','outgoing'))
                archive=torch.cat([archived[prefix+'/long_tail_agree'],archived[prefix+'/long_tail_disagree']])
                be=max(float((original-reference).abs().max()),float((original-native).abs().max()),float((original-archive).abs().max()),cache_error)
                assert be<=1e-10 and torch.equal(original.argmax(-1),reference.argmax(-1));evaluations+=len(x)
                saved=dict(original=original,donor=donor);scores[run['seed']][prefix]=saved
                conditions={}
                for mode in MODES:
                    z=patched(model,c,donor,mode);rz=patched(model,rc,donor,mode,True)
                    oe=float((z-rz).abs().max());assert oe<=1e-10 and torch.equal(z.argmax(-1),rz.argmax(-1))
                    saved[mode]=z;conditions[mode]=z;evaluations+=len(x)
                    val=metrics(original,original[donor],z,y);scalar=scalar_metrics(reference,reference[donor],rz,y)
                    se=max(abs(val[k]-scalar[k]) for k in val if val[k] is not None);assert se<=1e-10;max_scalar=max(max_scalar,se)
                    records.append(dict(seed=run['seed'],pool=prefix,condition=mode,**val))
                sham=float((conditions['sham']-original).abs().max());both=float((conditions['both']-original[donor]).abs().max())
                qkv=float((conditions['QKV_all']-conditions['attention']).abs().max())
                negative=float((conditions['residual']-original[donor]).abs().max())
                assert max(sham,both,qkv)<=1e-10 and negative>1e-6
                errors.append(dict(seed=run['seed'],pool=prefix,baseline_cache_error=be,sham_error=sham,both_donor_error=both,
                                   qkv_attention_error=qkv,omitted_attention_negative_effect=negative))
                for layer,cache in enumerate(c,1):
                    w=cache['weights'][:,:,-1]
                    for h in (0,1):
                        attention_rows.append(dict(seed=run['seed'],pool=prefix,block=layer,head=h,
                                                   initial=float(w[:,h,0].mean()),reset=float(w[:,h,1].mean()),
                                                   suffix=float(w[:,h,2:-1].sum(-1).mean()),query=float(w[:,h,-1].mean())))
            print(f"Verified path substitutions: {run['seed']} {prefix}",flush=True)
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    DIRECTORY.mkdir();torch.save(scores,DIRECTORY/'scores.pt')
    result=dict(scope='matched same-model activation substitutions; saved training/test inputs',preserved_sha256=hashes,
                metrics=records,attention_inspection=attention_rows,controls=errors,scores_sha256=digest(DIRECTORY/'scores.pt'),
                verification=dict(passed=True,evaluations=evaluations,argmax_changes=0,max_scalar_error=max_scalar,control_cells=len(errors)))
    save_json(DIRECTORY/'report.json',result)
    for name,rows in [('metrics',records),('attention',attention_rows)]:
        with (DIRECTORY/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result['verification']),flush=True)


if __name__=='__main__':main()
