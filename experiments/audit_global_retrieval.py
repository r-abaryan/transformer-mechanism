"""Independent global-relay equations, family controls and ablation audit."""
import hashlib
import json
import torch
from torch.nn import functional as F
from audit_kv_cache_equivalence import norm, rotation
from global_retrieval import ROOT, SEEDS, DISTANCES, ARMS, PREFIX, forward, model_new, original_forward, tensors, batch_indices
from mqa_training import hash_state
from position_distractors import digest
from recurrent_retrieval import write


def independent(model,ids,arm):
    x=model.embedding.weight[ids];n=ids.shape[1];positions=torch.arange(n)+4
    for b in model.blocks:
        z=norm(x,b.norm1);m=b.attn
        q,k,v=[z@getattr(m,t+'_weight').T+getattr(m,t+'_bias') for t in ('q','k','v')]
        q=q.reshape(len(ids),n,2,8).permute(0,2,1,3)
        k,v=[u.reshape(len(ids),n,1,8).permute(0,2,1,3).expand(len(ids),2,n,8) for u in (k,v)]
        q,k=rotation(q,positions),rotation(k,positions)
        outputs=[]
        for i in range(n):
            ix=[j for j in range(i+1) if arm=='full' or i-j<=2 or arm=='global' and (i==n//2 or j==n//2)]
            s=(q[:,:,i,None,:]*k[:,:,ix,:]).sum(-1)/(8**.5)
            e=(s-s.max(-1,keepdim=True).values).exp();a=e/e.sum(-1,keepdim=True)
            outputs.append((a[:,:,:,None]*v[:,:,ix,:]).sum(-2))
        h=torch.stack(outputs,2).permute(0,2,1,3).reshape(len(ids),n,16)
        x=x+h@m.out_proj.weight.T+m.out_proj.bias
        y=norm(x,b.norm2)@b.linear1.weight.T+b.linear1.bias
        x=x+y.clamp_min(0)@b.linear2.weight.T+b.linear2.bias
    return norm(x,model.final_norm)@model.output.weight.T


def audit_data(payload):
    assert payload['generator_seed']==2026100205
    seen=set();bases=set();baselines={};count=0;wrong=0
    for name,pool in payload['pools'].items():
        n=int(name.split('_')[-1]);families=128 if name=='train_17' else 32
        assert len(pool['bases'])==families and len(pool['rows'])==families*30
        for base in pool['bases']:
            assert len(base)==n-1 and base[n//2]==11 and all(0<=v<11 for i,v in enumerate(base) if i!=n//2)
            signature=(n,tuple(base));assert signature not in bases;bases.add(signature)
        bygroup={d:[] for d in DISTANCES}
        for row in pool['rows']:
            x,y,d,f=row['ids'],row['label'],row['distance'],row['family']
            assert len(x)==n and x[-1]==18 and tuple(x) not in seen
            seen.add(tuple(x));marked=[(i,v) for i,v in enumerate(x) if 12<=v<18]
            assert len(marked)==1
            pos,answer=marked[0];assert answer==y and n-1-pos==d
            base=pool['bases'][f]
            assert all(v==base[i] for i,v in enumerate(x[:-1]) if i!=pos)
            bygroup[d].append((x,y,f));count+=1
            wrong+=int(12+(y-12+1)%6!=answer)
        baselines[name]={}
        for d,rows in bygroup.items():
            assert len(rows)==families*6
            for f in range(families):
                variants=[(x,y) for x,y,ff in rows if ff==f]
                assert [y for x,y in variants]==list(range(12,18))
                if d>4:assert len({tuple(x[-5:]) for x,y in variants})==1
            baselines[name][str(d)]=dict(oracle=1.,constant12=sum(y==12 for x,y,f in rows)/len(rows),
                last_context=sum(x[-2]==y for x,y,f in rows)/len(rows),uniform_six_expected=1/6)
    assert wrong==count
    return dict(passed=True,examples=count,families=len(bases),shifted_label_errors_detected=wrong,baselines=baselines)


def preflight(data):
    cells=[]
    for seed in SEEDS:
        torch.manual_seed(seed);m=model_new('mqa');x,y,d=data['train_17']
        ids=torch.cat([x[d==v][:6] for v in DISTANCES]);labels=torch.cat([y[d==v][:6] for v in DISTANCES])
        assert torch.equal(forward(m,ids,'full'),original_forward(m,ids))
        for arm in ARMS:
            a=forward(m,ids,arm);b=independent(m,ids,arm)
            ga=torch.autograd.grad(F.cross_entropy(a[:,-1],labels),tuple(m.parameters()))
            gb=torch.autograd.grad(F.cross_entropy(b[:,-1],labels),tuple(m.parameters()))
            oe=float((a-b).abs().max());ge=max(float((u-v).abs().max()) for u,v in zip(ga,gb))
            assert max(oe,ge)<1e-10
            cells.append(dict(seed=seed,arm=arm,logit_error=oe,gradient_error=ge))
        with torch.no_grad():
            family=x[d==16][:6]
            local=forward(m,family,'local')[:,-1];wrong=forward(m,family,'full')[:,-1]
            invariant=float((local-local[:1]).abs().max());effect=float((wrong-wrong[:1]).abs().max())
            assert invariant<=1e-10 and effect>1e-6
            cells[-1].update(unreachable_change=invariant,omitted_window_detected=effect)
        with torch.no_grad():
            n=ids.shape[1];sample=ids[:1].clone();modified=sample.clone();modified[:,-1]=0
            safe=forward(m,sample,'global');altered=forward(m,modified,'global')
            assert float((safe[:,:-1]-altered[:,:-1]).abs().max())<=1e-10
            family=x[d==5][:6];g=forward(m,family,'global')[:,-1]
            assert float((g-g[:1]).abs().max())<=1e-10
    return dict(passed=True,cells=cells)


def metrics(scores,y,d):
    out={};pred=scores.argmax(-1)
    logprobs=scores-scores.logsumexp(-1,keepdim=True)
    for g in ('all','near','after_global','before_global',*map(str,DISTANCES)):
        ids=[i for i,v in enumerate(d.tolist()) if g=='all' or g=='near' and v<=4 or g=='after_global' and v==5 or g=='before_global' and v>=12 or g==str(v)]
        count=len(ids);correct=sum(int(pred[i])==int(y[i]) for i in ids)
        out[g]=dict(count=count,correct=correct,accuracy=correct/count,loss=sum(-float(logprobs[i,y[i]]) for i in ids)/count)
    return out


def independent_decision(runs):
    result={}
    for arm in ARMS:
        fit=val=short=long=True
        rows=[r for r in runs if r['arm']==arm]
        assert len(rows)==3
        for r in rows:
            if r['status']!='trained':fit=val=short=long=False;continue
            for distance in DISTANCES:
                if arm=='local' and distance>4 or arm=='global' and distance==5:continue
                g=str(distance)
                fit &= r['metrics']['train_17'][g]['accuracy']>=.99
                val &= r['metrics']['validation_17'][g]['accuracy']>=.95
                short &= r['metrics']['test_17'][g]['accuracy']>=.95
                long &= r['metrics']['test_21'][g]['accuracy']>=.95
        result[arm]=dict(training_fit=fit,validation_reliability=val,short_reliability=fit and val and short,longer_transfer=fit and val and long)
    return result


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rp=ROOT/f'results/{PREFIX}.json';target=ROOT/f'results/{PREFIX}_audit.json';assert not target.exists()
    r=json.loads(rp.read_text());assert r['phase']=='complete'
    for p,h in {**r['historical'],**r['provenance']}.items():assert digest(ROOT/p)==h,p
    assert digest(ROOT/f'results/{PREFIX}_profile.json')==r['profile_sha256']
    ip=ROOT/f'results/{PREFIX}_data.json';assert digest(ip)==r['input_sha256']
    payload=json.loads(ip.read_text());assert audit_data(payload)==r['input_checks'];data=tensors(payload)
    pre=json.loads((ROOT/f'results/{PREFIX}_pretest.json').read_text())
    assert pre['phase']=='all_attempts_finished_before_test' and len(pre['runs'])==9
    assert all(not any(k.startswith('test') for k in run['metrics']) for run in pre['runs'])
    assert [run['checkpoint_sha256'] for run in pre['runs']]==[run['checkpoint_sha256'] for run in r['runs']]
    cells=[]
    for seed in SEEDS:
        paired=[run for run in r['runs'] if run['seed']==seed]
        assert len(paired)==3 and len({run['initial_sha256'] for run in paired})==1 and len({run['stream_sha256'] for run in paired})==1
        gen=torch.Generator().manual_seed(seed+1000);stream=hashlib.sha256()
        for _ in range(6000):stream.update(json.dumps(batch_indices(gen).tolist()).encode())
        assert all(run['stream_sha256']==stream.hexdigest() for run in paired)
    for run in r['runs']:
        if run['status']!='trained':continue
        for k in ('checkpoint','scores'):assert digest(ROOT/run[k])==run[k+'_sha256']
        torch.manual_seed(run['seed']);m=model_new('mqa');assert hash_state(m.state_dict())==run['initial_sha256']
        cp=torch.load(ROOT/run['checkpoint'],weights_only=True);m.load_state_dict(cp['model'])
        assert cp['step']==run['step']==6000 and all(int(s['step'])==6000 for s in cp['optimizer']['state'].values())
        saved=torch.load(ROOT/run['scores'],weights_only=True)
        for mode in ([run['arm'],'ablated'] if run['arm']=='global' else [run['arm']]):
            arm='local' if mode=='ablated' else mode
            for pool,(x,y,d) in data.items():
                with torch.no_grad():z=torch.cat([independent(m,c,arm)[:,-1] for c in x.split(128)])
                key='ablated/'+pool if mode=='ablated' else pool
                stored=run['ablated'][pool] if mode=='ablated' else run['metrics'][pool]
                err=float((z-saved[key]).abs().max());changes=int((z.argmax(-1)!=saved[key].argmax(-1)).sum())
                assert err<1e-9 and changes==0
                checked=metrics(z,y,d);me=max(abs(value-stored[g][k]) for g,vs in checked.items() for k,value in vs.items())
                assert me<1e-9
                invariance={}
                unreachable=(5,12,16) if arm=='local' else (5,) if arm=='global' else ()
                for distance in unreachable:
                    scores=z[d==distance].reshape(-1,6,19)
                    variation=float((scores-scores[:,:1]).abs().max());assert variation<1e-9
                    assert checked[str(distance)]['accuracy']<=1/6+1e-12
                    invariance[str(distance)]=variation
                cells.append(dict(seed=run['seed'],arm=run['arm'],mode=mode,pool=pool,predictions=len(y),logit_error=err,argmax_changes=changes,metric_error=me,unreachable_invariance=invariance))
    assert independent_decision(r['runs'])==r['decision']
    out=dict(passed=True,report_sha256=digest(rp),evaluations=len(cells),predictions=sum(c['predictions'] for c in cells),
        max_logit_error=max(c['logit_error'] for c in cells),max_metric_error=max(c['metric_error'] for c in cells),argmax_changes=sum(c['argmax_changes'] for c in cells),
        decision=r['decision'],historical_artifacts_preserved=len(r['historical']),frozen_files_checked=len(r['provenance']),cells=cells)
    write(target,out);print(json.dumps({k:v for k,v in out.items() if k!='cells'}))


if __name__=='__main__':main()
