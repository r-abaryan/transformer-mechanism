"""Separate recurrent kernel, model gradients, labels and final predictions."""
import hashlib
import json
import torch
from torch.nn import functional as F
from audit_kv_cache_equivalence import norm, rotation
from audit_mqa_training import independent as softmax_reference, audit_inputs
from audit_recurrent_retrieval import baselines, metrics
from linear_attention import recurrent
from linear_retrieval import ROOT, SEEDS, ARMS, models, forward, datasets, evaluate, decision, write, hash_state, digest


def independent(model,ids,arm,omit_denominator=False):
    if arm=='softmax':return softmax_reference(model,ids)
    x=model.embedding.weight[ids];n=ids.shape[1];pos=torch.arange(n)+4
    for block in model.blocks:
        z=norm(x,block.norm1);a=block.attn
        q,k,v=[z@getattr(a,t+'_weight').T+getattr(a,t+'_bias') for t in ('q','k','v')]
        q=q.reshape(len(ids),n,2,8).permute(0,2,1,3)
        k,v=[t.reshape(len(ids),n,1,8).permute(0,2,1,3).expand(len(ids),2,n,8) for t in (k,v)]
        q,k=rotation(q,pos),rotation(k,pos)
        y,_=recurrent(q,k,v,torch.ones(len(ids),1,n,dtype=torch.bool))
        if omit_denominator:
            fq=torch.where(q<0,q.exp(),q+1);fk=torch.where(k<0,k.exp(),k+1)
            y=y*(fq*fk.cumsum(-2)).sum(-1,keepdim=True)
        y=y.permute(0,2,1,3).reshape(len(ids),n,16)
        x=x+y@a.out_proj.weight.T+a.out_proj.bias
        h=norm(x,block.norm2)@block.linear1.weight.T+block.linear1.bias
        x=x+h.clamp_min(0)@block.linear2.weight.T+block.linear2.bias
    return norm(x,model.final_norm)@model.output.weight.T


def preflight(payload,data):
    labels=audit_inputs(payload);rows=[]
    # A shifted-label fixture must disagree with the independent last-match oracle.
    _,labels4=data['train_4'];assert bool(((labels4-12+1)%6+12!=labels4).all())
    for seed in SEEDS:
        pair=models(seed);assert hash_state(pair['softmax'].state_dict())==hash_state(pair['linear'].state_dict())
        for arm,model in pair.items():
            x,y=data['train_4'];x,y=x[:9],y[:9]
            a=forward(model,x,arm);b=independent(model,x,arm)
            ga=torch.autograd.grad(F.cross_entropy(a[:,-1],y),tuple(model.parameters()))
            gb=torch.autograd.grad(F.cross_entropy(b[:,-1],y),tuple(model.parameters()))
            oe=float((a-b).abs().max().detach());ge=max(float((u-v).abs().max()) for u,v in zip(ga,gb))
            with torch.no_grad():
                pe=float((forward(model,x[:,:5],arm)-a[:,:5]).abs().max())
                be=float((forward(model,x[:1],arm)-a[:1]).abs().max())
                bad=float((independent(model,x,arm,True)-a).abs().max()) if arm=='linear' else None
            assert max(oe,ge,pe,be)<1e-10
            if bad is not None:assert bad>1e-6
            rows.append(dict(seed=seed,arm=arm,output_error=oe,gradient_error=ge,prefix_error=pe,batch_error=be,missing_denominator_effect=bad))
    return dict(passed=True,labels=labels,shifted_labels_detected=True,cases=rows)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rp=ROOT/'results/linear_retrieval.json';r=json.loads(rp.read_text());assert r['phase']=='complete'
    assert all(digest(ROOT/p)==h for p,h in r['provenance'].items())
    assert all(digest(ROOT/p)==h for p,h in r['historical'].items())
    payload,data=datasets();labels=audit_inputs(payload);assert baselines(data)==r['baselines']
    pre=json.loads((ROOT/'results/linear_retrieval_pretest.json').read_text());assert pre['phase']=='all_attempts_finished_before_test'
    assert {(s,a) for s in SEEDS for a in ARMS}=={(v['seed'],v['arm']) for v in r['runs']}
    cells=[]
    for seed in SEEDS:
        rows=[v for v in r['runs'] if v['seed']==seed]
        assert len({v['initial_sha256'] for v in rows})==1
        assert len({v['stream_sha256'] for v in rows})==1
        g=torch.Generator().manual_seed(seed+1000);stream=hashlib.sha256()
        for step in range(1,6001):stream.update(json.dumps(torch.randint(len(data[f'train_{3 if step%2 else 4}'][0]),(64,),generator=g).tolist()).encode())
        for run in rows:
            if run['status']!='trained':continue
            assert run['stream_sha256']==stream.hexdigest()
            model=models(seed)[run['arm']];assert hash_state(model.state_dict())==run['initial_sha256']
            cp=ROOT/run['checkpoint'];assert digest(cp)==run['checkpoint_sha256']
            old=next(v for v in pre['runs'] if v['seed']==seed and v['arm']==run['arm'])
            assert old['checkpoint_sha256']==run['checkpoint_sha256'] and not any(k.startswith('test_') for k in old['metrics'])
            state=torch.load(cp,weights_only=True);assert state['step']==6000 and all(v['step'].item()==6000 for v in state['optimizer']['state'].values())
            model.load_state_dict(state['model']);sp=ROOT/run['scores'];assert digest(sp)==run['scores_sha256'];saved=torch.load(sp,weights_only=True)
            for key,(x,y) in data.items():
                _,scores=evaluate(model,(x,y),run['arm'],independent)
                err=float((scores-saved[key]).abs().max());changes=int((scores.argmax(-1)!=saved[key].argmax(-1)).sum())
                actual=metrics(scores,x,y);me=max(abs(v-run['metrics'][key][k]) for k,v in actual.items())
                assert err<1e-10 and changes==0 and me<1e-10
                cells.append(dict(seed=seed,arm=run['arm'],split=key,predictions=len(y),logit_error=err,metric_error=me,argmax_changes=changes))
            print(dict(audited_seed=seed,arm=run['arm']),flush=True)
    assert decision(r['runs'])==r['decision']
    p=ROOT/'results/linear_retrieval_audit.json';assert not p.exists()
    result=dict(passed=True,report_sha256=digest(rp),labels=labels,cells=cells,evaluations=len(cells),predictions=sum(c['predictions'] for c in cells),max_logit_error=max(c['logit_error'] for c in cells),max_metric_error=max(c['metric_error'] for c in cells),historical_reports_preserved=len(r['historical']))
    write(p,result);print({k:v for k,v in result.items() if k not in ('cells','labels')},flush=True)


if __name__=='__main__':main()
