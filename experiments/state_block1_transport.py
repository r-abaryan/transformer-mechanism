"""Source K/V substitutions in block 1, followed by an unmodified block 2."""
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from state_tracking import ROOT, model_new, forward, tensors, answer, save_json
from audit_state_tracking import oracle
from audit_kv_cache_equivalence import norm, rotation
from positional_comparison import rotate
from position_distractors import digest
from state_initial_paths import trace, metrics, scalar_metrics

DIRECTORY=ROOT/'results/state_block1_transport'
MODES=('sham','K_noninitial','V_noninitial','KV_noninitial','KV_reset','KV_suffix',
       'KV_query','KV_head0','KV_head1','initial_state','both')


def finish(model,x,reference):
    b=model.blocks[1];m=b.attn;n=x.shape[1];positions=torch.arange(n)+4
    z=norm(x,b.norm1) if reference else b.norm1(x)
    parts=[z@getattr(m,t+'_weight').T+getattr(m,t+'_bias') if reference else
           F.linear(z,getattr(m,t+'_weight'),getattr(m,t+'_bias')) for t in ('q','k','v')]
    q=parts[0].reshape(len(x),n,2,8).transpose(1,2)
    k,v=[t.reshape(len(x),n,model.groups,8).transpose(1,2) for t in parts[1:]]
    if reference:
        q,k=rotation(q,positions),rotation(k.expand(-1,2,-1,-1),positions)
        h=F.scaled_dot_product_attention(q,k,v.expand(-1,2,-1,-1),is_causal=True).transpose(1,2).reshape(len(x),n,16)
        x=x+h@m.out_proj.weight.T+m.out_proj.bias
        z=norm(x,b.norm2)@b.linear1.weight.T+b.linear1.bias
        x=x+z.clamp_min(0)@b.linear2.weight.T+b.linear2.bias
        return (norm(x,model.final_norm)@model.output.weight.T)[:,-1]
    q,k=rotate(q,positions),rotate(k,positions)
    k,v=[t.repeat_interleave(2//model.groups,dim=1) for t in (k,v)]
    mask=torch.ones(n,n,dtype=torch.bool).triu(1)
    a=(q@k.transpose(-1,-2)/(8**.5)).masked_fill(mask,-torch.inf).softmax(-1)
    h=(a@v).transpose(1,2).reshape(len(x),n,16)
    x=x+m.out_proj(h);x=x+b.linear2(F.relu(b.linear1(b.norm2(x))))
    return model.output(model.final_norm(x))[:,-1]


def patch(model,c,donor,mode,reference=False):
    b=model.blocks[0];m=b.attn;old=c[0];q=old['q'];n=q.shape[2]
    if mode in ('sham','initial_state'):
        outgoing=old['outgoing'].clone()
    else:
        k,v=[old[t].expand(-1,2,-1,-1).clone() for t in ('k','v')]
        heads=(0,) if mode=='KV_head0' else (1,) if mode=='KV_head1' else (0,1)
        for h in heads:
            if mode!='V_noninitial':k[:,h,0]=old['k'][donor,0,0]
            if mode!='K_noninitial':v[:,h,0]=old['v'][donor,0,0]
        if reference:
            mixed=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(len(q),n,16)
            new_attention=mixed@m.out_proj.weight.T+m.out_proj.bias
        else:
            mask=torch.ones(n,n,dtype=torch.bool).triu(1)
            weights=(q@k.transpose(-1,-2)/(8**.5)).masked_fill(mask,-torch.inf).softmax(-1)
            mixed=(weights@v).transpose(1,2).reshape(len(q),n,16);new_attention=m.out_proj(mixed)
        positions=[1] if mode=='KV_reset' else list(range(2,n-1)) if mode=='KV_suffix' else [n-1] if mode=='KV_query' else list(range(1,n))
        attention=old['attention'].clone();attention[:,positions]=new_attention[:,positions]
        x=old['incoming']+attention
        if reference:
            z=norm(x,b.norm2)@b.linear1.weight.T+b.linear1.bias
            outgoing=x+z.clamp_min(0)@b.linear2.weight.T+b.linear2.bias
        else:outgoing=x+b.linear2(F.relu(b.linear1(b.norm2(x))))
    if mode in ('initial_state','both'):outgoing[:,0]=old['outgoing'][donor,0]
    return finish(model,outgoing,reference),outgoing


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert not DIRECTORY.exists()
    protocol=ROOT/'docs/protocols/state-block1-transport.md';parent=ROOT/'results/state_reset_exposure'
    old=json.loads((parent/'report.json').read_text(encoding='utf-8'))
    pools=json.loads((parent/'data.json').read_text(encoding='utf-8'))['evaluation']
    runs=[r for r in old['runs'] if r['arm']=='independent'];assert {r['seed'] for r in runs}=={3011,3019,3023}
    paths=[protocol,Path(__file__),parent/'report.json',parent/'data.json',parent/'audit.json',
           ROOT/'experiments/state_initial_paths.py',ROOT/'results/state_initial_paths/report.json']+[ROOT/p for p in old['source_dependencies']]
    paths+=[ROOT/r[k] for r in runs for k in ('checkpoint','scores')]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    assert digest(parent/'data.json')==old['data_sha256'] and all(digest(ROOT/p)==h for p,h in old['source_dependencies'].items())
    scores={};records=[];controls=[];cells=[];evaluations=0;max_scalar=0.
    for run in runs:
        assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256'] and digest(ROOT/run['scores'])==run['scores_sha256']
        model=model_new().eval();model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        archive=torch.load(ROOT/run['scores'],weights_only=True);scores[run['seed']]={}
        for prefix in ('train/12','test/12'):
            rows=pools[prefix+'/long_tail_agree']+pools[prefix+'/long_tail_disagree'];x,y=tensors(rows)
            donor=(torch.arange(len(rows))+len(rows)//2)%len(rows)
            assert torch.equal(x[:,1:],x[donor,1:]) and torch.equal(y,y[donor]) and bool((x[:,0]+x[donor,0]==9).all())
            assert all(oracle(ids)==label==answer(ids[0]-4,ids[1:-1]) for ids,label in rows)
            with torch.no_grad():
                original,c=trace(model,x);reference,rc=trace(model,x,True)
                cached=torch.cat([archive[prefix+'/long_tail_agree'],archive[prefix+'/long_tail_disagree']])
                base_error=max(float((original-reference).abs().max()),float((original-cached).abs().max()),float((original-forward(model,x)[:,-1]).abs().max()))
                assert base_error<=1e-10 and torch.equal(original.argmax(-1),reference.argmax(-1));evaluations+=len(x)
                saved=dict(original=original,donor=donor);scores[run['seed']][prefix]=saved
                conditions={};state_error=0.
                for mode in MODES:
                    z,out=patch(model,c,donor,mode);rz,rout=patch(model,rc,donor,mode,True)
                    oe=float((z-rz).abs().max());se=float((out-rout).abs().max())
                    assert max(oe,se)<=1e-10 and torch.equal(z.argmax(-1),rz.argmax(-1));evaluations+=len(x)
                    cells.append(dict(seed=run['seed'],pool=prefix,condition=mode,count=len(x),logit_error=oe,state_error=se,argmax_changes=0))
                    saved[mode]=z;conditions[mode]=z
                    if mode=='KV_noninitial':state_error=float((out[:,1:]-c[0]['outgoing'][donor,1:]).abs().max())
                    val=metrics(original,original[donor],z,y);scalar=scalar_metrics(reference,reference[donor],rz,y)
                    me=max(abs(val[k]-scalar[k]) for k in val if val[k] is not None);assert me<=1e-10;max_scalar=max(max_scalar,me)
                    records.append(dict(seed=run['seed'],pool=prefix,condition=mode,**val))
                sham=float((conditions['sham']-original).abs().max());both=float((conditions['both']-original[donor]).abs().max())
                negative=float((conditions['KV_noninitial']-original[donor]).abs().max())
                assert max(sham,both,state_error)<=1e-10 and negative>1e-6
                controls.append(dict(seed=run['seed'],pool=prefix,baseline_error=base_error,sham_error=sham,both_donor_error=both,
                                     noninitial_donor_state_error=state_error,omitted_initial_state_effect=negative))
            print(f"Verified block-1 transport: {run['seed']} {prefix}",flush=True)
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    DIRECTORY.mkdir();torch.save(scores,DIRECTORY/'scores.pt')
    result=dict(scope='same-model matched source substitutions, block 1; no training',preserved_sha256=hashes,metrics=records,controls=controls,
                numerical_cells=cells,scores_sha256=digest(DIRECTORY/'scores.pt'),
                verification=dict(passed=True,evaluations=evaluations,argmax_changes=0,max_logit_error=max(v['logit_error'] for v in cells),
                                  max_state_error=max(v['state_error'] for v in cells),max_scalar_error=max_scalar))
    save_json(DIRECTORY/'report.json',result)
    with (DIRECTORY/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(result['verification']),flush=True)


if __name__=='__main__':main()
