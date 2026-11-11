"""All-row head-output ablation, with independent library weight-column checks."""
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from norm_placement import ROOT, model_for
from position_distractors import digest, pe, logits
from position_training import metrics


def forward(model, ids, block_index=-1, removed=()):
    x=model.embedding(ids)+pe(torch.arange(ids.shape[1]),16)
    batch,n,d=x.shape; mask=torch.ones(n,n,dtype=torch.bool).triu(1)
    inspection=[]; error=0.
    for index,b in enumerate(model.blocks):
        z=b.norm1(x); q,k,v=F.linear(z,b.self_attn.in_proj_weight,b.self_attn.in_proj_bias).chunk(3,-1)
        q,k,v=[t.reshape(batch,n,2,8).transpose(1,2) for t in (q,k,v)]
        a=(q@k.transpose(-2,-1)/math.sqrt(8)).masked_fill(mask,float('-inf')).softmax(-1)
        heads=a@v; out=b.self_attn.out_proj
        contributions=[F.linear(heads[:,h],out.weight[:,h*8:(h+1)*8]) for h in range(2)]
        merged=heads.transpose(1,2).reshape(batch,n,d)
        error=max(error,(sum(contributions)+out.bias-F.linear(merged,out.weight,out.bias)).abs().max().item())
        inspection.append((a,torch.stack(contributions,1)))
        if index==block_index:
            heads=heads.clone()
            for h in removed: heads[:,h]=0
        x=x+F.linear(heads.transpose(1,2).reshape(batch,n,d),out.weight,out.bias)
        x=x+b.linear2(b.activation(b.linear1(b.norm2(x))))
    return model.output(model.final_norm(x))[:,-1],inspection,error


def summarize(scores,y,base,ids):
    pred=scores.argmax(-1); correct=pred==y; prior=base.argmax(-1)==y
    rank=(ids[:,:6:2]==ids[:,-1:]).long().argmax(-1)
    return dict(**metrics(scores,y),changed=int((pred!=base.argmax(-1)).sum()),
                harmed=int((prior&~correct).sum()),rescued=int((~prior&correct).sum()),
                rank_accuracy=[correct[rank==j].double().mean().item() for j in range(3)])


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    rp=ROOT/'results/head_ablation.json'; dp=ROOT/'results/head_ablation_scores.pt'
    assert not rp.exists() and not dp.exists()
    prior_path=ROOT/'results/norm_extension.json'; prior=json.loads(prior_path.read_text())
    inp=ROOT/'results/norm_extension_inputs.json'; assert digest(inp)==prior['input_sha256']
    rows=json.loads(inp.read_text()); ids=torch.tensor([x for x,y in rows]); y=torch.tensor([y for x,y in rows])
    report=dict(input_sha256=digest(inp),prior_sha256=digest(prior_path),
                protocol_sha256=digest(ROOT/'docs/head-ablation-protocol.md'),
                source_sha256={n:digest(ROOT/'experiments'/n) for n in ('head_ablation.py','norm_placement.py','position_distractors.py','position_training.py','reserved_cycle.py','compare_depth.py')},runs=[])
    saved=[]
    conditions=[('ordinary',-1,())]+[(f'b{b+1}_{name}',b,heads) for b in range(2) for name,heads in [('h0',(0,)),('h1',(1,)),('both',(0,1))]]
    for original in prior['runs']:
        if (original['placement'],original['schedule'])!=('pre','constant'): continue
        cp=ROOT/original['checkpoint']; assert digest(cp)==original['checkpoint_sha256']
        weights=torch.load(cp,weights_only=True)['model']; model=model_for('pre').eval(); model.load_state_dict(weights)
        base=logits(model,ids); assert metrics(base,y)==original['test_6000']
        run=dict(seed=original['seed'],checkpoint=original['checkpoint'],checkpoint_sha256=digest(cp),conditions={},inspection=[])
        tensors={}; max_sum=0.; max_parity=0.
        with torch.no_grad():
            attention=[[],[]]; contribution=[[],[]]
            for name,b,heads in conditions:
                parts=[]
                for chunk in ids.split(256):
                    s,cache,err=forward(model,chunk,b,heads); parts.append(s); max_sum=max(max_sum,err)
                    if name=='ordinary':
                        for j,(a,c) in enumerate(cache): attention[j].append(a); contribution[j].append(c)
                scores=torch.cat(parts); assert torch.isfinite(scores).all()
                # Independent intervention: use library attention with zeroed W_O columns.
                library=model_for('pre').eval(); library.load_state_dict(weights)
                if b>=0:
                    for h in heads: library.blocks[b].self_attn.out_proj.weight[:,h*8:(h+1)*8]=0
                err=(scores-logits(library,ids)).abs().max().item(); max_parity=max(max_parity,err); assert err<1e-10
                run['conditions'][name]=summarize(scores,y,base,ids); tensors[name]=scores
            rank=(ids[:,:6:2]==ids[:,-1:]).long().argmax(-1); target=2*rank+1
            for b in range(2):
                a=torch.cat(attention[b]); c=torch.cat(contribution[b])
                for h in range(2):
                    final=a[:,h,-1]; mass=final.gather(1,target[:,None]).squeeze(1)
                    run['inspection'].append(dict(block=b+1,head=h,final_distribution=final.mean(0).tolist(),
                        target_value_mass=mass.mean().item(),target_value_top=(final.argmax(-1)==target).double().mean().item(),
                        preceding_key_mass=a[:,h,[1,3,5],[0,2,4]].mean().item(),
                        projected_final_l2=c[:,h,-1].norm(dim=-1).mean().item()))
            run['loss_interaction']={f'b{b}':run['conditions'][f'b{b}_both']['loss']-run['conditions'][f'b{b}_h0']['loss']-run['conditions'][f'b{b}_h1']['loss']+run['conditions']['ordinary']['loss'] for b in (1,2)}
        assert max_sum<1e-10 and digest(cp)==original['checkpoint_sha256']
        run['verification']=dict(max_sum_error=max_sum,max_library_error=max_parity)
        report['runs'].append(run); saved.append(dict(seed=run['seed'],scores=tensors))
        print(json.dumps(dict(seed=run['seed'],accuracy={n:c['accuracy'] for n,c in run['conditions'].items()})),flush=True)
    assert len(report['runs'])==5
    torch.save(saved,dp); report['scores_sha256']=digest(dp)
    rp.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
