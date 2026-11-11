"""Educational trace of the first saved six-pair example and its reversal."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from ffn_width import ROOT, model_new, forward
from positional_comparison import rotate
from audit_ffn_width import independent
from position_distractors import digest


def main():
    torch.set_num_threads(1)
    report_path=ROOT/'results/ffn_width.json'; report=json.loads(report_path.read_text())
    for name,h in report['provenance'].items():
        p=ROOT/'docs/ffn-width-protocol.md' if name=='protocol' else ROOT/'experiments'/name
        assert digest(p)==h
    run=next(r for r in report['runs'] if r['seed']==1201 and r['group']=='ff32')
    cp=ROOT/run['checkpoint']; assert digest(cp)==run['checkpoint_sha256']
    dp=ROOT/'results/ffn_width_data.json';assert digest(dp)==report['input_sha256']
    rows=json.loads(dp.read_text())['datasets']['test_6'][:2]
    ids=torch.tensor([x for x,y in rows]); labels=torch.tensor([y for x,y in rows])
    model=model_new('ff32').eval();model.load_state_dict(torch.load(cp,weights_only=True)['model'])
    layers=[]; n=ids.shape[1];positions=torch.arange(n)+4;mask=torch.ones(n,n,dtype=torch.bool).triu(1)
    with torch.no_grad():
        x=model.embedding(ids); embedding=x.clone()
        for block in model.blocks:
            start=x.clone(); z=block.norm1(x)
            q,k,v=F.linear(z,block.self_attn.in_proj_weight,block.self_attn.in_proj_bias).chunk(3,-1)
            q,k,v=[t.reshape(len(ids),n,2,8).transpose(1,2) for t in (q,k,v)]
            qr,kr=rotate(q,positions),rotate(k,positions)
            scores=qr@kr.transpose(-1,-2)/(8**.5);weights=scores.masked_fill(mask,float('-inf')).softmax(-1)
            mixed=weights@v; attention=block.self_attn.out_proj(mixed.transpose(1,2).reshape(len(ids),n,16))
            residual=start+attention; fn=block.norm2(residual); hidden=block.linear1(fn); activated=F.relu(hidden); branch=block.linear2(activated);x=residual+branch
            layers.append(dict(input=start,norm1=z,q=q,k=k,v=v,rotated_q=qr,rotated_k=kr,unmasked_scores=scores,weights=weights,mixed_values=mixed,attention_branch=attention,after_attention=residual,norm2=fn,ffn_preactivation=hidden,ffn_activation=activated,ffn_branch=branch,output=x.clone()))
            assert (weights.masked_select(mask)==0).all()
            assert (weights.sum(-1)-1).abs().max()<1e-12
        normalized=model.final_norm(x); logits=model.output(normalized); probabilities=logits[:,-1].softmax(-1)
        implementation_error=(logits-forward(model,ids,'ff32')).abs().max().item()
        independent_error=(logits[:,-1]-independent(model,ids,'ff32')).abs().max().item()
        sp=ROOT/'results/ffn_width_scores.pt';assert digest(sp)==report['scores_sha256']
        saved=next(d for d in torch.load(sp,weights_only=True) if d['seed']==1201 and d['group']=='ff32')['scores']['6'][:2]
        archived_error=(logits[:,-1]-saved).abs().max().item()
        assert max(implementation_error,independent_error,archived_error)<1e-10
    def serial(obj):
        if isinstance(obj,torch.Tensor):return obj.tolist()
        if isinstance(obj,dict):return {k:serial(v) for k,v in obj.items()}
        if isinstance(obj,list):return [serial(v) for v in obj]
        return obj
    examples=[]
    for i,(tokens,label) in enumerate(rows):
        pairs=list(zip(tokens[:-2:2],tokens[1:-2:2])); matches=[j for j,(k,v) in enumerate(pairs) if k==tokens[-1]]
        assert len(matches)==2 and label==pairs[matches[-1]][1]
        oldpos,newpos=[2*j+1 for j in matches]; heads=[]
        for layer,values in enumerate(layers):
            for head in range(2):
                row=values['weights'][i,head,-1];top=int(row.argmax())
                heads.append(dict(block=layer+1,head=head+1,top_token_position=top,top_weight=row[top].item(),older_value_weight=row[oldpos].item(),latest_value_weight=row[newpos].item(),latest_value_score=values['unmasked_scores'][i,head,-1,newpos].item(),query_vector=values['rotated_q'][i,head,-1].tolist(),latest_key_vector=values['rotated_k'][i,head,newpos].tolist()))
        examples.append(dict(row=i,tokens=tokens,pairs=pairs,query=tokens[-1],label=label,prediction=int(logits[i,-1].argmax()),correct_probability=probabilities[i,label].item(),older_value_position=oldpos,latest_value_position=newpos,heads=heads,final_logits=logits[i,-1].tolist(),final_probabilities=probabilities[i].tolist()))
    assert examples[0]['pairs'][::-1]==examples[1]['pairs'] and labels[0]!=labels[1]
    result=dict(selection='First seed1201 ff32; test_6 rows0 and1, selected by index without filtering predictions. Previously evaluated data; illustrative only.',examples=examples,checks=dict(passed=True,implementation_error=implementation_error,independent_error=independent_error,archived_error=archived_error),provenance=dict(report=digest(report_path),data=digest(dp),checkpoint=digest(cp),scores=digest(sp),source=digest(Path(__file__))),tensors=serial(dict(embedding=embedding,mask=mask,layers=layers,final_normalized=normalized,logits=logits)))
    (ROOT/'results/trained_recall_trace.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='tensors'},indent=2))


if __name__=='__main__':main()
