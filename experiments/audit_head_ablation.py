"""Audit saved predictions and inspect attention through the library API."""
import json
from pathlib import Path
import torch
from head_ablation import ROOT, summarize
from norm_placement import model_for
from position_distractors import digest, pe


def main():
    torch.set_num_threads(1)
    rp=ROOT/'results/head_ablation.json'; report=json.loads(rp.read_text())
    for n,h in report['source_sha256'].items(): assert digest(ROOT/'experiments'/n)==h
    assert digest(ROOT/'docs/head-ablation-protocol.md')==report['protocol_sha256']
    inp=ROOT/'results/norm_extension_inputs.json'; assert digest(inp)==report['input_sha256']
    dp=ROOT/'results/head_ablation_scores.pt'; assert digest(dp)==report['scores_sha256']
    rows=json.loads(inp.read_text()); ids=torch.tensor([x for x,y in rows]); y=torch.tensor([y for x,y in rows])
    rank=(ids[:,:6:2]==ids[:,-1:]).long().argmax(-1); target=2*rank+1
    saved=torch.load(dp,weights_only=True); errors=[]; count=0
    assert len(saved)==len(report['runs'])==5
    for run,detail in zip(report['runs'],saved):
        assert run['seed']==detail['seed'] and len(detail['scores'])==7
        scores=detail['scores']; base=scores['ordinary']
        for name,s in scores.items():
            assert summarize(s,y,base,ids)==run['conditions'][name]; count+=1
        cp=ROOT/run['checkpoint']; assert digest(cp)==run['checkpoint_sha256']
        model=model_for('pre').eval(); model.load_state_dict(torch.load(cp,weights_only=True)['model'])
        attentions=[[],[]]
        with torch.no_grad():
            for chunk in ids.split(256):
                x=model.embedding(chunk)+pe(torch.arange(8),16); mask=torch.ones(8,8,dtype=torch.bool).triu(1)
                for j,b in enumerate(model.blocks):
                    z=b.norm1(x); a,w=b.self_attn(z,z,z,attn_mask=mask,need_weights=True,average_attn_weights=False)
                    attentions[j].append(w); x=x+a; x=x+b.linear2(b.activation(b.linear1(b.norm2(x))))
        for item in run['inspection']:
            a=torch.cat(attentions[item['block']-1])[:,item['head']]; final=a[:,-1]
            expected=dict(final_distribution=final.mean(0).tolist(),target_value_mass=final.gather(1,target[:,None]).mean().item(),
                          target_value_top=(final.argmax(-1)==target).double().mean().item(),preceding_key_mass=a[:,[1,3,5],[0,2,4]].mean().item())
            for k,v in expected.items():
                err=(torch.tensor(v,dtype=torch.float64)-torch.tensor(item[k],dtype=torch.float64)).abs().max().item()
                errors.append(err); assert err<1e-10
    result=dict(passed=True,conditions=count,max_library_inspection_error=max(errors),report_sha256=digest(rp),source_sha256=digest(Path(__file__)))
    (ROOT/'results/head_ablation_audit.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
