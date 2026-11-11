"""Matched Q-only, K-only and joint replacements in both offset directions."""
import json
from pathlib import Path
import torch
from reserved_cycle import model_new
from activation_paths import forward
from qkv_paths import project,attention
from position_routing import measures
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out=ROOT/'results/query_key.json'; dp=ROOT/'results/query_key_scores.json'
    assert not out.exists() and not dp.exists()
    pp=ROOT/'results/position_routing.json'; prior=json.loads(pp.read_text())
    sp=ROOT/'results/position_training_splits.json'; assert digest(sp)==prior['split_sha256']
    for n,h in prior['source_sha256'].items(): assert digest(ROOT/'experiments'/n)==h
    rows=json.loads(sp.read_text())['test']; x=torch.tensor([x for x,y in rows]); y=torch.tensor([y for x,y in rows])
    report=dict(prior_sha256=digest(pp),split_sha256=digest(sp),protocol_sha256=digest(ROOT/'docs/query-key-protocol.md'),
                source_sha256={n:digest(ROOT/'experiments'/n) for n in ('query_key.py','position_routing.py','qkv_paths.py','activation_paths.py','reserved_cycle.py','compare_depth.py','position_distractors.py')},runs=[])
    details=[]
    with torch.no_grad():
        for old in prior['runs']:
            if old['group']!='varied': continue
            path=ROOT/old['checkpoint']; assert digest(path)==old['checkpoint_sha256']
            model=model_new().eval(); model.load_state_dict(torch.load(path,weights_only=True))
            cache={}; refs={}; scores_saved={}; err=0.
            for shift in (4,8):
                ss=[]; qq=[[],[],[]]
                for chunk in x.split(256):
                    s,c=forward(model,chunk,shift); qkv=project(model,c['r1']); a=attention(model,*qkv)
                    err=max(err,(a-c['a1']).abs().max().item(),(forward(model,chunk,shift,{'a1':a},('a1',))[0]-s).abs().max().item())
                    ss.append(s)
                    for parts,t in zip(qq,qkv): parts.append(t)
                s=torch.cat(ss); cache[shift]=(s,tuple(torch.cat(t) for t in qq))
                refs[str(shift)]=measures(s,y)[0]; scores_saved[str(shift)]=s.tolist()
                for k in ('accuracy','loss'): assert abs(refs[str(shift)][k]-old['references'][str(shift)][k])<1e-10
            conditions={}; interactions={}
            for donor,recipient in ((4,8),(8,4)):
                for mode,replace in [('Q',(0,)),('K',(1,)),('QK',(0,1))]:
                    outputs=[]
                    for start in range(0,len(x),256):
                        end=start+256
                        qkv=[cache[donor if i in replace else recipient][1][i][start:end] for i in range(3)]
                        a=attention(model,*qkv)
                        outputs.append(forward(model,x[start:end],recipient,{'a1':a},('a1',))[0])
                    name=f'{donor}_to_{recipient}_{mode}'; s=torch.cat(outputs)
                    m=measures(s,y,cache[recipient][0].argmax(-1))[0]
                    if mode=='QK':
                        for k in ('accuracy','loss'): assert abs(m[k]-old['conditions'][f'{donor}_to_{recipient}_matched_QK'][k])<1e-10
                    conditions[name]=m; scores_saved[name]=s.tolist()
                prefix=f'{donor}_to_{recipient}'
                interactions[prefix]=conditions[prefix+'_QK']['accuracy']-conditions[prefix+'_Q']['accuracy']-conditions[prefix+'_K']['accuracy']+refs[str(recipient)]['accuracy']
            assert err<1e-10 and digest(path)==old['checkpoint_sha256']
            report['runs'].append(dict(seed=old['seed'],checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],references=refs,conditions=conditions,accuracy_interaction=interactions,max_control_error=err))
            details.append(dict(seed=old['seed'],scores=scores_saved))
            print(json.dumps(dict(seed=old['seed'],accuracy={k:v['accuracy'] for k,v in conditions.items()})),flush=True)
    dp.write_text(json.dumps(details,separators=(',',':'))+'\n'); report['scores_sha256']=digest(dp)
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
