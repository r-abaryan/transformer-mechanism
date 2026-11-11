"""Cross donor target-slot and answer agreement for frozen QK patches."""
import json
import random
from pathlib import Path
import torch
from reserved_cycle import model_new
from activation_paths import forward
from qkv_paths import project, attention
from position_routing import measures
from position_training import mapping
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out=ROOT/'results/donor_slots.json'; dp=ROOT/'results/donor_slots_scores.json'; ip=ROOT/'results/donor_slots_indices.json'
    assert not any(p.exists() for p in (out,dp,ip))
    priorpath=ROOT/'results/position_routing.json'; prior=json.loads(priorpath.read_text())
    sp=ROOT/'results/position_training_splits.json'; assert digest(sp)==prior['split_sha256']
    for name,sha in prior['source_sha256'].items(): assert digest(ROOT/'experiments'/name)==sha
    rows=json.loads(sp.read_text())['test']; x=torch.tensor([x for x,y in rows]); y=torch.tensor([y for x,y in rows])
    rank=(x[:,:6:2]==x[:,-1,None]).long().argmax(1)
    maps=[mapping(ids) for ids,target in rows]
    pools={(r,v):[i for i in range(len(x)) if rank[i]==r and y[i]==v] for r in range(3) for v in range(12,18)}
    rng=random.Random(20261222); indices={}
    for same_slot in (True,False):
        for same_answer in (True,False):
            name=f'{"same" if same_slot else "different"}_slot_{"same" if same_answer else "different"}_answer'
            selected=[]
            for i in range(len(x)):
                r=int(rank[i]) if same_slot else rng.choice([r for r in range(3) if r!=rank[i]])
                v=int(y[i]) if same_answer else rng.choice([v for v in range(12,18) if v!=y[i]])
                choices=[j for j in pools[r,v] if maps[j]!=maps[i]]
                j=rng.choice(choices)
                assert (rank[j]==rank[i]).item()==same_slot and (y[j]==y[i]).item()==same_answer
                selected.append(j)
            indices[name]=selected
    ip.write_text(json.dumps(indices,separators=(',',':'))+'\n')
    report=dict(prior_sha256=digest(priorpath),split_sha256=digest(sp),indices_sha256=digest(ip),
                protocol_sha256=digest(ROOT/'docs/donor-slot-protocol.md'),
                source_sha256={n:digest(ROOT/'experiments'/n) for n in ('donor_slots.py','position_routing.py','position_training.py','reserved_cycle.py','activation_paths.py','qkv_paths.py','position_distractors.py','compare_depth.py')},donors={},runs=[])
    for name,idx in indices.items():
        slot_y=x[torch.arange(len(x)),2*rank[idx]+1]
        report['donors'][name]=dict(unique_donor_rows=len(set(idx)),donor_rank_counts=torch.bincount(rank[idx],minlength=3).tolist(),
                                   slot_value_equals_donor_answer=(slot_y==y[idx]).double().mean().item())
    details=[]
    with torch.no_grad():
        for old in prior['runs']:
            path=ROOT/old['checkpoint']; assert digest(path)==old['checkpoint_sha256']
            model=model_new().eval(); model.load_state_dict(torch.load(path,weights_only=True))
            caches={}; error=0.
            for shift in (4,8):
                ss=[]; qq=[[],[],[]]
                for chunk in x.split(256):
                    s,c=forward(model,chunk,shift); qkv=project(model,c['r1']); a=attention(model,*qkv)
                    error=max(error,(a-c['a1']).abs().max().item(),(forward(model,chunk,shift,{'a1':a},('a1',))[0]-s).abs().max().item())
                    ss.append(s)
                    for parts,t in zip(qq,qkv): parts.append(t)
                caches[shift]=(torch.cat(ss),tuple(torch.cat(t) for t in qq))
                m,_=measures(caches[shift][0],y)
                for k in ('accuracy','loss'): assert abs(m[k]-old['references'][str(shift)][k])<1e-10
            base=caches[8][0].argmax(-1); conditions={}; scores_saved={'baseline':caches[8][0].tolist()}
            for name,selection in {'matched':list(range(len(x))),**indices}.items():
                idx=torch.tensor(selection); outputs=[]
                for start in range(0,len(x),256):
                    end=start+256
                    q,k=[caches[4][1][i][idx[start:end]] for i in (0,1)]; v=caches[8][1][2][start:end]
                    a=attention(model,q,k,v)
                    outputs.append(forward(model,x[start:end],8,{'a1':a},('a1',))[0])
                s=torch.cat(outputs); m,_=measures(s,y,base,y[idx]); pred=s.argmax(-1)
                slot_y=x[torch.arange(len(x)),2*rank[idx]+1]; distinct=slot_y!=y[idx]
                m.update(recipient_donor_slot_agreement=(pred==slot_y).double().mean().item(),
                         donor_original_accuracy=(caches[4][0].argmax(-1)[idx]==y[idx]).double().mean().item(),
                         distinct_labels_count=int(distinct.sum()),
                         distinct_labels_slot_agreement=(pred[distinct]==slot_y[distinct]).double().mean().item() if distinct.any() else None,
                         distinct_labels_donor_agreement=(pred[distinct]==y[idx][distinct]).double().mean().item() if distinct.any() else None)
                if name=='matched':
                    for k in ('accuracy','loss'): assert abs(m[k]-old['conditions']['4_to_8_matched_QK'][k])<1e-10
                conditions[name]=m; scores_saved[name]=s.tolist()
            assert error<1e-10 and digest(path)==old['checkpoint_sha256']
            report['runs'].append(dict(seed=old['seed'],group=old['group'],checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],max_control_error=error,conditions=conditions))
            details.append(dict(seed=old['seed'],group=old['group'],scores=scores_saved))
            print(json.dumps(dict(seed=old['seed'],group=old['group'],accuracy={n:m['accuracy'] for n,m in conditions.items()})),flush=True)
    dp.write_text(json.dumps(details,separators=(',',':'))+'\n'); report['scores_sha256']=digest(dp)
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
