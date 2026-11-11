"""Fresh whole-mapping confirmation of fixed joint QK donors."""
import itertools
import json
import random
import statistics
from pathlib import Path
import torch
from reserved_cycle import model_new
from activation_paths import forward
from qkv_paths import project, attention
from position_routing import measures
from position_training import mapping
from position_distractors import digest, logits

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out=ROOT/'results/joint_routing_confirmation.json'
    sp=ROOT/'results/joint_routing_confirmation_inputs.json'
    dp=ROOT/'results/joint_routing_confirmation_scores.json'
    assert not any(p.exists() for p in (out,sp,dp))
    priorpath=ROOT/'results/position_training.json'; prior=json.loads(priorpath.read_text())
    assert prior['phase']=='complete'
    for name,h in prior['source_sha256'].items(): assert digest(ROOT/'experiments'/name)==h
    selected=[r for r in prior['runs'] if r['group']=='varied']
    assert [r['seed'] for r in selected]==[461,487,509,541,571]
    for r in selected: assert digest(ROOT/r['checkpoint'])==r['checkpoint_sha256']
    excluded=set(); exclusion_hashes={}
    for name in ('reserved_cycle_splits.json','position_training_splits.json'):
        path=ROOT/'results'/name; exclusion_hashes[name]=digest(path)
        for rows in json.loads(path.read_text()).values(): excluded.update(mapping(x) for x,y in rows)
    assert len(excluded)==4800
    rng=random.Random(20261223); mappings=[]
    for values in itertools.combinations(range(12,18),3):
        candidates=[tuple(zip(keys,vs)) for keys in itertools.combinations(range(12),3) for vs in itertools.permutations(values) if tuple(zip(keys,vs)) not in excluded]
        rng.shuffle(candidates); assert len(candidates)>=20; mappings.extend(candidates[:20])
    assert len(set(mappings))==400 and not set(mappings)&excluded
    rows=[([t for pair in order for t in pair]+[18,q],v) for m in mappings for order in itertools.permutations(m) for q,v in m]
    assert len(rows)==7200 and all(dict(mapping(x))[x[-1]]==y for x,y in rows)
    assert all(sum(y==v for x,y in rows)==1200 for v in range(12,18))
    assert all(sum(x[:6:2].index(x[-1])==r for x,y in rows)==2400 for r in range(3))
    sp.write_text(json.dumps(rows,separators=(',',':'))+'\n')
    report=dict(phase='inputs_and_manifest_saved_before_predictions',torch=torch.__version__,device='cpu',dtype='float64',
                prior_sha256=digest(priorpath),inputs_sha256=digest(sp),exclusion_hashes=exclusion_hashes,excluded_mappings=4800,mappings=400,sequences=7200,
                protocol_sha256=digest(ROOT/'docs/joint-routing-confirmation-protocol.md'),
                source_sha256={n:digest(ROOT/'experiments'/n) for n in ('confirm_joint_routing.py','reserved_cycle.py','compare_depth.py','activation_paths.py','qkv_paths.py','position_routing.py','position_training.py','position_distractors.py')},
                checkpoints=[{k:r[k] for k in ('seed','checkpoint','checkpoint_sha256')} for r in selected],runs=[])
    def save(): out.write_text(json.dumps(report,indent=2)+'\n')
    save()
    x=torch.tensor([x for x,y in rows]); y=torch.tensor([y for x,y in rows]); details=[]
    with torch.no_grad():
        for old in selected:
            path=ROOT/old['checkpoint']; model=model_new().eval(); model.load_state_dict(torch.load(path,weights_only=True))
            cache={}; scores={}; error=0.
            for shift in (4,8,12):
                ss=[]; qq=[[],[],[]]
                for chunk in x.split(256):
                    s,c=forward(model,chunk,shift); qkv=project(model,c['r1']); a=attention(model,*qkv)
                    library=logits(model,chunk,(torch.arange(8)+shift).expand_as(chunk))
                    sham=forward(model,chunk,shift,{'a1':a},('a1',))[0]
                    error=max(error,(s-library).abs().max().item(),(a-c['a1']).abs().max().item(),(s-sham).abs().max().item())
                    ss.append(s)
                    for parts,t in zip(qq,qkv): parts.append(t)
                cache[shift]=tuple(torch.cat(t) for t in qq); scores[f'{shift}_ordinary']=torch.cat(ss)
            for shift in (8,12):
                ss=[]
                for start in range(0,len(x),256):
                    end=start+256
                    a=attention(model,cache[4][0][start:end],cache[4][1][start:end],cache[shift][2][start:end])
                    ss.append(forward(model,x[start:end],shift,{'a1':a},('a1',))[0])
                scores[f'{shift}_joint']=torch.cat(ss)
            conditions={}
            for name,s in scores.items():
                shift=int(name.split('_')[0]); original=scores[f'{shift}_ordinary'].argmax(-1)
                m=measures(s,y,original)[0]
                m['every_order_query_correct']=(s.argmax(-1)==y).reshape(400,18).all(1).double().mean().item()
                conditions[name]=m
            assert error<1e-10 and digest(path)==old['checkpoint_sha256']
            report['runs'].append(dict(seed=old['seed'],conditions=conditions,max_control_error=error))
            details.append(dict(seed=old['seed'],scores={n:s.tolist() for n,s in scores.items()}))
            save(); print(json.dumps(dict(seed=old['seed'],accuracy={n:m['accuracy'] for n,m in conditions.items()})),flush=True)
    criteria={}
    for shift,threshold,gain_threshold in ((8,.95,.10),(12,.90,.05)):
        gain=statistics.mean(r['conditions'][f'{shift}_joint']['accuracy']-r['conditions'][f'{shift}_ordinary']['accuracy'] for r in report['runs'])
        all_pass=all(r['conditions'][f'{shift}_joint']['accuracy']>=threshold for r in report['runs'])
        criteria[str(shift)]=dict(joint_all_accuracy=all_pass,mean_gain=gain,gain_pass=gain>=gain_threshold,
                                  joint_pass=all_pass and gain>=gain_threshold,
                                  ordinary_all_90=all(r['conditions'][f'{shift}_ordinary']['accuracy']>=.9 for r in report['runs']))
    dp.write_text(json.dumps(details,separators=(',',':'))+'\n')
    report.update(scores_sha256=digest(dp),criteria=criteria,phase='complete'); save(); print(json.dumps(criteria))


if __name__=='__main__': main()
