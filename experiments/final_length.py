"""Final length boundary for the current canonical-position recipe."""
import itertools
import json
import random
from pathlib import Path
import torch
from reserved_cycle import model_new
from canonical_positions import canonical_positions
from position_distractors import digest,logits
from position_routing import measures
from activation_paths import forward

ROOT=Path(__file__).resolve().parents[1]


def identity(ids): return tuple(sorted(zip(ids[:-2:2],ids[1:-2:2])))


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out=ROOT/'results/final_length.json'; sp=ROOT/'results/final_length_inputs.json'; dp=ROOT/'results/final_length_scores.json'
    assert not any(p.exists() for p in (out,sp,dp))
    priorpath=ROOT/'results/position_training.json'; prior=json.loads(priorpath.read_text())
    for n,h in prior['source_sha256'].items(): assert digest(ROOT/'experiments'/n)==h
    runs=[r for r in prior['runs'] if r['group']=='varied']; assert len(runs)==5
    excluded=set(); exclusions={}
    for name in ('reserved_cycle_splits.json','position_training_splits.json','joint_routing_confirmation_inputs.json','canonical_positions_inputs.json'):
        p=ROOT/'results'/name; exclusions[name]=digest(p); d=json.loads(p.read_text())
        for rows in (d.values() if isinstance(d,dict) else [d]): excluded.update(identity(x) for x,y in rows)
    assert len(excluded)==5600
    rng=random.Random(20261225); datasets={}
    for n,per_combination in ((3,21),(4,28),(6,420)):
        maps=[]
        for values in itertools.combinations(range(12,18),n):
            chosen=set()
            while len(chosen)<per_combination:
                keys=sorted(rng.sample(range(12),n)); vs=rng.sample(values,n); m=tuple(zip(keys,vs))
                if m in chosen or m in excluded: continue
                chosen.add(m); maps.append(m)
        assert len(set(maps))==420
        rows=[]
        for m in maps:
            orders=rng.sample(list(itertools.permutations(m)),6)
            for order in orders:
                for q,v in m: rows.append(([t for pair in order for t in pair]+[18,q],v))
        assert len(rows)==420*6*n
        assert all(dict(identity(x))[x[-1]]==y for x,y in rows)
        assert all(sum(y==v for x,y in rows)==len(rows)//6 for v in range(12,18))
        assert all(sum(x[:-2:2].index(x[-1])==r for x,y in rows)==len(rows)//n for r in range(n))
        datasets[str(n)]=rows
    sp.write_text(json.dumps(datasets,separators=(',',':'))+'\n')
    for r in runs: assert digest(ROOT/r['checkpoint'])==r['checkpoint_sha256']
    report=dict(phase='inputs_saved_before_predictions',prior_sha256=digest(priorpath),inputs_sha256=digest(sp),exclusion_hashes=exclusions,
                protocol_sha256=digest(ROOT/'docs/final-length-protocol.md'),
                source_sha256={n:digest(ROOT/'experiments'/n) for n in ('final_length.py','canonical_positions.py','reserved_cycle.py','compare_depth.py','position_distractors.py','position_routing.py','activation_paths.py')},
                checkpoints=[{k:r[k] for k in ('seed','checkpoint','checkpoint_sha256')} for r in runs],runs=[])
    def save(): out.write_text(json.dumps(report,indent=2)+'\n')
    save(); details=[]
    with torch.no_grad():
        for old in runs:
            model=model_new().eval(); model.load_state_dict(torch.load(ROOT/old['checkpoint'],weights_only=True))
            conditions={}; scores={}; error=0.
            for name,rows in datasets.items():
                n=int(name); x=torch.tensor([x for x,y in rows]); y=torch.tensor([y for x,y in rows])
                pos=canonical_positions(torch.arange(x.shape[1]).expand_as(x))
                s=logits(model,x,pos); explicit=torch.cat([forward(model,c,4)[0] for c in x.split(256)])
                error=max(error,(s-explicit).abs().max().item()); correct=s.argmax(-1)==y
                rank=(x[:,:-2:2]==x[:,-1,None]).long().argmax(1)
                m=measures(s,y)[0]; m.update(sequences=len(rows),all_sampled_variants_correct=correct.reshape(420,6*n).all(1).double().mean().item(),accuracy_by_rank=[correct[rank==i].double().mean().item() for i in range(n)])
                conditions[name]=m; scores[name]=s.tolist()
            assert error<1e-10 and digest(ROOT/old['checkpoint'])==old['checkpoint_sha256']
            report['runs'].append(dict(seed=old['seed'],conditions=conditions,max_control_error=error)); details.append(dict(seed=old['seed'],scores=scores)); save()
            print(json.dumps(dict(seed=old['seed'],accuracy={n:m['accuracy'] for n,m in conditions.items()})),flush=True)
    dp.write_text(json.dumps(details,separators=(',',':'))+'\n')
    report.update(phase='complete',scores_sha256=digest(dp),criteria={str(n):all(r['conditions'][str(n)]['accuracy']>=t for r in report['runs']) for n,t in ((3,.99),(4,.90),(6,.90))}); save(); print(json.dumps(report['criteria']))


if __name__=='__main__': main()
