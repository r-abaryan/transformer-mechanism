"""Single-pass canonical position adapter for consecutive-position recall."""
import itertools
import json
import random
from pathlib import Path
import torch
from reserved_cycle import model_new
from position_distractors import digest,logits
from activation_paths import forward
from position_routing import measures
from position_training import mapping

ROOT=Path(__file__).resolve().parents[1]


def canonical_positions(positions):
    if positions.ndim!=2 or positions.shape[1]==0 or positions.dtype!=torch.long:
        raise ValueError('Expected a nonempty matrix of integer positions')
    relative=positions-positions[:,:1]
    if not torch.equal(relative,torch.arange(positions.shape[1]).expand_as(relative)):
        raise ValueError('Canonicalization requires consecutive positions')
    return relative+4


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out=ROOT/'results/canonical_positions.json'; sp=ROOT/'results/canonical_positions_inputs.json'; dp=ROOT/'results/canonical_positions_scores.json'
    assert not any(p.exists() for p in (out,sp,dp))
    pp=ROOT/'results/position_training.json'; prior=json.loads(pp.read_text())
    for n,h in prior['source_sha256'].items(): assert digest(ROOT/'experiments'/n)==h
    runs=[r for r in prior['runs'] if r['group']=='varied']; assert len(runs)==5
    exclusions={}; excluded=set()
    for name in ('reserved_cycle_splits.json','position_training_splits.json','joint_routing_confirmation_inputs.json'):
        path=ROOT/'results'/name; exclusions[name]=digest(path); data=json.loads(path.read_text())
        groups=data.values() if isinstance(data,dict) else [data]
        for rows in groups: excluded.update(mapping(x) for x,y in rows)
    assert len(excluded)==5200
    rng=random.Random(20261224); maps=[]
    for vs in itertools.combinations(range(12,18),3):
        candidates=[tuple(zip(ks,v)) for ks in itertools.combinations(range(12),3) for v in itertools.permutations(vs) if tuple(zip(ks,v)) not in excluded]
        rng.shuffle(candidates); maps.extend(candidates[:20])
    assert len(set(maps))==400 and not set(maps)&excluded
    rows=[([t for pair in order for t in pair]+[18,q],v) for m in maps for order in itertools.permutations(m) for q,v in m]
    assert len(rows)==7200
    assert all(dict(mapping(x))[x[-1]]==y for x,y in rows)
    assert all(sum(y==v for x,y in rows)==1200 for v in range(12,18))
    assert all(sum(x[:6:2].index(x[-1])==r for x,y in rows)==2400 for r in range(3))
    for r in runs: assert digest(ROOT/r['checkpoint'])==r['checkpoint_sha256']
    sp.write_text(json.dumps(rows,separators=(',',':'))+'\n')
    report=dict(phase='inputs_saved_before_predictions',prior_sha256=digest(pp),inputs_sha256=digest(sp),exclusion_hashes=exclusions,
                protocol_sha256=digest(ROOT/'docs/canonical-position-protocol.md'),
                source_sha256={n:digest(ROOT/'experiments'/n) for n in ('canonical_positions.py','reserved_cycle.py','compare_depth.py','position_distractors.py','activation_paths.py','position_training.py','position_routing.py')},
                checkpoints=[{k:r[k] for k in ('seed','checkpoint','checkpoint_sha256')} for r in runs],runs=[])
    def save(): out.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try: canonical_positions(torch.tensor([[0,1,3]]))
    except ValueError: report['gapped_positions_rejected']=True
    else: raise AssertionError('Gapped positions accepted')
    x=torch.tensor([x for x,y in rows]); y=torch.tensor([y for x,y in rows]); details=[]
    with torch.no_grad():
        for old in runs:
            model=model_new().eval(); model.load_state_dict(torch.load(ROOT/old['checkpoint'],weights_only=True))
            scores={}; error=0.
            for shift in (0,4,8,12):
                pos=(torch.arange(8)+shift).expand_as(x)
                ordinary=logits(model,x,pos); canonical=logits(model,x,canonical_positions(pos))
                explicit=torch.cat([forward(model,c,shift)[0] for c in x.split(256)])
                error=max(error,(ordinary-explicit).abs().max().item())
                scores[f'{shift}_ordinary']=ordinary; scores[f'{shift}_canonical']=canonical
            for shift in (0,4,8,12): error=max(error,(scores[f'{shift}_canonical']-scores['4_ordinary']).abs().max().item())
            assert error<1e-10 and digest(ROOT/old['checkpoint'])==old['checkpoint_sha256']
            conditions={}
            for name,s in scores.items():
                m=measures(s,y)[0]; m['every_order_query_correct']=(s.argmax(-1)==y).reshape(400,18).all(1).double().mean().item(); conditions[name]=m
            report['runs'].append(dict(seed=old['seed'],conditions=conditions,max_control_error=error))
            details.append(dict(seed=old['seed'],scores={k:s.tolist() for k,s in scores.items()})); save()
            print(json.dumps(dict(seed=old['seed'],accuracy={k:m['accuracy'] for k,m in conditions.items() if k.endswith('ordinary') or k=='0_canonical'})),flush=True)
    dp.write_text(json.dumps(details,separators=(',',':'))+'\n')
    report.update(phase='complete',scores_sha256=digest(dp),criteria=dict(canonical_all_99=all(r['conditions']['0_canonical']['accuracy']>=.99 for r in report['runs']),zero_preserved=all(r['conditions']['0_ordinary']['accuracy']-r['conditions']['0_canonical']['accuracy']<=.01 for r in report['runs']),offset_identity=True))
    save(); print(json.dumps(report['criteria']))


if __name__=='__main__': main()
