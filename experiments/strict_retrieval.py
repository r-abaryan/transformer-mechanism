"""Distinct values, whole mappings held out from frozen-model training."""
import itertools
import json
import statistics
from pathlib import Path
import torch
from torch.nn import functional as F
from activation_paths import forward
from compare_depth import StackedDecoder
from qkv_paths import project,attention
from position_distractors import digest,logits

ROOT=Path(__file__).resolve().parents[1]


def mapping(ids): return tuple(sorted(zip(ids[:-2:2],ids[1:-2:2])))


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    output=ROOT/'results/strict_retrieval.json'; splitpath=ROOT/'results/strict_retrieval_inputs.json'
    assert not output.exists() and not splitpath.exists()
    trainpath=ROOT/'results/depth_confirmation_splits.json'; splits=json.loads(trainpath.read_text())
    seen={mapping(x) for x,y in splits['train']}
    eligible=sorted(tuple(zip(k,v)) for k in itertools.combinations(range(8),3) for v in itertools.permutations(range(8,12),3) if tuple(zip(k,v)) not in seen)
    assert len(eligible)==21
    selected=set()
    for missing in range(8,12):
        group=[m for m in eligible if missing not in dict(m).values()]; assert len(group)>=4
        selected.update(group[:4])
    rows=[]; flags=[]
    for m in eligible:
        assert m not in seen
        for order in itertools.permutations(m):
            for query,target in m:
                ids=[t for pair in order for t in pair]+[12,query]
                assert dict(order)[query]==target and len(set(ids[1:6:2]))==3
                rows.append((ids,target)); flags.append(m in selected)
    x=torch.tensor([r[0] for r in rows]); y=torch.tensor([r[1] for r in rows]); primary=torch.tensor(flags)
    assert primary.sum()==288 and len(rows)==378
    assert all(int(((y==v)&primary).sum())==72 for v in range(8,12))
    rank=(x[:,:6:2]==x[:,-1,None]).long().argmax(1)
    assert all(int(((rank==i)&primary).sum())==96 for i in range(3))
    splitpath.write_text(json.dumps(dict(rows=rows,primary=flags,eligible_mappings=eligible),separators=(',',':'))+'\n')
    overlap={n:len(set(eligible)&{mapping(ids) for ids,t in data}) for n,data in splits.items()}
    cfpath=ROOT/'results/value_counterfactual_inputs.json'
    cf=json.loads(cfpath.read_text()); cf_maps={mapping(ids) for f in cf for ids in f['inputs'].values()}
    oldparents=ROOT/'results/pair_shift_confirmation_parents.json'
    from confirm_pair_shifts import crop_rows
    oldmaps={mapping(ids) for ids,t in crop_rows(json.loads(oldparents.read_text()))}
    priorpath=ROOT/'results/pair_shift_confirmation.json'; prior=json.loads(priorpath.read_text())
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',prior_sha256=digest(priorpath),split_sha256=digest(splitpath),
        training_split_sha256=digest(trainpath),overlap_mapping_counts=dict(confirmation_splits=overlap,value_counterfactuals=len(set(eligible)&cf_maps),pair_confirmation_crops=len(set(eligible)&oldmaps)),
        protocol_sha256=digest(ROOT/'docs/strict-retrieval-protocol.md'),
        source_sha256={n:digest(ROOT/'experiments'/n) for n in ('strict_retrieval.py','activation_paths.py','qkv_paths.py','compare_depth.py','position_distractors.py','confirm_pair_shifts.py')},runs=[])
    shifts=(0,1,2,4,6,8)
    for run in prior['runs']:
        path=ROOT/run['checkpoint']; assert digest(path)==run['checkpoint_sha256']
        model=StackedDecoder(16,32,2).double().eval(); model.load_state_dict(torch.load(path,weights_only=True))
        conditions={}; error_max=0.
        with torch.no_grad():
            for shift in shifts:
                base,cache=forward(model,x,shift)
                err=(base-logits(model,x,(torch.arange(8)+shift).expand_as(x))).abs().max().item(); error_max=max(error_max,err); assert err<1e-10
                layer=model.blocks[0].self_attn; z=model.blocks[0].norm1(cache['r0'])
                q,k,v=[t.reshape(len(x),8,2,8).transpose(1,2) for t in F.linear(z,layer.in_proj_weight,layer.in_proj_bias).chunk(3,-1)]
                mask=torch.ones(8,8,dtype=torch.bool).triu(1)
                w=(q@k.transpose(-2,-1)/(8**.5)).masked_fill(mask,float('-inf')).softmax(-1)
                for row in (3,5): w[:,:,row]=0; w[:,:,row,row-1]=1
                assert (w.sum(-1)-1).abs().max()<1e-12 and w.masked_select(mask[None,None].expand_as(w)).abs().max()==0
                a=F.linear((w@v).transpose(1,2).reshape(len(x),8,16),layer.out_proj.weight,layer.out_proj.bias)
                patched,pc=forward(model,x,shift,{'a0':a},('a0',))
                before=base.argmax(-1)==y
                for mode,s,c in [('original',base,cache),('pair',patched,pc)]:
                    assert torch.isfinite(s).all()
                    q,k,v=project(model,c['r1']); err=(attention(model,q,k,v)-c['a1']).abs().max().item(); error_max=max(error_max,err); assert err<1e-10
                    probs=((q[:,7].reshape(-1,2,8)[:,:,None,:]*k.reshape(-1,8,2,8).transpose(1,2)).sum(-1)/(8**.5)).softmax(-1)
                    mass=probs.gather(-1,(2*rank+1)[:,None,None].expand(-1,2,1)).squeeze(-1)
                    correct=s.argmax(-1)==y; losses=F.cross_entropy(s,y,reduction='none')
                    stats={}
                    for name,sel in [('primary',primary),('all_eligible',torch.ones(len(x),dtype=torch.bool))]:
                        stats[name]=dict(count=int(sel.sum()),accuracy=correct[sel].double().mean().item(),loss=losses[sel].mean().item(),
                            every_order_query_correct=correct[sel].reshape(-1,18).all(1).double().mean().item(),target_attention=mass[sel].mean(0).tolist(),
                            rescued=int((correct&~before&sel).sum()),harmed=int((~correct&before&sel).sum()))
                    conditions[f'{shift}_{mode}']=dict(groups=stats,predictions=s.argmax(-1).tolist())
        assert digest(path)==run['checkpoint_sha256']
        report['runs'].append(dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(path),reconstruction_error=error_max,conditions=conditions))
    report['criteria']={str(s):dict(every_seed_90=all(r['conditions'][f'{s}_pair']['groups']['primary']['accuracy']>=.9 for r in report['runs']),
        mean_gain=statistics.mean(r['conditions'][f'{s}_pair']['groups']['primary']['accuracy']-r['conditions'][f'{s}_original']['groups']['primary']['accuracy'] for r in report['runs'])) for s in shifts if s}
    for c in report['criteria'].values(): c['gain_at_least_5']=c['mean_gain']>=.05
    report['all_positive_shifts_pass']=all(c['every_seed_90'] and c['gain_at_least_5'] for c in report['criteria'].values())
    report['compact_preserved']=all(r['conditions']['0_original']['groups']['primary']['accuracy']-r['conditions']['0_pair']['groups']['primary']['accuracy']<=.01 for r in report['runs'])
    output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report['criteria']))


if __name__=='__main__': main()
