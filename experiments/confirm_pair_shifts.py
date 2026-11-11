"""Fixed paired-key intervention on fresh parents across fixed shifts."""
import itertools
import json
import statistics
from pathlib import Path
import torch
from torch.nn import functional as F
from activation_paths import forward
from compare_depth import StackedDecoder
from position_distractors import digest,logits
from train_associative_recall import datasets,generate
from confirm_depth import identity

ROOT=Path(__file__).resolve().parents[1]


def crop_rows(rows):
    crops=[]
    for ids,target in rows:
        q=ids[:-2:2].index(ids[-1])
        for others in itertools.combinations([i for i in range(6) if i!=q],2):
            crop=[ids[p] for i in sorted((q,*others)) for p in (2*i,2*i+1)]+ids[-2:]
            assert dict(zip(crop[:-2:2],crop[1:-2:2]))[crop[-1]]==target
            crops.append((crop,target))
    return crops


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    output=ROOT/'results/pair_shift_confirmation.json'; splitpath=ROOT/'results/pair_shift_confirmation_parents.json'
    assert not output.exists() and not splitpath.exists()
    original=datasets()
    depaths=[ROOT/'results/depth_confirmation_splits.json',ROOT/'results/subset_confirmation_splits.json']
    depth,subset=[json.loads(p.read_text()) for p in depaths]
    excluded={identity(r) for r in original['six_pairs']+depth['six_pairs']+subset['six_pairs']}
    parents,ids=generate(20261201,512,6,excluded); assert not ids&excluded
    splitpath.write_text(json.dumps(parents,separators=(',',':'))+'\n')
    crops=crop_rows(parents); x=torch.tensor([r[0] for r in crops]); y=torch.tensor([r[1] for r in crops])
    cropids=[identity(r) for r in crops]
    oldcropids={identity(r) for r in crop_rows(subset['six_pairs'])}
    trainids={identity(r) for r in depth['train']}
    priorpath=ROOT/'results/direct_pair.json'; prior=json.loads(priorpath.read_text())
    shifts=(0,1,2,4,6,8); modes=('original','pair','self','other_key')
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',prior_sha256=digest(priorpath),split_sha256=digest(splitpath),
        previous_split_hashes={str(p.relative_to(ROOT)):digest(p) for p in depaths},
        overlap=dict(excluded_parent_count=len(excluded),parent_overlap=0,distinct_crops=len(set(cropids)),
                     crop_instances_in_previous_diagnostic=sum(i in oldcropids for i in cropids),
                     crop_instances_in_confirmation_train=sum(i in trainids for i in cropids)),
        protocol_sha256=digest(ROOT/'docs/pair-confirmation-protocol.md'),
        source_sha256={n:digest(ROOT/'experiments'/n) for n in ('confirm_pair_shifts.py','activation_paths.py','compare_depth.py','position_distractors.py','train_associative_recall.py','confirm_depth.py')},runs=[])
    for run in prior['runs']:
        path=ROOT/run['checkpoint']; assert digest(path)==run['checkpoint_sha256']
        model=StackedDecoder(16,32,2).double().eval(); model.load_state_dict(torch.load(path,weights_only=True))
        pieces={f'{s}_{m}':[] for s in shifts for m in modes}; error_max=0.
        with torch.no_grad():
            for shift in shifts:
                for chunk in x.split(256):
                    base,cache=forward(model,chunk,shift)
                    library=logits(model,chunk,(torch.arange(8)+shift).expand_as(chunk))
                    error=(base-library).abs().max().item(); error_max=max(error_max,error); assert error<1e-10
                    layer=model.blocks[0].self_attn; z=model.blocks[0].norm1(cache['r0'])
                    q,k,v=[t.reshape(len(chunk),8,2,8).transpose(1,2) for t in F.linear(z,layer.in_proj_weight,layer.in_proj_bias).chunk(3,-1)]
                    mask=torch.ones(8,8,dtype=torch.bool).triu(1)
                    weights=(q@k.transpose(-2,-1)/(8**.5)).masked_fill(mask,float('-inf')).softmax(-1)
                    for mode in modes:
                        w=weights.clone()
                        if mode!='original':
                            for row in (3,5):
                                col={'pair':row-1,'self':row,'other_key':row-3}[mode]
                                w[:,:,row]=0; w[:,:,row,col]=1
                        assert (w.sum(-1)-1).abs().max()<1e-12
                        assert w.masked_select(mask[None,None].expand_as(w)).abs().max()==0
                        a=F.linear((w@v).transpose(1,2).reshape(len(chunk),8,16),layer.out_proj.weight,layer.out_proj.bias)
                        s,_=forward(model,chunk,shift,{'a0':a},('a0',))
                        if mode=='original':
                            error=(s-base).abs().max().item(); error_max=max(error_max,error); assert error<1e-10
                        pieces[f'{shift}_{mode}'].append(s)
        scores={n:torch.cat(v) for n,v in pieces.items()}; conditions={}
        for name,s in scores.items():
            assert torch.isfinite(s).all()
            reference=name.split('_')[0]+'_original'; before=scores[reference].argmax(-1)==y; now=s.argmax(-1)==y
            conditions[name]=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(s,y).item(),reference=reference,
                every_crop_correct=now.reshape(512,10).all(1).double().mean().item(),rescued=int((now&~before).sum()),harmed=int((~now&before).sum()))
        assert digest(path)==run['checkpoint_sha256']
        report['runs'].append(dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(path),reconstruction_error=error_max,conditions=conditions))
        print(json.dumps(dict(seed=run['seed'],pair_accuracy={s:conditions[f'{s}_pair']['accuracy'] for s in shifts})),flush=True)
    report['criteria']={str(s):dict(every_seed_90=all(r['conditions'][f'{s}_pair']['accuracy']>=.9 for r in report['runs']),
        control_gains={m:statistics.mean(r['conditions'][f'{s}_pair']['accuracy']-r['conditions'][f'{s}_{m}']['accuracy'] for r in report['runs']) for m in ('self','other_key')}) for s in shifts if s}
    for c in report['criteria'].values(): c['each_control_gain_10']=all(v>=.1 for v in c['control_gains'].values())
    report['every_seed_every_positive_shift_90']=all(c['every_seed_90'] for c in report['criteria'].values())
    report['compact_preserved']=all(r['conditions']['0_original']['accuracy']-r['conditions']['0_pair']['accuracy']<=.01 for r in report['runs'])
    output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report['criteria']))


if __name__=='__main__': main()
