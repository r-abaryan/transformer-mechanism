"""First value-row forcing with both later-row backgrounds."""
import itertools
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from activation_paths import forward
from compare_depth import StackedDecoder
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    output=ROOT/'results/first_value.json'; assert not output.exists()
    priorpath=ROOT/'results/direct_pair.json'; prior=json.loads(priorpath.read_text())
    splitpath=ROOT/'results/subset_confirmation_splits.json'; assert digest(splitpath)==prior['split_sha256']
    crops,labels=[],[]
    for ids,target in json.loads(splitpath.read_text())['six_pairs']:
        q=ids[:-2:2].index(ids[-1])
        for others in itertools.combinations([i for i in range(6) if i!=q],2):
            crop=[ids[p] for i in sorted((q,*others)) for p in (2*i,2*i+1)]+ids[-2:]
            assert dict(zip(crop[:-2:2],crop[1:-2:2]))[crop[-1]]==target
            crops.append(crop); labels.append(target)
    x=torch.tensor(crops); y=torch.tensor(labels)
    ranks=(x[:,:6:2]==x[:,-1,None]).long().argmax(1)
    specs={'original':{},'later_pair':{3:2,5:4},'first_pair':{1:0},'first_self':{1:1},'all_pair':{1:0,3:2,5:4},'first_self_later_pair':{1:1,3:2,5:4}}
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',prior_sha256=digest(priorpath),split_sha256=digest(splitpath),
        protocol_sha256=digest(ROOT/'docs/first-value-protocol.md'),
        source_sha256={n:digest(ROOT/'experiments'/n) for n in ('first_value.py','activation_paths.py','compare_depth.py','position_distractors.py')},runs=[])
    for run in prior['runs']:
        path=ROOT/run['checkpoint']; assert digest(path)==run['checkpoint_sha256']
        model=StackedDecoder(16,32,2).double().eval(); model.load_state_dict(torch.load(path,weights_only=True))
        pieces={f'{s}_{m}':[] for s in (0,4) for m in specs}; error_max=0.
        with torch.no_grad():
            for shift in (0,4):
                for chunk in x.split(256):
                    base,cache=forward(model,chunk,shift)
                    layer=model.blocks[0].self_attn
                    z=model.blocks[0].norm1(cache['r0'])
                    q,k,v=[t.reshape(len(chunk),8,2,8).transpose(1,2) for t in F.linear(z,layer.in_proj_weight,layer.in_proj_bias).chunk(3,-1)]
                    mask=torch.ones(8,8,dtype=torch.bool).triu(1)
                    weights=(q@k.transpose(-2,-1)/(8**.5)).masked_fill(mask,float('-inf')).softmax(-1)
                    for mode,patch in specs.items():
                        w=weights.clone()
                        for row,col in patch.items(): w[:,:,row]=0; w[:,:,row,col]=1
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
            reference=name.split('_')[0]+'_original'
            before=scores[reference].argmax(-1)==y; now=s.argmax(-1)==y
            conditions[name]=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(s,y).item(),reference=reference,
                rescued=int((now&~before).sum()),harmed=int((~now&before).sum()),
                query_rank_accuracy=[now[ranks==i].double().mean().item() for i in range(3)])
        for shift in (0,4):
            for new,old in [('original','original'),('later_pair','pair')]:
                for m in ('accuracy','loss'): assert abs(conditions[f'{shift}_{new}'][m]-run['conditions'][f'{shift}_{old}'][m])<1e-10
        assert digest(path)==run['checkpoint_sha256']
        report['runs'].append(dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(path),reconstruction_error=error_max,conditions=conditions))
    report['criteria']=dict(all_pair_shifted_every_seed_90=all(r['conditions']['4_all_pair']['accuracy']>=.9 for r in report['runs']),
        compact_loss_at_most_one_point_every_seed=all(r['conditions']['0_original']['accuracy']-r['conditions']['0_all_pair']['accuracy']<=.01 for r in report['runs']))
    report['query_rank_counts']=[int((ranks==i).sum()) for i in range(3)]
    output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report['criteria']))


if __name__=='__main__': main()
