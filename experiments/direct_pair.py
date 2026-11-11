"""Direct block-1 source forcing with normal downstream execution."""
import itertools
import json
import math
import statistics
from pathlib import Path
import torch
from torch.nn import functional as F
from activation_paths import forward
from compare_depth import StackedDecoder
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output=ROOT/'results/direct_pair.json'; assert not output.exists()
    priorpath=ROOT/'results/position_shifts.json'; prior=json.loads(priorpath.read_text())
    splitpath=ROOT/'results/subset_confirmation_splits.json'; assert digest(splitpath)==prior['split_sha256']
    crops,labels=[],[]
    for ids,target in json.loads(splitpath.read_text())['six_pairs']:
        q=ids[:-2:2].index(ids[-1])
        for others in itertools.combinations([i for i in range(6) if i!=q],2):
            crop=[ids[p] for i in sorted((q,*others)) for p in (2*i,2*i+1)]+ids[-2:]
            assert dict(zip(crop[:-2:2],crop[1:-2:2]))[crop[-1]]==target
            crops.append(crop); labels.append(target)
    x=torch.tensor(crops); y=torch.tensor(labels)
    modes=('original','pair','self','other_key')
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',prior_sha256=digest(priorpath),split_sha256=digest(splitpath),
        protocol_sha256=digest(ROOT/'docs/direct-pair-protocol.md'),
        source_sha256={n:digest(ROOT/'experiments'/n) for n in ('direct_pair.py','activation_paths.py','compare_depth.py','position_distractors.py')},runs=[])
    for run in prior['runs']:
        path=ROOT/run['checkpoint']; assert digest(path)==run['checkpoint_sha256']
        model=StackedDecoder(16,32,2).double().eval(); model.load_state_dict(torch.load(path,weights_only=True))
        pieces={f'{s}_{m}':[] for s in (0,4) for m in modes}; error_max=0.
        with torch.no_grad():
            for shift in (0,4):
                for chunk in x.split(256):
                    base,cache=forward(model,chunk,shift)
                    layer=model.blocks[0].self_attn
                    z=model.blocks[0].norm1(cache['r0'])
                    q,k,v=[t.reshape(len(chunk),8,2,8).transpose(1,2) for t in F.linear(z,layer.in_proj_weight,layer.in_proj_bias).chunk(3,-1)]
                    mask=torch.ones(8,8,dtype=torch.bool).triu(1)
                    weights=(q@k.transpose(-2,-1)/math.sqrt(8)).masked_fill(mask,float('-inf')).softmax(-1)
                    for mode in modes:
                        aweights=weights.clone()
                        if mode!='original':
                            for row in (3,5):
                                col={'pair':row-1,'self':row,'other_key':row-3}[mode]
                                aweights[:,:,row]=0; aweights[:,:,row,col]=1
                        assert (aweights.sum(-1)-1).abs().max()<1e-12
                        assert aweights.masked_select(mask[None,None].expand_as(aweights)).abs().max()==0
                        a=F.linear((aweights@v).transpose(1,2).reshape(len(chunk),8,16),layer.out_proj.weight,layer.out_proj.bias)
                        logits,_=forward(model,chunk,shift,{'a0':a},('a0',))
                        if mode=='original':
                            err=max((a-cache['a0']).abs().max().item(),(logits-base).abs().max().item())
                            error_max=max(error_max,err); assert err<1e-10
                        pieces[f'{shift}_{mode}'].append(logits)
        scores={n:torch.cat(v) for n,v in pieces.items()}; conditions={}
        for name,s in scores.items():
            assert torch.isfinite(s).all()
            reference=name.split('_')[0]+'_original'
            before=scores[reference].argmax(-1)==y; now=s.argmax(-1)==y
            conditions[name]=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(s,y).item(),
                                 reference=reference,rescued=int((now&~before).sum()),harmed=int((~now&before).sum()))
        for shift in (0,4):
            for m in ('accuracy','loss'): assert abs(conditions[f'{shift}_original'][m]-run['lengths']['six_pairs'][f'global_{shift}'][m])<1e-10
        assert digest(path)==run['checkpoint_sha256']
        report['runs'].append(dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(path),reconstruction_error=error_max,conditions=conditions))
    gains={m:statistics.mean(r['conditions']['4_pair']['accuracy']-r['conditions'][f'4_{m}']['accuracy'] for r in report['runs']) for m in ('self','other_key')}
    report['criteria']=dict(shifted_pair_every_seed_90=all(r['conditions']['4_pair']['accuracy']>=.9 for r in report['runs']),
        control_advantages=gains,each_control_advantage_10=all(v>=.1 for v in gains.values()),
        compact_loss_at_most_one_point_every_seed=all(r['conditions']['0_original']['accuracy']-r['conditions']['0_pair']['accuracy']<=.01 for r in report['runs']))
    output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report['criteria']))


if __name__=='__main__':
    main()
