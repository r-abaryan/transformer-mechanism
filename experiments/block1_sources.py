"""Inspect and intervene on source rows in block-1 attention."""
import itertools
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from activation_paths import forward
from compare_depth import StackedDecoder
from qkv_paths import project,attention
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output=ROOT/'results/block1_sources.json'
    assert not output.exists()
    priorpath=ROOT/'results/block1_keys.json'; prior=json.loads(priorpath.read_text())
    splitpath=ROOT/'results/subset_confirmation_splits.json'; assert digest(splitpath)==prior['split_sha256']
    crops,labels=[],[]
    for ids,target in json.loads(splitpath.read_text())['six_pairs']:
        q=ids[:-2:2].index(ids[-1])
        for others in itertools.combinations([i for i in range(6) if i!=q],2):
            crop=[ids[p] for i in sorted((q,*others)) for p in (2*i,2*i+1)]+ids[-2:]
            assert dict(zip(crop[:-2:2],crop[1:-2:2]))[crop[-1]]==target
            crops.append(crop); labels.append(target)
    x=torch.tensor(crops); y=torch.tensor(labels)
    modes=('original','force_pair','force_self','force_other_key','remove_pair','remove_self')
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',prior_sha256=digest(priorpath),split_sha256=digest(splitpath),
        protocol_sha256=digest(ROOT/'docs/block1-source-protocol.md'),
        source_sha256={n:digest(ROOT/'experiments'/n) for n in ('block1_sources.py','qkv_paths.py','activation_paths.py','compare_depth.py','position_distractors.py')},runs=[])
    for run in prior['runs']:
        path=ROOT/run['checkpoint']; assert digest(path)==run['checkpoint_sha256']
        model=StackedDecoder(16,32,2).double().eval(); model.load_state_dict(torch.load(path,weights_only=True))
        pieces={f'{s}_{m}':[] for s in ('compact','shifted') for m in modes}
        observations={s:[] for s in ('compact','shifted')}; error_max=0.
        with torch.no_grad():
            for chunk in x.split(256):
                _,cc=forward(model,chunk,0); _,sc=forward(model,chunk,4)
                cqkv=project(model,cc['r1']); sqkv=project(model,sc['r1'])
                layer=model.blocks[0].self_attn
                mask=torch.ones(8,8,dtype=torch.bool).triu(1)
                for state,cache in [('compact',cc),('shifted',sc)]:
                    z=model.blocks[0].norm1(cache['r0'])
                    q,k,v=[t.reshape(len(chunk),8,2,8).transpose(1,2) for t in F.linear(z,layer.in_proj_weight,layer.in_proj_bias).chunk(3,-1)]
                    scores=(q@k.transpose(-2,-1)/math.sqrt(8)).masked_fill(mask,float('-inf'))
                    original=scores.softmax(-1)
                    obs=[]
                    for row in (1,3,5):
                        obs.append(torch.stack((original[:,:,row,row-1],original[:,:,row,row],original[:,:,row,list(range(0,row-1,2))].sum(-1)),-1))
                    observations[state].append(torch.stack(obs,2))
                    for mode in modes:
                        weights=original.clone()
                        for row in (3,5):
                            if mode.startswith('force_'):
                                col={'force_pair':row-1,'force_self':row,'force_other_key':row-3}[mode]
                                weights[:,:,row]=0; weights[:,:,row,col]=1
                            elif mode.startswith('remove_'):
                                col=row-1 if mode=='remove_pair' else row
                                altered=scores[:,:,row].clone(); altered[:,:,col]=float('-inf')
                                weights[:,:,row]=altered.softmax(-1)
                        assert (weights.sum(-1)-1).abs().max()<1e-12
                        assert weights.masked_select(mask[None,None].expand_as(weights)).abs().max()==0
                        mixed=(weights@v).transpose(1,2).reshape(len(chunk),8,16)
                        a=F.linear(mixed,layer.out_proj.weight,layer.out_proj.bias)
                        if mode=='original':
                            err=(a-cache['a0']).abs().max().item(); error_max=max(error_max,err); assert err<1e-10
                        candidate=sc['r0']+a+sc['f0']
                        newk=project(model,candidate)[1]
                        q2,k2,v2=[t.clone() for t in sqkv]
                        q2[:,7]=cqkv[0][:,7]; k2[:,[1,3,5]]=newk[:,[1,3,5]]
                        a2=attention(model,q2,k2,v2)
                        pieces[f'{state}_{mode}'].append(forward(model,chunk,4,{'a1':a2},('a1',))[0])
        scores={n:torch.cat(v) for n,v in pieces.items()}; conditions={}
        for name,s in scores.items():
            assert torch.isfinite(s).all()
            reference=name.split('_')[0]+'_original'
            before=scores[reference].argmax(-1)==y; now=s.argmax(-1)==y
            conditions[name]=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(s,y).item(),
                                 reference=reference,rescued=int((now&~before).sum()),harmed=int((~now&before).sum()))
        for name,old in [('compact_original','frozen_010'),('shifted_original','frozen_000')]:
            for m in ('accuracy','loss'): assert abs(conditions[name][m]-run['conditions'][old][m])<1e-10
        assert digest(path)==run['checkpoint_sha256']
        report['runs'].append(dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(path),reconstruction_error=error_max,
            conditions=conditions,attention_observations={n:torch.cat(v).mean(0).tolist() for n,v in observations.items()}))
    report['observation_axes']=['head','value_slot_1_3_5','paired_key_self_other_keys']
    report['all_seeds_accuracy_90']={n:all(r['conditions'][n]['accuracy']>=.9 for r in report['runs']) for n in conditions}
    report['removal_drop_10_every_seed']={f'{s}_{m}':all(r['conditions'][f'{s}_original']['accuracy']-r['conditions'][f'{s}_{m}']['accuracy']>=.1 for r in report['runs']) for s in ('compact','shifted') for m in ('remove_pair','remove_self')}
    output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report['all_seeds_accuracy_90']))


if __name__=='__main__':
    main()
