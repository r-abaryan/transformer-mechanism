"""Explicit block-2 attention with matched/unrelated Q/K/V interventions."""
import itertools
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from activation_paths import forward
from compare_depth import StackedDecoder
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def project(model,residual):
    block=model.blocks[1]
    z=block.norm1(residual)
    return F.linear(z,block.self_attn.in_proj_weight,block.self_attn.in_proj_bias).chunk(3,-1)


def attention(model,q,k,v):
    b,n,d=q.shape
    q,k,v=[t.reshape(b,n,2,8).transpose(1,2) for t in (q,k,v)]
    mask=torch.ones(n,n,dtype=torch.bool).triu(1)
    a=((q@k.transpose(-2,-1))/math.sqrt(8)).masked_fill(mask,float('-inf')).softmax(-1)
    assert a.masked_select(mask[None,None].expand_as(a)).abs().max()==0
    assert (a.sum(-1)-1).abs().max()<1e-12
    merged=(a@v).transpose(1,2).reshape(b,n,d)
    layer=model.blocks[1].self_attn.out_proj
    return F.linear(merged,layer.weight,layer.bias)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output=ROOT/'results/qkv_paths.json'
    assert not output.exists()
    priorpath=ROOT/'results/donor_controls.json'
    prior=json.loads(priorpath.read_text())
    splitpath=ROOT/'results/subset_confirmation_splits.json'
    assert digest(splitpath)==prior['split_sha256']
    crops,labels=[],[]
    for ids,target in json.loads(splitpath.read_text())['six_pairs']:
        q=ids[:-2:2].index(ids[-1])
        for others in itertools.combinations([i for i in range(6) if i!=q],2):
            crop=[ids[p] for i in sorted((q,*others)) for p in (2*i,2*i+1)]+ids[-2:]
            assert dict(zip(crop[:-2:2],crop[1:-2:2]))[crop[-1]]==target
            crops.append(crop); labels.append(target)
    x=torch.tensor(crops); y=torch.tensor(labels)
    donors={'matched':torch.arange(len(x)),**{n:torch.tensor(v) for n,v in prior['donor_indices'].items()}}
    replacements={'Q':(0,),'K':(1,),'QK':(0,1),'V':(2,),'QKV':(0,1,2)}
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',prior_sha256=digest(priorpath),split_sha256=digest(splitpath),
        protocol_sha256=digest(ROOT/'docs/qkv-protocol.md'),
        source_sha256={n:digest(ROOT/'experiments'/n) for n in ('qkv_paths.py','activation_paths.py','compare_depth.py','position_distractors.py')},runs=[])
    for run in prior['runs']:
        path=ROOT/run['checkpoint']
        assert digest(path)==run['checkpoint_sha256']
        model=StackedDecoder(16,32,2).double().eval()
        model.load_state_dict(torch.load(path,weights_only=True))
        caches={}; baseline={}; errors=dict(attention=0.,sham=0.)
        with torch.no_grad():
            for name,shift in [('compact',0),('shifted',4)]:
                tensors=[[],[],[]]; scores=[]
                for chunk in x.split(256):
                    s,cache=forward(model,chunk,shift)
                    qkv=project(model,cache['r1'])
                    reconstructed=attention(model,*qkv)
                    error=(reconstructed-cache['a1']).abs().max().item()
                    errors['attention']=max(errors['attention'],error); assert error<1e-10
                    for parts,t in zip(tensors,qkv): parts.append(t)
                    scores.append(s)
                    if name=='shifted':
                        sham=forward(model,chunk,4,{'a1':reconstructed},('a1',))[0]
                        error=(s-sham).abs().max().item()
                        errors['sham']=max(errors['sham'],error); assert error<1e-10
                caches[name]=tuple(torch.cat(t) for t in tensors)
                baseline[name]=torch.cat(scores)
            correct=baseline['shifted'].argmax(-1)==y
            conditions={}
            for source,idx in donors.items():
                for name,replace in replacements.items():
                    outputs=[]
                    for start in range(0,len(x),256):
                        end=start+256
                        qkv=[caches['compact'][i][idx[start:end]] if i in replace else caches['shifted'][i][start:end] for i in range(3)]
                        a=attention(model,*qkv)
                        outputs.append(forward(model,x[start:end],4,{'a1':a},('a1',))[0])
                    s=torch.cat(outputs); assert torch.isfinite(s).all()
                    pred=s.argmax(-1); now=pred==y
                    conditions[f'{source}_{name}']=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(s,y).item(),
                        donor_answer_agreement=(pred==y[idx]).double().mean().item(),rescued=int((now&~correct).sum()),harmed=int((~now&correct).sum()))
                    if name=='QKV':
                        old=run['conditions']['rescue' if source=='matched' else source]
                        for metric in ('accuracy','loss','donor_answer_agreement'):
                            assert abs(conditions[f'{source}_{name}'][metric]-old[metric])<1e-10
            references={n:dict(accuracy=(s.argmax(-1)==y).double().mean().item(),loss=F.cross_entropy(s,y).item()) for n,s in baseline.items()}
            for n in references:
                for metric in references[n]: assert abs(references[n][metric]-run['conditions'][n][metric])<1e-12
        assert digest(path)==run['checkpoint_sha256']
        report['runs'].append(dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(path),errors=errors,references=references,conditions=conditions))
    report['all_seeds_accuracy_90']={n:all(r['conditions'][n]['accuracy']>=.9 for r in report['runs']) for n in conditions}
    report['different_answer_agreement_90']={n:all(r['conditions'][f'different_answer_{n}']['donor_answer_agreement']>=.9 for r in report['runs']) for n in replacements}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['all_seeds_accuracy_90']))


if __name__=='__main__':
    main()
