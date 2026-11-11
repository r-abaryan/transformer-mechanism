"""Construct block-2 K from mixed block-1 residual/attention/FFN components."""
import itertools
import json
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
    output=ROOT/'results/block1_keys.json'
    assert not output.exists()
    priorpath=ROOT/'results/target_distractor_keys.json'
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
    names=['frozen_'+''.join(map(str,b)) for b in itertools.product((0,1),repeat=3)]
    names+=['live_'+''.join(map(str,b)) for b in itertools.product((0,1),repeat=2)]
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',prior_sha256=digest(priorpath),split_sha256=digest(splitpath),
        protocol_sha256=digest(ROOT/'docs/block1-key-protocol.md'),
        source_sha256={n:digest(ROOT/'experiments'/n) for n in ('block1_keys.py','qkv_paths.py','activation_paths.py','compare_depth.py','position_distractors.py')},runs=[])
    for run in prior['runs']:
        path=ROOT/run['checkpoint']; assert digest(path)==run['checkpoint_sha256']
        model=StackedDecoder(16,32,2).double().eval()
        model.load_state_dict(torch.load(path,weights_only=True))
        pieces={n:[] for n in names}; reconstruction=0.
        with torch.no_grad():
            for chunk in x.split(256):
                _,cc=forward(model,chunk,0); _,sc=forward(model,chunk,4)
                states=[sc,cc]
                block=model.blocks[0]
                def ffn(u):
                    return block.linear2(block.activation(block.linear1(block.norm2(u))))
                for state in states:
                    error=(state['r0']+state['a0']+state['f0']-state['r1']).abs().max().item()
                    error=max(error,(ffn(state['r0']+state['a0'])-state['f0']).abs().max().item())
                    reconstruction=max(reconstruction,error); assert error<1e-10
                compact=project(model,cc['r1']); shifted=project(model,sc['r1'])
                for name in names:
                    mode,bits=name.split('_'); bits=list(map(int,bits))
                    u=states[bits[0]]['r0']+states[bits[1]]['a0']
                    f=states[bits[2]]['f0'] if mode=='frozen' else ffn(u)
                    candidate=u+f
                    candidate_k=project(model,candidate)[1]
                    q,k,v=[t.clone() for t in shifted]
                    q[:,7]=compact[0][:,7]
                    k[:,[1,3,5]]=candidate_k[:,[1,3,5]]
                    a=attention(model,q,k,v)
                    pieces[name].append(forward(model,chunk,4,{'a1':a},('a1',))[0])
        scores={n:torch.cat(v) for n,v in pieces.items()}
        correct=scores['frozen_000'].argmax(-1)==y
        conditions={}
        for name,s in scores.items():
            assert torch.isfinite(s).all()
            now=s.argmax(-1)==y
            conditions[name]=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(s,y).item(),
                                 rescued=int((now&~correct).sum()),harmed=int((~now&correct).sum()))
        for name,old in [('frozen_000','forward_none'),('frozen_111','forward_all')]:
            for m in ('accuracy','loss'): assert abs(conditions[name][m]-run['conditions'][old][m])<1e-10
        for a,b in [('frozen_000','live_00'),('frozen_111','live_11')]:
            error=(scores[a]-scores[b]).abs().max().item()
            reconstruction=max(reconstruction,error); assert error<1e-10
        assert digest(path)==run['checkpoint_sha256']
        report['runs'].append(dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(path),reconstruction_error=reconstruction,conditions=conditions))
    report['all_seeds_accuracy_90']={n:all(r['conditions'][n]['accuracy']>=.9 for r in report['runs']) for n in names}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['all_seeds_accuracy_90']))


if __name__=='__main__':
    main()
