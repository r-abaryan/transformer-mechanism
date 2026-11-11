"""Frozen compact crops: global versus pair-only shifts."""
import itertools
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from compare_depth import StackedDecoder
from position_distractors import digest, logits

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output=ROOT/'results/position_shifts.json'
    assert not output.exists()
    priorpath=ROOT/'results/subset_confirmation.json'
    prior=json.loads(priorpath.read_text())
    splitpath=ROOT/'results/subset_confirmation_splits.json'
    assert digest(splitpath)==prior['split_sha256']
    splits=json.loads(splitpath.read_text())
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',prior_sha256=digest(priorpath),split_sha256=digest(splitpath),
                protocol_sha256=digest(ROOT/'docs/shift-protocol.md'),
                source_sha256={n:digest(ROOT/'experiments'/n) for n in ('position_shifts.py','position_distractors.py','compare_depth.py')},runs=[])
    for run in prior['runs']:
        checkpoint=ROOT/run['checkpoint']
        assert digest(checkpoint)==run['checkpoint_sha256']
        model=StackedDecoder(16,32,2).double().eval()
        model.load_state_dict(torch.load(checkpoint,weights_only=True))
        result=dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(checkpoint),lengths={})
        for name,rows in splits.items():
            crops,labels=[],[]
            for ids,target in rows:
                n=(len(ids)-2)//2
                q=ids[:-2:2].index(ids[-1])
                subsets=list(itertools.combinations([i for i in range(n) if i!=q],2))
                for others in subsets:
                    kept=sorted((q,*others))
                    crop=[ids[p] for i in kept for p in (2*i,2*i+1)]+ids[-2:]
                    assert dict(zip(crop[:-2:2],crop[1:-2:2]))[crop[-1]]==target
                    crops.append(crop); labels.append(target)
            x=torch.tensor(crops); y=torch.tensor(labels); k=len(subsets)
            base=logits(model,x); prediction=base.argmax(-1); correct=prediction==y
            old=run['lengths'][name]['conditions']['compact']
            assert abs(correct.double().mean().item()-old['accuracy'])<1e-12
            assert abs(F.cross_entropy(base,y).item()-old['loss'])<1e-12
            cases={}
            compact=torch.arange(8)
            for shift in (0,1,2,4,6,8):
                for mode in ('global','pairs'):
                    pos=compact.clone()
                    if mode=='global':
                        pos+=shift
                        assert torch.equal(pos[:,None]-pos[None,:],compact[:,None]-compact[None,:])
                    else:
                        pos[:6]+=shift
                        assert torch.equal(pos[:6]-pos[7],compact[:6]-compact[7]+shift)
                    scores=logits(model,x,pos.expand_as(x))
                    changed=scores.argmax(-1); now=changed==y
                    error=(scores-base).abs().max().item()
                    if shift==0: assert error<1e-10
                    cases[f'{mode}_{shift}']=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(scores,y).item(),
                        all_subsets_correct=now.reshape(len(rows),k).all(1).double().mean().item(),
                        max_logit_change=error,changed_predictions=int((changed!=prediction).sum()),
                        rescued=int((now&~correct).sum()),harmed=int((~now&correct).sum()))
            result['lengths'][name]=cases
        assert digest(checkpoint)==run['checkpoint_sha256']
        report['runs'].append(result)
    report['criteria']={n:{str(s):dict(global_all_seeds_90=all(r['lengths'][n][f'global_{s}']['accuracy']>=.9 for r in report['runs']),
                                     global_logits_invariant=all(r['lengths'][n][f'global_{s}']['max_logit_change']<=1e-10 for r in report['runs']))
                          for s in (1,2,4,6,8)} for n in splits}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['criteria']))


if __name__=='__main__':
    main()
