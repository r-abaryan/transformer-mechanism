"""Full fixed-content offset/gap/tail positional factorial."""
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
    output=ROOT/'results/position_layout.json'
    assert not output.exists()
    priorpath=ROOT/'results/subset_confirmation.json'
    prior=json.loads(priorpath.read_text())
    splitpath=ROOT/'results/subset_confirmation_splits.json'
    assert digest(splitpath)==prior['split_sha256']
    splits=json.loads(splitpath.read_text())
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',prior_sha256=digest(priorpath),split_sha256=digest(splitpath),
                protocol_sha256=digest(ROOT/'docs/layout-protocol.md'),
                source_sha256={n:digest(ROOT/'experiments'/n) for n in ('position_layout.py','position_distractors.py','compare_depth.py')},runs=[])
    for run in prior['runs']:
        checkpoint=ROOT/run['checkpoint']
        assert digest(checkpoint)==run['checkpoint_sha256']
        model=StackedDecoder(16,32,2).double().eval()
        model.load_state_dict(torch.load(checkpoint,weights_only=True))
        result=dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(checkpoint),lengths={})
        for name,rows in splits.items():
            crops,indices,labels=[],[],[]
            for ids,target in rows:
                n=(len(ids)-2)//2
                q=ids[:-2:2].index(ids[-1])
                combinations=list(itertools.combinations([i for i in range(n) if i!=q],2))
                for others in combinations:
                    kept=sorted((q,*others))
                    idx=[p for i in kept for p in (2*i,2*i+1)]+[len(ids)-2,len(ids)-1]
                    crop=[ids[p] for p in idx]
                    assert dict(zip(crop[:-2:2],crop[1:-2:2]))[crop[-1]]==target
                    crops.append(crop); indices.append(idx); labels.append(target)
            k=len(combinations)
            x=torch.tensor(crops); original=torch.tensor(indices); y=torch.tensor(labels)
            compact=torch.arange(8).expand_as(original)
            offset=original[:,0:1]
            gaps=original[:,:6]-offset-compact[:,:6]
            assert (gaps>=0).all()
            cells={}; correctness={}; sham=0.
            for a,g,t in itertools.product((0,1),repeat=3):
                key=f'{a}{g}{t}'
                pos=compact.clone()
                pos[:,:6]+=a*offset+g*gaps
                if t: pos[:,-2:]=original[:,-2:]
                if key=='111': assert torch.equal(pos,original)
                scores=logits(model,x,pos)
                if key=='000':
                    sham=(scores-logits(model,x)).abs().max().item()
                    assert sham<1e-10
                now=scores.argmax(-1)==y
                correctness[key]=now
                cells[key]=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(scores,y).item(),
                                all_subsets_correct=now.reshape(len(rows),k).all(1).double().mean().item())
                if key in ('000','110','111'):
                    old=run['lengths'][name]['conditions'][{'000':'compact','110':'tail_reset','111':'original_positions'}[key]]
                    for metric in cells[key]: assert abs(cells[key][metric]-old[metric])<1e-12
            effects={}
            for axis,label in enumerate(('offset','gaps','tail')):
                for key in cells:
                    if key[axis]!='0': continue
                    on=key[:axis]+'1'+key[axis+1:]
                    before,after=correctness[key],correctness[on]
                    effects[f'{label}:{key}->{on}']=dict(accuracy_difference=cells[on]['accuracy']-cells[key]['accuracy'],
                                                        rescued=int((after&~before).sum()),harmed=int((~after&before).sum()))
            result['lengths'][name]=dict(cells=cells,effects=effects,sham_error=sham)
        assert digest(checkpoint)==run['checkpoint_sha256']
        report['runs'].append(result)
    report['all_seeds_accuracy_90']={n:{c:all(r['lengths'][n]['cells'][c]['accuracy']>=.9 for r in report['runs'])
                                       for c in report['runs'][0]['lengths'][n]['cells']} for n in splits}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['all_seeds_accuracy_90']))


if __name__=='__main__':
    main()
