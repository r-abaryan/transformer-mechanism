"""Reverse and answer-controlled unrelated second-attention donors."""
import itertools
import json
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
    output=ROOT/'results/donor_controls.json'
    assert not output.exists()
    priorpath=ROOT/'results/activation_paths.json'
    prior=json.loads(priorpath.read_text())
    splitpath=ROOT/'results/subset_confirmation_splits.json'
    assert digest(splitpath)==prior['split_sha256']
    rows=json.loads(splitpath.read_text())['six_pairs']
    crops,labels=[],[]
    for ids,target in rows:
        q=ids[:-2:2].index(ids[-1])
        for others in itertools.combinations([i for i in range(6) if i!=q],2):
            crop=[ids[p] for i in sorted((q,*others)) for p in (2*i,2*i+1)]+ids[-2:]
            assert dict(zip(crop[:-2:2],crop[1:-2:2]))[crop[-1]]==target
            crops.append(crop); labels.append(target)
    maps=[tuple(sorted(zip(c[:-2:2],c[1:-2:2]))) for c in crops]
    donors={}
    for mode in ('same_answer','different_answer'):
        chosen=[]
        for i,crop in enumerate(crops):
            start=(i//10+1)*10
            candidate=next(j for offset in range(len(crops)) if
                (j:=(start+offset)%len(crops))//10!=i//10 and crops[j][-1]!=crop[-1] and maps[j]!=maps[i]
                and ((labels[j]==labels[i])==(mode=='same_answer')))
            chosen.append(candidate)
        donors[mode]=chosen
    x=torch.tensor(crops); y=torch.tensor(labels)
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',prior_sha256=digest(priorpath),split_sha256=digest(splitpath),donor_indices=donors,
                protocol_sha256=digest(ROOT/'docs/donor-controls-protocol.md'),
                source_sha256={n:digest(ROOT/'experiments'/n) for n in ('donor_controls.py','activation_paths.py','compare_depth.py','position_distractors.py')},runs=[])
    for run in prior['runs']:
        checkpoint=ROOT/run['checkpoint']
        assert digest(checkpoint)==run['checkpoint_sha256']
        model=StackedDecoder(16,32,2).double().eval()
        model.load_state_dict(torch.load(checkpoint,weights_only=True))
        cache={}; scores={}
        with torch.no_grad():
            for name,shift in [('compact',0),('shifted',4)]:
                outputs=[]; attention=[]
                for chunk in x.split(256):
                    s,c=forward(model,chunk,shift)
                    outputs.append(s); attention.append(c['a1'])
                scores[name]=torch.cat(outputs); cache[name]=torch.cat(attention)
            sham=0.
            cases=[('rescue',4,'compact',None),('reverse',0,'shifted',None),
                   ('same_answer',4,'compact',donors['same_answer']),('different_answer',4,'compact',donors['different_answer']),
                   ('compact_sham',0,'compact',None),('shifted_sham',4,'shifted',None)]
            for name,shift,source,indices in cases:
                donor=cache[source] if indices is None else cache[source][torch.tensor(indices)]
                outputs=[]
                for start in range(0,len(x),256):
                    s,_=forward(model,x[start:start+256],shift,{'a1':donor[start:start+256]},('a1',))
                    outputs.append(s)
                s=torch.cat(outputs)
                if name.endswith('_sham'):
                    error=(s-scores[source]).abs().max().item()
                    sham=max(sham,error); assert error<1e-10
                else: scores[name]=s
        conditions={}
        for name,s in scores.items():
            assert torch.isfinite(s).all()
            base='compact' if name in ('compact','reverse') else 'shifted'
            before=scores[base].argmax(-1)==y
            pred=s.argmax(-1); now=pred==y
            donor_labels=y[torch.tensor(donors[name])] if name in donors else y
            conditions[name]=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(s,y).item(),
                donor_answer_agreement=(pred==donor_labels).double().mean().item(),
                rescued=int((now&~before).sum()),harmed=int((~now&before).sum()),reference=base)
        for name,old in [('compact','compact'),('shifted','shifted'),('rescue','a1')]:
            for m in ('accuracy','loss'): assert abs(conditions[name][m]-run['conditions'][old][m])<1e-12
        assert digest(checkpoint)==run['checkpoint_sha256']
        report['runs'].append(dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(checkpoint),sham_error=sham,conditions=conditions))
    report['criteria']=dict(reverse_drop_10_every_seed=all(r['conditions']['compact']['accuracy']-r['conditions']['reverse']['accuracy']>=.1 for r in report['runs']),
        rescue_minus_same_answer_mean=statistics.mean(r['conditions']['rescue']['accuracy']-r['conditions']['same_answer']['accuracy'] for r in report['runs']),
        different_answer_agreement_90_every_seed=all(r['conditions']['different_answer']['donor_answer_agreement']>=.9 for r in report['runs']))
    report['criteria']['rescue_advantage_10']=report['criteria']['rescue_minus_same_answer_mean']>=.1
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['criteria']))


if __name__=='__main__':
    main()
