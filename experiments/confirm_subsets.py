"""Fresh long examples, exhaustive query-preserving three-pair crops."""
import itertools
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from compare_depth import StackedDecoder
from position_distractors import digest, logits
from train_associative_recall import datasets, generate, tensors
from confirm_depth import identity

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output=ROOT/'results/subset_confirmation.json'
    splitfile=ROOT/'results/subset_confirmation_splits.json'
    assert not output.exists() and not splitfile.exists()
    previous=json.loads((ROOT/'results/depth_confirmation_splits.json').read_text())
    original=datasets()
    splits={}
    exclusions={}
    for name,n,seed in [('four_pairs',4,20261121),('six_pairs',6,20261122)]:
        excluded={identity(r) for r in original[name]+previous[name]}
        rows,ids=generate(seed,512,n,excluded)
        assert not ids&excluded and len(ids)==512
        splits[name]=rows
        exclusions[name]=dict(excluded=len(excluded),overlap=0)
    splitfile.write_text(json.dumps(splits,separators=(',',':'))+'\n')
    baseline=ROOT/'results/depth_confirmation.json'
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',split_sha256=digest(splitfile),
                baseline_sha256=digest(baseline),exclusions=exclusions,
                protocol_sha256=digest(ROOT/'docs/subset-confirmation-protocol.md'),
                source_sha256={n:digest(ROOT/'experiments'/n) for n in
                               ('confirm_subsets.py','position_distractors.py','compare_depth.py','train_associative_recall.py','confirm_depth.py')},runs=[])
    for run in json.loads(baseline.read_text())['runs']:
        if run['model']!='deep': continue
        checkpoint=ROOT/run['checkpoint']
        assert digest(checkpoint)==run['checkpoint_sha256']
        model=StackedDecoder(16,32,2).double().eval()
        model.load_state_dict(torch.load(checkpoint,weights_only=True))
        result=dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(checkpoint),lengths={})
        for name,rows in splits.items():
            x,y=tensors(rows)
            base=logits(model,x)
            error=(base-logits(model,x,torch.arange(x.shape[1]).expand_as(x))).abs().max().item()
            assert error<1e-10
            correct=base.argmax(-1)==y
            crops,positions=[],[]
            for ids,target in rows:
                n=(len(ids)-2)//2
                q=ids[:-2:2].index(ids[-1])
                subsets=list(itertools.combinations([i for i in range(n) if i!=q],2))
                assert len(set(subsets))==len(subsets)
                for others in subsets:
                    kept=sorted((q,*others))
                    idx=[p for i in kept for p in (2*i,2*i+1)]+[len(ids)-2,len(ids)-1]
                    crop=[ids[p] for p in idx]
                    assert dict(zip(crop[:-2:2],crop[1:-2:2]))[crop[-1]]==target
                    crops.append(crop); positions.append(idx)
            k=len(subsets)
            assert k in (3,10) and len(crops)==len(rows)*k
            crop=torch.tensor(crops); pos=torch.tensor(positions)
            reset=pos.clone(); reset[:,-2:]=torch.tensor([6,7])
            targets=y.repeat_interleave(k)
            conditions={}
            for label,p in [('original_positions',pos),('tail_reset',reset),('compact',None)]:
                scores=logits(model,crop,p)
                now=(scores.argmax(-1)==targets).reshape(len(rows),k)
                conditions[label]=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(scores,targets).item(),
                    all_subsets_correct=now.all(1).double().mean().item(),first_subset_accuracy=now[:,0].double().mean().item(),
                    rescued=int((now&~correct[:,None]).sum()),harmed=int((~now&correct[:,None]).sum()),
                    correct_subsets_per_episode=now.sum(1).tolist())
            result['lengths'][name]=dict(original_accuracy=correct.double().mean().item(),original_loss=F.cross_entropy(base,y).item(),
                                        sham_error=error,subsets_per_episode=k,conditions=conditions)
        assert digest(checkpoint)==run['checkpoint_sha256']
        report['runs'].append(result)
    report['criteria']={n:{metric:all(r['lengths'][n]['conditions']['tail_reset'][metric]>=.9 for r in report['runs'])
                          for metric in ('accuracy','all_subsets_correct')} for n in splits}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['criteria']))


if __name__=='__main__':
    main()
