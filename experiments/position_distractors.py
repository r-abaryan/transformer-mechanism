"""Paired position and distractor interventions on frozen deep models."""
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from compare_depth import StackedDecoder
from train_associative_recall import tensors

ROOT = Path(__file__).resolve().parents[1]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pe(positions, width):
    angles = positions.double()[...,None] * 10000**(-torch.arange(0,width,2,dtype=torch.float64)/width)
    return torch.stack((angles.sin(),angles.cos()),-1).flatten(-2)


def logits(model, x, positions=None):
    pieces=[]
    with torch.no_grad():
        for start in range(0,len(x),256):
            chunk=x[start:start+256]
            handle=None
            if positions is not None:
                delta=pe(positions[start:start+256],model.embedding.embedding_dim)-pe(torch.arange(chunk.shape[1]),model.embedding.embedding_dim)
                handle=model.embedding.register_forward_hook(lambda module,args,output:output+delta)
            try:
                pieces.append(model(chunk)[:,-1])
            finally:
                if handle is not None:
                    handle.remove()
    result=torch.cat(pieces)
    assert torch.isfinite(result).all()
    return result


def variants(rows):
    x,y=tensors(rows)
    positions=torch.arange(x.shape[1]).expand_as(x).clone()
    query=positions.clone(); query[:,-1]=7
    tail=query.clone(); tail[:,-2]=6
    cropped,indices=[],[]
    for ids,target in rows:
        count=(len(ids)-2)//2
        chosen=ids[:-2:2].index(ids[-1])
        retained=sorted([chosen]+[i for i in range(count) if i!=chosen][:2])
        idx=[p for i in retained for p in (2*i,2*i+1)]+[len(ids)-2,len(ids)-1]
        row=[ids[i] for i in idx]
        assert dict(zip(row[:-2:2],row[1:-2:2]))[row[-1]]==target
        cropped.append(row); indices.append(idx)
    crop=torch.tensor(cropped); gap=torch.tensor(indices)
    gap_tail=gap.clone(); gap_tail[:,-2:]=torch.tensor([6,7])
    return y,dict(original=(x,None),query_reset=(x,query),tail_reset=(x,tail),
                  crop_original_positions=(crop,gap),crop_tail_reset=(crop,gap_tail),crop_compact=(crop,None))


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    path=ROOT/'results/position_distractors.json'
    assert not path.exists(), 'Preserve existing results'
    basepath=ROOT/'results/depth_confirmation.json'
    base=json.loads(basepath.read_text())
    splitpath=ROOT/base['split_file']
    assert digest(splitpath)==base['split_sha256']
    splits=json.loads(splitpath.read_text())
    report=dict(baseline_sha256=digest(basepath),split_sha256=digest(splitpath),
                protocol_sha256=digest(ROOT/'docs/position-distractor-protocol.md'),
                source_sha256={n:digest(ROOT/'experiments'/n) for n in ('position_distractors.py','compare_depth.py','train_associative_recall.py')},
                torch=torch.__version__,device='cpu',dtype='float64',runs=[])
    for run in base['runs']:
        if run['model']!='deep': continue
        checkpoint=ROOT/run['checkpoint']
        assert digest(checkpoint)==run['checkpoint_sha256']
        model=StackedDecoder(run['width'],run['hidden'],run['depth']).double().eval()
        model.load_state_dict(torch.load(checkpoint,weights_only=True))
        result=dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(checkpoint),lengths={})
        for name in ('four_pairs','six_pairs'):
            y,cases=variants(splits[name])
            x=cases['original'][0]
            original=logits(model,x)
            sham=logits(model,x,torch.arange(x.shape[1]).expand_as(x))
            error=(original-sham).abs().max().item()
            assert error<1e-10
            correct=original.argmax(-1)==y
            conditions={}
            for label,(ids,pos) in cases.items():
                scores=original if label=='original' else logits(model,ids,pos)
                now=scores.argmax(-1)==y
                conditions[label]=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(scores,y).item(),
                                       rescued=int((now&~correct).sum()),harmed=int((~now&correct).sum()))
            for key in ('accuracy','loss'):
                assert abs(conditions['original'][key]-run['metrics'][name][key])<1e-12
            result['lengths'][name]=dict(sham_max_logit_error=error,conditions=conditions)
        assert digest(checkpoint)==run['checkpoint_sha256']
        report['runs'].append(result)
    report['all_seeds_at_least_90']={n:{c:all(r['lengths'][n]['conditions'][c]['accuracy']>=.9 for r in report['runs'])
                                         for c in report['runs'][0]['lengths'][n]['conditions']} for n in ('four_pairs','six_pairs')}
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['all_seeds_at_least_90']))


if __name__=='__main__':
    main()
