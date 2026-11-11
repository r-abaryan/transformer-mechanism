"""Fixed-budget associative recall with canonical split isolation."""
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import torch
from torch.nn import functional as F
from verify_decoder import TinyDecoder

ROOT=Path(__file__).resolve().parents[1]


def generate(seed,count,pairs,excluded):
    rng=random.Random(seed)
    rows=[]
    identities=set()
    while len(rows)<count:
        keys=rng.sample(range(8),pairs)
        mapping=list(zip(keys,[rng.randrange(8,12) for _ in keys]))
        query=rng.choice(keys)
        identity=(tuple(sorted(mapping)),query)
        if identity in excluded or identity in identities:
            continue
        identities.add(identity)
        ids=[item for pair in mapping for item in pair]+[12,query]
        rows.append((ids,dict(mapping)[query]))
    return rows,identities


def datasets():
    splits={}
    excluded=set()
    for name,seed,count,pairs in [('train',20261040,8192,3),('validation',20261041,1024,3),
                                  ('test',20261042,1024,3),('four_pairs',20261043,1024,4),
                                  ('six_pairs',20261044,1024,6)]:
        rows,identities=generate(seed,count,pairs,excluded)
        assert not identities&excluded
        excluded.update(identities)
        splits[name]=rows
    return splits


def tensors(rows):
    return torch.tensor([x for x,y in rows]),torch.tensor([y for x,y in rows])


def evaluate(model,data):
    model.eval()
    x,y=data
    with torch.no_grad():
        logits=torch.cat([model(chunk)[:,-1] for chunk in x.split(256)])
        prediction=logits.argmax(-1)
        answer=dict(accuracy=(prediction==y).double().mean().item(),loss=F.cross_entropy(logits,y).item(),
                    invalid_prediction_rate=((prediction<8)|(prediction>11)).double().mean().item())
    model.train()
    return answer


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    splits=datasets()
    data={name:tensors(rows) for name,rows in splits.items()}
    # Independent lookup and sequence contract checks before any training.
    for rows in splits.values():
        for ids,y in rows:
            keys,values=ids[:-2:2],ids[1:-2:2]
            assert len(keys)==len(set(keys)) and ids[-2]==12
            assert values[keys.index(ids[-1])]==y
    majority=max(range(8,12),key=lambda v:sum(y==v for _,y in splits['train']))
    baselines={name:dict(majority=sum(y==majority for _,y in rows)/len(rows),
                         first=sum(x[1]==y for x,y in rows)/len(rows),
                         last=sum(x[-3]==y for x,y in rows)/len(rows),oracle=1.0,uniform_expected=.25,
                         target_counts=[sum(y==v for _,y in rows) for v in range(8,12)],
                         query_position_counts=[sum(x[:-2:2].index(x[-1])==p for x,y in rows) for p in range((len(rows[0][0])-2)//2)])
               for name,rows in splits.items()}
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',steps=3000,batch=64,lr=.1,
                baselines=baselines,runs=[],
                split_hashes={n:hashlib.sha256(json.dumps(r,separators=(',',':')).encode()).hexdigest() for n,r in splits.items()},
                source_sha256={n:hashlib.sha256((ROOT/'experiments'/n).read_bytes()).hexdigest() for n in ('train_associative_recall.py','verify_decoder.py')},
                protocol_sha256=hashlib.sha256((ROOT/'docs/associative-recall-protocol.md').read_bytes()).hexdigest())
    directory=ROOT/'results/associative_recall_checkpoints'
    directory.mkdir(exist_ok=True)
    for seed in (11,29,47,73,101):
        torch.manual_seed(seed)
        model=TinyDecoder(13,16,2,32).double()
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        generator=torch.Generator().manual_seed(seed+1000)
        curve=[dict(step=0,**evaluate(model,data['validation']))]
        start=time.perf_counter()
        for step in range(1,3001):
            x,y=data['train']
            indices=torch.randint(len(x),(64,),generator=generator)
            optimizer.zero_grad()
            loss=F.cross_entropy(model(x[indices])[:,-1],y[indices])
            assert torch.isfinite(loss), (seed,step)
            loss.backward()
            optimizer.step()
            if step%500==0:
                observed=evaluate(model,data['validation'])
                curve.append(dict(step=step,**observed))
                print(json.dumps(dict(seed=seed,step=step,validation=observed)),flush=True)
        checkpoint=directory/f'seed_{seed}.pt'
        torch.save(model.state_dict(),checkpoint)
        result=dict(seed=seed,parameters=sum(p.numel() for p in model.parameters()),
                    seconds_including_validation=time.perf_counter()-start,examples_processed=192000,tokens_processed=1536000,
                    validation_curve=curve,metrics={n:evaluate(model,d) for n,d in data.items()},
                    checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        report['runs'].append(result)
        (ROOT/'results/associative_recall_training.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(seed=seed,final=result['metrics'])),flush=True)
    report['summary']={name:dict(mean_accuracy=statistics.mean(r['metrics'][name]['accuracy'] for r in report['runs']),
                               sd_accuracy=statistics.stdev(r['metrics'][name]['accuracy'] for r in report['runs']),
                               mean_loss=statistics.mean(r['metrics'][name]['loss'] for r in report['runs']),
                               all_at_least_90=all(r['metrics'][name]['accuracy']>=.9 for r in report['runs'])) for name in data}
    (ROOT/'results/associative_recall_training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
    main()
