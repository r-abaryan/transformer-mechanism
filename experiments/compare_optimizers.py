"""Matched-initialization/data fixed-recipe SGD versus Adam diagnostic."""
import hashlib
import io
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F
from train_associative_recall import datasets,tensors,evaluate
from verify_decoder import TinyDecoder

ROOT=Path(__file__).resolve().parents[1]


def state_hash(model):
    # Canonical numeric representation rather than serialization metadata.
    return hashlib.sha256(json.dumps({k:v.detach().tolist() for k,v in model.state_dict().items()},
                                    separators=(',',':')).encode()).hexdigest()


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    splits=datasets()
    previous=json.loads((ROOT/'results/associative_recall_training.json').read_text())
    data={n:tensors(splits[n]) for n in ('train','validation')}
    for n in data:
        assert hashlib.sha256(json.dumps(splits[n],separators=(',',':')).encode()).hexdigest()==previous['split_hashes'][n]
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',runs=[],steps=3000,batch=64,
                split_hashes={n:previous['split_hashes'][n] for n in data},
                recipes=dict(sgd=dict(lr=.1,momentum=0,weight_decay=0),
                             adam=dict(lr=.001,betas=[.9,.999],eps=1e-8,weight_decay=0)),
                protocol_sha256=hashlib.sha256((ROOT/'docs/optimizer-comparison-protocol.md').read_bytes()).hexdigest(),
                source_sha256={n:hashlib.sha256((ROOT/'experiments'/n).read_bytes()).hexdigest() for n in
                               ('compare_optimizers.py','train_associative_recall.py','verify_decoder.py')})
    directory=ROOT/'results/optimizer_comparison_checkpoints'
    directory.mkdir(exist_ok=True)
    for seed in (11,29,47,73,101):
        seed_results=[]
        for recipe in ('sgd','adam'):
            torch.manual_seed(seed)
            model=TinyDecoder(13,16,2,32).double()
            initial_hash=state_hash(model)
            optimizer=(torch.optim.SGD(model.parameters(),lr=.1) if recipe=='sgd' else
                       torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0))
            generator=torch.Generator().manual_seed(seed+1000)
            batch_hash=hashlib.sha256()
            curve=[dict(step=0,**{n:evaluate(model,d) for n,d in data.items()})]
            start=time.perf_counter()
            x,y=data['train']
            for step in range(1,3001):
                indices=torch.randint(len(x),(64,),generator=generator)
                batch_hash.update(json.dumps(indices.tolist(),separators=(',',':')).encode())
                optimizer.zero_grad()
                loss=F.cross_entropy(model(x[indices])[:,-1],y[indices])
                assert torch.isfinite(loss),(seed,recipe,step,'loss')
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),(seed,recipe,step,'gradient')
                optimizer.step()
                if step%500==0:
                    scores={n:evaluate(model,d) for n,d in data.items()}
                    curve.append(dict(step=step,**scores))
                    print(json.dumps(dict(seed=seed,recipe=recipe,step=step,validation=scores['validation'])),flush=True)
            final={n:evaluate(model,d) for n,d in data.items()}
            checkpoint=directory/f'{recipe}_{seed}.pt'
            torch.save(model.state_dict(),checkpoint)
            result=dict(seed=seed,recipe=recipe,initialization_sha256=initial_hash,minibatches_sha256=batch_hash.hexdigest(),
                        metrics=final,curve=curve,seconds_including_evaluation=time.perf_counter()-start,
                        parameters=sum(p.numel() for p in model.parameters()),examples_processed=192000,tokens_processed=1536000,
                        checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            if recipe=='sgd':
                original=next(r for r in previous['runs'] if r['seed']==seed)
                error=max(abs(final[n][metric]-original['metrics'][n][metric]) for n in data for metric in ('accuracy','loss'))
                assert error<1e-10,(seed,error)
                result['prior_reproduction_error']=error
            seed_results.append(result)
            report['runs'].append(result)
            (ROOT/'results/optimizer_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
        assert seed_results[0]['initialization_sha256']==seed_results[1]['initialization_sha256']
        assert seed_results[0]['minibatches_sha256']==seed_results[1]['minibatches_sha256']
    gains=[]
    for seed in (11,29,47,73,101):
        pair={r['recipe']:r for r in report['runs'] if r['seed']==seed}
        gains.append(pair['adam']['metrics']['validation']['accuracy']-pair['sgd']['metrics']['validation']['accuracy'])
    adam=[r for r in report['runs'] if r['recipe']=='adam']
    report['criteria']=dict(adam_train_all_90=all(r['metrics']['train']['accuracy']>=.9 for r in adam),
                            adam_validation_all_90=all(r['metrics']['validation']['accuracy']>=.9 for r in adam),
                            mean_paired_validation_gain=statistics.mean(gains),sd_paired_validation_gain=statistics.stdev(gains),
                            mean_gain_at_least_5_points=statistics.mean(gains)>=.05)
    (ROOT/'results/optimizer_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['criteria']),flush=True)


if __name__=='__main__':
    main()
